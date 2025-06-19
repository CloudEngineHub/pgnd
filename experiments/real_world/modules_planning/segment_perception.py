from pathlib import Path
import os
import time
import numpy as np
import torch
import cv2
import open3d as o3d
from threadpoolctl import threadpool_limits
import multiprocess as mp
from functools import partial
from PIL import Image
import supervision as sv

from pgnd.utils import get_root
root: Path = get_root(__file__)

from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.pcd_utils import depth2fgpcd


def get_mask_raw(depth, intr, extr, bbox, depth_threshold=[0, 2]):
    points = depth2fgpcd(depth, intr).reshape(-1, 3)
    mask = np.logical_and((depth > depth_threshold[0]), (depth < depth_threshold[1]))  # (H, W)

    points = (np.linalg.inv(extr) @ np.concatenate([points, np.ones((points.shape[0], 1)).astype(np.float32)], axis=1).T).T[:, :3]  # (N, 3)
    mask_bbox = np.logical_and(
        np.logical_and(points[:, 0] > bbox[0][0], points[:, 0] < bbox[0][1]),
        np.logical_and(points[:, 1] > bbox[1][0], points[:, 1] < bbox[1][1])
    )  # does not include z axis
    mask_bbox = mask_bbox.reshape(depth.shape[0], depth.shape[1])
    mask = np.logical_and(mask, mask_bbox)
    return mask


def segment_process_func(cameras_output, intrs, extrs, text_prompts, processor, grounding_model, image_predictor, bbox, device, show_annotation=True):
    colors_list = []
    depths_list = []
    pts_list = []
    for ck, cv in cameras_output.items():

        image = cv["color"].copy()
        depth = cv["depth"].copy() / 1000.0
        image = Image.fromarray(image)

        # ground
        inputs = processor(images=image, text=text_prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.325,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        input_boxes = results[0]["boxes"].cpu().numpy()
        objects = results[0]["labels"]

        depth_mask = get_mask_raw(depth, intrs[ck], extrs[ck], bbox)

        multi_objs = False
        if len(objects) > 1:
            objects_masked = []
            input_boxes_masked = []
            if intrs is None or extrs is None:
                print("No camera intrinsics and extrinsics provided")
                return {
                    "color": [],
                    "depth": [],
                    "pts": [],
                }
            for i, obj in enumerate(objects):
                if obj == '':
                    continue
                box = input_boxes[i].astype(int)
                if (box[3] - box[1]) * (box[2] - box[0]) > 500 * 400:
                    continue
                depth_mask_box = depth_mask[box[1]:box[3], box[0]:box[2]]
                if depth_mask_box.sum() > 0:
                    objects_masked.append(obj)
                    input_boxes_masked.append(box)
            objects = objects_masked
            input_boxes = input_boxes_masked
            if len(objects) == 0:
                print("No objects detected")
                return {
                    "color": [],
                    "depth": [],
                    "pts": [],
                }
            elif len(objects) > 1:
                multi_objs = True

        image_predictor.set_image(np.array(image.convert("RGB")))
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        if masks.ndim == 3:
            pass
        elif masks.ndim == 4:
            assert multi_objs
            masks = masks.squeeze(1)
        masks = masks.astype(bool)

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(objects, start=1)}
        object_ids = np.arange(1, len(objects) + 1)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        if show_annotation:
            annotated_frame = box_annotator.annotate(scene=np.array(image).astype(np.uint8), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            colors_list.append(annotated_frame)
        else:
            colors_list.append(np.array(image))

        depths_list.append(cv["depth"].copy())

        masks = np.logical_or.reduce(masks, axis=0, keepdims=True)
        masks = np.logical_and(masks, depth_mask)
        masks = masks.reshape(-1)
        assert masks.shape[0] == depth.shape[0] * depth.shape[1]
        points = depth2fgpcd(depth, intrs[ck]).reshape(-1, 3)
        points = (np.linalg.inv(extrs[ck]) @ np.concatenate([points, np.ones((points.shape[0], 1)).astype(np.float32)], axis=1).T).T[:, :3]  # (N, 3)
        points = points[masks]
        pts_list.append(points)

    return {
        "color": colors_list,
        "depth": depths_list,
        "pts": pts_list,
    }


class SegmentPerception(mp.Process):

    def __init__(
        self,
        realsense: MultiRealsense | SingleRealsense, 
        capture_fps, 
        record_fps, 
        record_time,
        exp_name=None,
        bbox=None,
        data_dir="data",
        text_prompts="white cotton rope.",
        show_annotation=True,
        device=None,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose

        self.capture_fps = capture_fps
        self.record_fps = record_fps
        self.record_time = record_time
        self.exp_name = exp_name
        self.data_dir = data_dir
        self.bbox = bbox

        self.text_prompts = text_prompts
        self.show_annotation = show_annotation

        if self.exp_name is None:
            assert self.record_fps == 0

        self.realsense = realsense
        self.perception_q = mp.Queue(maxsize=1)

        self.num_cam = len(realsense.cameras.keys())
        self.alive = mp.Value('b', False)
        self.record_restart = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)
        self.do_process = mp.Value('b', True)

        self.intrs = mp.Array('d', [0.0] * 9 * self.num_cam)
        self.extrs = mp.Array('d', [0.0] * 16 * self.num_cam)

    def log(self, msg):
        if self.verbose:
            print(f"\033[92m{self.name}: {msg}\033[0m")

    @property
    def can_record(self):
        return self.record_fps != 0
    
    def update_intrinsics(self, intrs):
        self.intrs[:] = intrs.flatten()
    
    def update_extrinsics(self, extrs):
        self.extrs[:] = extrs.flatten()

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        realsense = self.realsense

        # i = self.index
        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time

        cameras_output = None
        recording_frame = float("inf")  # local record step index (since current record start), record fps
        record_start_frame = 0  # global step index (since process start), capture fps
        is_recording = False  # recording state flag
        timestamps_f = None

        checkpoint = str(root.parent / "weights/sam2/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        
        process_func = partial(
            segment_process_func,
            text_prompts=self.text_prompts,
            processor=processor,
            grounding_model=grounding_model,
            image_predictor=image_predictor,
            bbox=self.bbox,
            device=device,
            show_annotation=self.show_annotation,
        )

        while self.alive.value:
            try: 
                if not self.do_process.value:
                    if not self.perception_q.empty():
                        self.perception_q.get()
                    time.sleep(1)
                    continue
                cameras_output = realsense.get(out=cameras_output)
                get_time = time.time()
                timestamps = [cameras_output[i]['timestamp'].item() for i in range(self.num_cam)]  # type: ignore
                if is_recording and not all([abs(timestamps[i] - timestamps[i+1]) < 0.05 for i in range(self.num_cam - 1)]):
                    print(f"Captured at different timestamps: {[f'{x:.2f}' for x in timestamps]}")

                # treat captured time and record time as the same
                process_start_time = get_time

                intrs = np.frombuffer(self.intrs.get_obj()).reshape((self.num_cam, 3, 3))
                extrs = np.frombuffer(self.extrs.get_obj()).reshape((self.num_cam, 4, 4))

                if intrs.sum() == 0 or extrs.sum() == 0:
                    print("No camera intrinsics and extrinsics provided")
                    time.sleep(1)
                    continue

                process_out = process_func(cameras_output, intrs, extrs)
                self.log(f"process time: {time.time() - process_start_time}")
            
                if not self.perception_q.full():
                    self.perception_q.put(process_out)

            except BaseException as e:
                print("Perception error: ", e.with_traceback())
                break

        if self.can_record:
            if timestamps_f is not None and not timestamps_f.closed:
                timestamps_f.close()
            finish_time = time.time()
        self.stop()
        print("Perception process stopped")


    def start(self):
        self.alive.value = True
        super().start()

    def stop(self):
        self.alive.value = False
        self.perception_q.close()
    
    def set_record_start(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_restart.value == False
        else:
            self.record_restart.value = True
            print("record restart cmd received")

    def set_record_stop(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_stop.value == False
        else:
            self.record_stop.value = True
            print("record stop cmd received")

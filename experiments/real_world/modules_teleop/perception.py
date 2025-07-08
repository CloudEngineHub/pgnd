from pathlib import Path
import sys
import numpy as np
import torch
import open3d as o3d
import cv2
from PIL import Image

import pgnd
from pgnd.utils import get_root
root: Path = get_root(__file__)

# groundingdino
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# sam
from segment_anything import SamPredictor, sam_model_registry

from utils.pcd_utils import depth2fgpcd
from utils.env_utils import get_bounding_box


class PerceptionModule:

    def __init__(self, vis_path, device="cuda:0", load_model=True):
        self.device = device
        self.vis_path = vis_path
        self.det_model = None
        self.sam_model = None
        if load_model:
            self.load_model()
    
    def load_model(self):
        if self.det_model is not None:
            print("Model already loaded")
            return
        device = self.device
        det_model = build_model(SLConfig.fromfile(
            root / '../third-party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'))
        checkpoint = torch.load(root / '../weights/groundingdino_swinb_cogcoor.pth', map_location="cpu")
        det_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        det_model.eval()
        det_model = det_model.to(device)

        sam = sam_model_registry["default"](checkpoint=root / '../weights/sam_vit_h_4b8939.pth')
        sam_model = SamPredictor(sam)
        sam_model.model = sam_model.model.to(device)

        self.det_model = det_model
        self.sam_model = sam_model
    
    def del_model(self):
        del self.det_model
        torch.cuda.empty_cache()
        del self.sam_model
        torch.cuda.empty_cache()
        self.det_model = None
        self.sam_model = None

    def detect(self, image, captions, box_thresholds):  # captions: list
        image = Image.fromarray(image)
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = caption.strip()
            if not caption.endswith("."):
                caption = caption + "."
            captions[i] = caption
        num_captions = len(captions)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image, None)  # 3, h, w

        image_tensor = image_tensor[None].repeat(num_captions, 1, 1, 1).to(self.device)

        with torch.no_grad():
            outputs = self.det_model(image_tensor, captions=captions)
        logits = outputs["pred_logits"].sigmoid()  # (num_captions, nq, 256)
        boxes = outputs["pred_boxes"]  # (num_captions, nq, 4)

        # filter output
        if isinstance(box_thresholds, list):
            filt_mask = logits.max(dim=2)[0] > torch.tensor(box_thresholds).to(device=self.device, dtype=logits.dtype)[:, None]
        else:
            filt_mask = logits.max(dim=2)[0] > box_thresholds
        labels = torch.ones((*logits.shape[:2], 1)) * torch.arange(logits.shape[0])[:, None, None]  # (num_captions, nq, 1)
        labels = labels.to(device=self.device, dtype=logits.dtype)  # (num_captions, nq, 1)
        logits = logits[filt_mask] # num_filt, 256
        boxes = boxes[filt_mask] # num_filt, 4
        labels = labels[filt_mask].reshape(-1).to(torch.int64) # num_filt,
        scores = logits.max(dim=1)[0] # num_filt,

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {captions[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
        return boxes, scores, labels


    def segment(self, image, boxes, scores, labels, text_prompts):
        # load sam model
        self.sam_model.set_image(image)

        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = self.sam_model.transform.apply_boxes_torch(boxes, image.shape[:2]), # (n_detection, 4)
            multimask_output = False,
        )

        masks = masks[:, 0, :, :] # (n_detection, H, W)
        text_labels = []
        for category in range(len(text_prompts)):
            text_labels = text_labels + [text_prompts[category].rstrip('.')] * (labels == category).sum().item()
        
        # remove masks where IoU are large
        num_masks = masks.shape[0]
        to_remove = []
        for i in range(num_masks):
            for j in range(i+1, num_masks):
                IoU = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
                if IoU > 0.9:
                    if scores[i].item() > scores[j].item():
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
        to_remove = np.unique(to_remove)
        to_keep = np.setdiff1d(np.arange(num_masks), to_remove)
        to_keep = torch.from_numpy(to_keep).to(device=self.device, dtype=torch.int64)
        masks = masks[to_keep]
        text_labels = [text_labels[i] for i in to_keep]
        
        aggr_mask = torch.zeros(masks[0].shape).to(device=self.device, dtype=torch.uint8)
        for obj_i in range(masks.shape[0]):
            aggr_mask[masks[obj_i]] = obj_i + 1

        # masks: (n_detection, H, W)
        # aggr_mask: (H, W)
        return (masks, aggr_mask, text_labels), (boxes, scores, labels)


    def get_mask(self, img, depth, intr, extr, bbox=None, depth_threshold=[0, 2], obj_names=[]):
        obj_list = ['table'] + [obj for obj in obj_names]
        text_prompts = [f"{obj}" for obj in obj_list]
        if bbox is None:
            bbox = get_bounding_box()

        points = depth2fgpcd(depth, intr).reshape(-1, 3)
        mask = np.logical_and((depth > depth_threshold[0]), (depth < depth_threshold[1]))  # (H, W)

        points = (np.linalg.inv(extr) @ np.concatenate([points, np.ones((points.shape[0], 1)).astype(np.float32)], axis=1).T).T[:, :3]  # (N, 3)
        mask_bbox = np.logical_and(
            np.logical_and(points[:, 0] > bbox[0][0], points[:, 0] < bbox[0][1]),
            np.logical_and(points[:, 1] > bbox[1][0], points[:, 1] < bbox[1][1])
        )  # does not include z axis
        mask_bbox = mask_bbox.reshape(depth.shape[0], depth.shape[1])
        mask = np.logical_and(mask, mask_bbox)

        # detect and segment
        assert len(obj_names) > 0
        boxes, scores, labels = self.detect(img, text_prompts, box_thresholds=0.5)
        H, W = img.shape[0], img.shape[1]
        boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=self.device, dtype=boxes.dtype)
        boxes[:, :2] -= boxes[:, 2:] / 2  # xywh to xyxy
        boxes[:, 2:] += boxes[:, :2]  # xywh to xyxy
        segmentation_results, _ = self.segment(img, boxes, scores, labels, text_prompts)
        masks, _, text_labels = segmentation_results
        masks = masks.detach().cpu().numpy()

        mask_objs = np.zeros(masks[0].shape, dtype=np.uint8)
        for obj_i in range(masks.shape[0]):
            if text_labels[obj_i] in obj_names:
                mask_objs = np.logical_or(mask_objs, masks[obj_i])
        for obj_i in range(masks.shape[0]):
            if text_labels[obj_i] == 'table':
                mask_objs = np.logical_and(mask_objs, ~masks[obj_i])
        return mask_objs


    def get_mask_raw(self, depth, intr, extr, bbox=None, depth_threshold=[0, 2]):
        if bbox is None:
            bbox = get_bounding_box()

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

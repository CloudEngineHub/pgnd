from pathlib import Path
import os
import time
import numpy as np
import cv2
import open3d as o3d
from threadpoolctl import threadpool_limits
import multiprocess as mp
import threading
from threading import Lock

from pgnd.utils import get_root
root: Path = get_root(__file__)

from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense


class Recorder(mp.Process):

    def __init__(
        self,
        realsense: MultiRealsense | SingleRealsense, 
        capture_fps, 
        record_fps, 
        record_time, 
        process_func,
        exp_name=None,
        data_dir="data",
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose

        self.capture_fps = capture_fps
        self.record_fps = record_fps
        self.record_time = record_time
        self.exp_name = exp_name
        self.data_dir = data_dir

        if self.exp_name is None:
            assert self.record_fps == 0

        self.realsense = realsense
        self.recorder_q = mp.Queue(maxsize=1)

        self.process_func = process_func
        self.num_cam = len(realsense.cameras.keys())
        self.alive = mp.Value('b', False)
        self.record_restart = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)

    def log(self, msg):
        if self.verbose:
            print(f"\033[92m{self.name}: {msg}\033[0m")

    @property
    def can_record(self):
        return self.record_fps != 0

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        realsense = self.realsense

        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time

        cameras_output = None
        recording_frame = float("inf")  # local record step index (since current record start), record fps
        record_start_frame = 0  # global step index (since process start), capture fps
        is_recording = False  # recording state flag
        timestamps_f = None

        while self.alive.value:
            try: 
                cameras_output = realsense.get(out=cameras_output)
                get_time = time.time()
                timestamps = [cameras_output[i]['timestamp'].item() for i in range(self.num_cam)]
                if is_recording and not all([abs(timestamps[i] - timestamps[i+1]) < 0.05 for i in range(self.num_cam - 1)]):
                    print(f"Captured at different timestamps: {[f'{x:.2f}' for x in timestamps]}")

                # treat captured time and record time as the same
                process_start_time = get_time
                process_out = self.process_func(cameras_output) if self.process_func is not None else cameras_output
                self.log(f"process time: {time.time() - process_start_time}")
            
                if not self.recorder_q.full():
                    self.recorder_q.put(process_out)             

                if self.can_record:
                    # recording state machine:
                    #           ---restart_cmd--->
                    # record                           not_record
                    #       <-- stop_cmd / timeover ----
                    if not is_recording and self.record_restart.value == True:
                        self.record_restart.value = False

                        recording_frame = 0
                        record_start_time = get_time
                        record_start_frame = cameras_output[0]['step_idx'].item()
                        
                        record_dir = root / "log" / self.data_dir / self.exp_name / f"{record_start_time:.0f}"
                        os.makedirs(record_dir, exist_ok=True)
                        timestamps_f = open(f'{record_dir}/timestamps.txt', 'a')
                        
                        for i in range(self.num_cam):
                            os.makedirs(f'{record_dir}/camera_{i}/rgb', exist_ok=True)
                            os.makedirs(f'{record_dir}/camera_{i}/depth', exist_ok=True)
                        is_recording = True
                    
                    elif is_recording and (
                        self.record_stop.value == True or 
                        (recording_frame >= record_time * record_fps)
                    ):
                        finish_time = get_time
                        print(f"is_recording {is_recording}, self.record_stop.value {self.record_stop.value}, recording time {recording_frame}, max recording time {record_time} * {record_fps}")
                        print(f"total time: {finish_time - record_start_time}")
                        print(f"fps: {recording_frame / (finish_time - record_start_time)}")
                        is_recording = False
                    
                        timestamps_f.close()
                        self.record_restart.value = False
                        self.record_stop.value = False
                    else:
                        self.record_restart.value = False
                        self.record_stop.value = False

                    # record the frame according to the record_fps
                    if is_recording and cameras_output[0]['step_idx'].item() >= (recording_frame * (capture_fps // record_fps) + record_start_frame):
                        timestamps_f.write(' '.join(
                            [str(cameras_output[i]['step_idx'].item()) for i in range(self.num_cam)] + 
                            [str(np.round(cameras_output[i]['timestamp'].item() - record_start_time, 3)) for i in range(self.num_cam)] + 
                            [str(cameras_output[i]['timestamp'].item()) for i in range(self.num_cam)]
                        ) + '\n')
                        timestamps_f.flush()
                        for i in range(self.num_cam):
                            cv2.imwrite(f'{record_dir}/camera_{i}/rgb/{recording_frame:06}.jpg', cameras_output[i]['color'])
                            cv2.imwrite(f'{record_dir}/camera_{i}/depth/{recording_frame:06}.png', cameras_output[i]['depth'])
                        recording_frame += 1

            except BaseException as e:
                print("Recorder error: ", e)
                break

        if self.can_record:
            if timestamps_f is not None and not timestamps_f.closed:
                timestamps_f.close()
            finish_time = time.time()

        self.stop()
        print("Recorder process stopped")


    def start(self):
        self.alive.value = True
        super().start()

    def stop(self):
        self.alive.value = False
        self.recorder_q.close()
    
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

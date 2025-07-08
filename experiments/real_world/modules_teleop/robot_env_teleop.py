from typing import Callable, List
from enum import Enum
import numpy as np
import multiprocess as mp
import time
import threading
import cv2
import pygame
import os
import pickle
import transforms3d
import subprocess
from pynput import keyboard
from pathlib import Path
from copy import deepcopy

from nclaw.utils import get_root, mkdir
root: Path = get_root(__file__)

from modules_planning.xarm_controller import XarmController
from modules_teleop.recorder import Recorder
from modules_teleop.teleop_keyboard import KeyboardTeleop
from gello.teleop_gello import GelloTeleop
from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense


class EnvEnum(Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2
    VERBOSE = 3

class RobotTeleopEnv(mp.Process):

    def __init__(
        self, 
        mode: str = '2D', # 2D or 3D
        exp_name: str = "recording",

        realsense: MultiRealsense | SingleRealsense | None = None,
        shm_manager: mp.managers.SharedMemoryManager | None = None,
        serial_numbers: list[str] | None= None,
        resolution: tuple[int, int] = (1280, 720),
        capture_fps: int = 30,
        enable_depth: bool = True,
        enable_color: bool = True,

        recorder: Recorder | None = None,
        record_fps: int | None = 0,
        record_time: float | None = 60 * 10,  # 10 minutes
        record_process_func: Callable | None = lambda x: x,  # default is identity function

        use_robot: bool = False,
        use_gello: bool = False,

        robot_ip: str | None = '192.168.1.196',
        gripper_enable: bool = False,
        calibrate_result_dir: Path = root / "log" / "latest_calibration",
        data_dir: Path = "data",
        debug: bool | int | None = False,
        
        bimanual: bool = False,
        bimanual_robot_ip: List[str] | None = ['192.168.1.196', '192.168.1.224'],

    ) -> None:

        # Debug level
        self.debug = 0 if debug is None else (2 if debug is True else debug)

        self.mode = mode
        self.exp_name = exp_name
        self.data_dir = data_dir

        self.bimanual = bimanual

        self.capture_fps = capture_fps
        self.record_fps = record_fps

        self.state = mp.Manager().dict()  # should be main explict exposed variable to the child class / process

        # Realsense camera setup
        # camera is always required for real env
        if realsense is not None:
            assert isinstance(realsense, MultiRealsense) or isinstance(realsense, SingleRealsense)
            self.realsense = realsense
            self.serial_numbers = list(self.realsense.cameras.keys())
        else:
            self.realsense = MultiRealsense(
                shm_manager=shm_manager,
                serial_numbers=serial_numbers,
                resolution=resolution,
                capture_fps=capture_fps,
                enable_depth=enable_depth,
                enable_color=enable_color,
                verbose=self.debug > EnvEnum.VERBOSE.value
            )
            self.serial_numbers = list(self.realsense.cameras.keys())
    
        # auto or manual
        # self.realsense.set_exposure(exposure=None)
        # self.realsense.set_white_balance(white_balance=None)
        self.realsense.set_exposure(exposure=100, gain=60)  # 100: bright, 60: dark
        self.realsense.set_white_balance(3800)

        # base calibration
        self.calibrate_result_dir = calibrate_result_dir
        with open(f'{self.calibrate_result_dir}/base.pkl', 'rb') as f:
            base = pickle.load(f)
        if self.bimanual:
            R_leftbase2board = base['R_leftbase2world']
            t_leftbase2board = base['t_leftbase2world']
            R_rightbase2board = base['R_rightbase2world']
            t_rightbase2board = base['t_rightbase2world']
            leftbase2world_mat = np.eye(4)
            leftbase2world_mat[:3, :3] = R_leftbase2board
            leftbase2world_mat[:3, 3] = t_leftbase2board
            self.state["b2w_l"] = leftbase2world_mat
            rightbase2world_mat = np.eye(4)
            rightbase2world_mat[:3, :3] = R_rightbase2board
            rightbase2world_mat[:3, 3] = t_rightbase2board
            self.state["b2w_r"] = rightbase2world_mat
        else:
            R_base2board = base['R_base2world']
            t_base2board = base['t_base2world']
            base2world_mat = np.eye(4)
            base2world_mat[:3, :3] = R_base2board
            base2world_mat[:3, 3] = t_base2board
            self.state["b2w"] = base2world_mat

        # camera calibration
        extr_list = []
        with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'rb') as f:
            rvecs = pickle.load(f)
        with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'rb') as f:
            tvecs = pickle.load(f)
        for i in range(len(self.serial_numbers)):
            device = self.serial_numbers[i]
            R_world2cam = cv2.Rodrigues(rvecs[device])[0]
            t_world2cam = tvecs[device][:, 0]
            extr_mat = np.eye(4)
            extr_mat[:3, :3] = R_world2cam
            extr_mat[:3, 3] = t_world2cam
            extr_list.append(extr_mat)
        self.state["extr"] = np.stack(extr_list)

        # save calibration
        mkdir(root / "log" / self.data_dir / self.exp_name / "calibration", overwrite=False, resume=False)
        subprocess.run(f'cp -r {self.calibrate_result_dir}/* {str(root)}/log/{self.data_dir}/{self.exp_name}/calibration', shell=True)

        # Recorder setup
        if recorder is not None:
            assert isinstance(recorder, Recorder)
            self.recorder = recorder
        else:
            self.recorder = Recorder(
                realsense=self.realsense,
                capture_fps=self.realsense.capture_fps,  # mush be the same as realsense capture fps 
                record_fps=record_fps,
                record_time=record_time,
                process_func=record_process_func,
                exp_name=exp_name,
                data_dir=data_dir,
                verbose=self.debug > EnvEnum.VERBOSE.value)

        # Robot setup
        self.use_robot = use_robot
        self.use_gello = use_gello
        if use_robot:
            if bimanual:
                self.left_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[0],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=0,
                    verbose=False,
                )
                self.right_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[1],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=1,
                    verbose=False,
                )
                self.xarm_controller = None
            else:
                self.xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip,
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=-1,
                    verbose=False,
                )
                self.left_xarm_controller = None
                self.right_xarm_controller = None
        else:
            self.left_xarm_controller = None
            self.right_xarm_controller = None
            self.xarm_controller = None

        # subprocess can only start a process object created by current process
        self._real_alive = mp.Value('b', False)

        self.start_time = 0
        mp.Process.__init__(self)
        self._alive = mp.Value('b', False)

        # pygame
        # Initialize a separate Pygame window for image display
        img_w, img_h = 848, 480
        col_num = 2
        self.screen_width, self.screen_height = img_w * col_num, img_h * len(self.realsense.serial_numbers)
        self.image_window = None

        # Shared memory for image data
        self.image_data = mp.Array('B', np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8).flatten())

        # robot eef
        self.eef_point = np.array([[0.0, 0.0, 0.175]])  # the eef point in the gripper frame

    def real_start(self, start_time) -> None:
        self._real_alive.value = True
        print("starting real env")
        
        # Realsense camera setup
        self.realsense.start()
        self.realsense.restart_put(start_time + 1)
        time.sleep(2)

        # Recorder setup
        if self.recorder is not None:
            self.recorder.start()
    
        # Robot setup
        if self.use_robot:
            if self.bimanual:
                self.left_xarm_controller.start()
                self.right_xarm_controller.start()
            else:
                self.xarm_controller.start()
        
        while not self.real_alive:
            self._real_alive.value = True
            print(".", end="")
            time.sleep(0.5)
        
        # get intrinsics
        intrs = self.get_intrinsics()
        intrs = np.array(intrs)
        np.save(root / "log" / self.data_dir / self.exp_name / "calibration" / "intrinsics.npy", intrs)
        
        print("real env started")

        self.update_real_state_t = threading.Thread(name="update_real_state", target=self.update_real_state)
        self.update_real_state_t.start()

    def real_stop(self, wait=False) -> None:
        self._real_alive.value = False
        if self.use_robot:
            if self.bimanual and self.left_xarm_controller.is_controller_alive:
                self.left_xarm_controller.stop()
            if self.bimanual and self.right_xarm_controller.is_controller_alive:
                self.right_xarm_controller.stop()
            if not self.bimanual and self.xarm_controller.is_controller_alive:
                self.xarm_controller.stop()
        if self.recorder is not None and self.recorder.alive.value:
            self.recorder.stop()
        self.realsense.stop(wait=False)

        self.image_display_thread.join()
        self.update_real_state_t.join()
        print("real env stopped")

    @property
    def real_alive(self) -> bool:
        alive = self._real_alive.value
        if self.recorder is not None:
            alive = alive and self.recorder.alive.value
        if self.use_robot:
            controller_alive = \
                (self.bimanual and self.left_xarm_controller.is_controller_alive and self.right_xarm_controller.is_controller_alive) \
                or (not self.bimanual and self.xarm_controller.is_controller_alive)
            alive = alive and controller_alive
        self._real_alive.value = alive
        return self._real_alive.value

    def _update_recorder(self) -> None:
        if self.recorder.alive.value:
            if not self.recorder.recorder_q.empty():
                self.state["recorder_out"] = {
                    "time": time.time(),
                    "value": self.recorder.recorder_q.get()
                }
        return

    def _update_robot(self) -> None:
        if self.bimanual:
            if self.left_xarm_controller.is_controller_alive and self.right_xarm_controller.is_controller_alive:
                if not self.left_xarm_controller.cur_trans_q.empty() and not self.right_xarm_controller.cur_trans_q.empty():
                    self.state["robot_out"] = {
                        "time": time.time(),
                        "left_value": self.left_xarm_controller.cur_trans_q.get(),
                        "right_value": self.right_xarm_controller.cur_trans_q.get()
                    }
                if not self.left_xarm_controller.cur_gripper_q.empty() and not self.right_xarm_controller.cur_trans_q.empty():
                    self.state["gripper_out"] = {
                        "time": time.time(),
                        "left_value": self.left_xarm_controller.cur_gripper_q.get(),
                        "right_value": self.right_xarm_controller.cur_gripper_q.get()
                    }
        else:
            if self.xarm_controller.is_controller_alive:
                if not self.xarm_controller.cur_trans_q.empty():
                    self.state["robot_out"] = {
                        "time": time.time(),
                        "value": self.xarm_controller.cur_trans_q.get()
                    }
                if not self.xarm_controller.cur_gripper_q.empty():
                    self.state["gripper_out"] = {
                        "time": time.time(),
                        "value": self.xarm_controller.cur_gripper_q.get()
                    }
        return

    def update_real_state(self) -> None:
        while self.real_alive:
            try:
                if self.use_robot:
                    self._update_robot()
                if self.recorder is not None:
                    self._update_recorder()
            except:
                print(f"Error in update_real_state")
                break
        print("update_real_state stopped")

    def display_image(self):
        pygame.init()
        self.image_window = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Image Display Window')
        while self._alive.value:
            # Extract image data from the shared array
            image = np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape((self.screen_height, self.screen_width, 3))
            pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))

            # Blit the image to the window
            self.image_window.blit(pygame_image, (0, 0))
            pygame.display.update()

            # Handle events (e.g., close window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()
                    pygame.quit()
                    return

            time.sleep(1 / self.realsense.capture_fps)  # 30 FPS
        print("Image display stopped")

    def start_image_display(self):
        # Start a thread for the image display loop
        self.image_display_thread = threading.Thread(name="display_image", target=self.display_image)
        self.image_display_thread.start()

    def run(self) -> None:
        if self.use_robot:
            if self.use_gello:
                teleop = GelloTeleop(bimanual=self.bimanual)
            else:
                teleop = KeyboardTeleop()
            teleop.start()
        time.sleep(1)

        robot_record_dir = root / "log" / self.data_dir / self.exp_name / "robot"
        os.makedirs(robot_record_dir, exist_ok=True)

        # initialize images
        rgbs = []
        depths = []
        resolution = self.realsense.resolution
        for i in range(len(self.realsense.serial_numbers)):
            rgbs.append(np.zeros((resolution[1], resolution[0], 3), np.uint8))
            depths.append(np.zeros((resolution[1], resolution[0]), np.uint16))

        fps = self.record_fps if self.record_fps > 0 else self.realsense.capture_fps  # visualization fps
        idx = 0
        while self.alive:
            try:
                tic = time.time()
                state = deepcopy(self.state)
                if self.bimanual:
                    b2w_l = state["b2w_l"]
                    b2w_r = state["b2w_r"]
                else:
                    b2w = state["b2w"]

                if teleop.record_start.value == True:
                    self.recorder.set_record_start()
                    teleop.record_start.value = False

                if teleop.record_stop.value == True:
                    self.recorder.set_record_stop()
                    teleop.record_stop.value = False

                idx += 1

                # update images from realsense to shared memory
                recorder_out = state.get("recorder_out", None)
                robot_out = state.get("robot_out", None)
                gripper_out = state.get("gripper_out", None)

                intrinsics = self.get_intrinsics()
                if recorder_out is not None:
                    for k, v in recorder_out['value'].items():
                        rgbs[k] = v["color"]
                        depths[k] = v["depth"]
                        intr = intrinsics[k]

                        l = 0.1
                        origin = state["extr"][k] @ np.array([0, 0, 0, 1])
                        x_axis = state["extr"][k] @ np.array([l, 0, 0, 1])
                        y_axis = state["extr"][k] @ np.array([0, l, 0, 1])
                        z_axis = state["extr"][k] @ np.array([0, 0, l, 1])
                        origin = origin[:3] / origin[2]
                        x_axis = x_axis[:3] / x_axis[2]
                        y_axis = y_axis[:3] / y_axis[2]
                        z_axis = z_axis[:3] / z_axis[2]
                        origin = intr @ origin
                        x_axis = intr @ x_axis
                        y_axis = intr @ y_axis
                        z_axis = intr @ z_axis
                        cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(x_axis[0]), int(x_axis[1])), (255, 0, 0), 2)
                        cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(y_axis[0]), int(y_axis[1])), (0, 255, 0), 2)
                        cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(z_axis[0]), int(z_axis[1])), (0, 0, 255), 2)
                        if self.use_robot:
                            eef_points = np.concatenate([self.eef_point, np.ones((self.eef_point.shape[0], 1))], axis=1)  # (n, 4)
                            eef_colors = [(0, 255, 255)]

                            eef_axis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # (3, 4)
                            eef_axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

                            if robot_out is not None:
                                assert gripper_out is not None
                                eef_points_world_vis = []
                                eef_points_vis = []
                                if self.bimanual:
                                    left_eef_world_list = []
                                    right_eef_world_list = []
                                    for val, b2w, eef_world_list in zip(["left_value", "right_value"], [b2w_l, b2w_r], [left_eef_world_list, right_eef_world_list]):
                                        e2b = robot_out[val]  # (4, 4)
                                        eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
                                        eef_points_vis.append(eef_points)
                                        eef_points_world_vis.append(eef_points_world)
                                        eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
                                        eef_world = np.concatenate([eef_points_world, eef_orientation_world], axis=0)  # (n+3, 3)
                                        eef_world_list.append(eef_world)
                                    left_eef_world = np.concatenate(left_eef_world_list, axis=0)  # (n+3, 3)
                                    right_eef_world = np.concatenate(right_eef_world_list, axis=0)  # (n+3, 3)
                                    eef_world = np.concatenate([left_eef_world, right_eef_world], axis=0)  # (2n+6, 3)
                                else:
                                    e2b = robot_out["value"]  # (4, 4)
                                    eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
                                    eef_points_vis.append(eef_points)
                                    eef_points_world_vis.append(eef_points_world)
                                    eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
                                    eef_world = np.concatenate([eef_points_world, eef_orientation_world], axis=0)  # (n+3, 3)
                                
                                # add gripper
                                if self.bimanual:
                                    left_gripper = gripper_out["left_value"]
                                    right_gripper = gripper_out["right_value"]
                                    gripper_world = np.array([left_gripper, right_gripper, 0.0])[None, :]  # (1, 3)
                                else:
                                    gripper = gripper_out["value"]
                                    gripper_world = np.array([gripper, 0.0, 0.0])[None, :]  # (1, 3)

                                eef_world = np.concatenate([eef_world, gripper_world], axis=0)  # (n+4, 3) or (2n+7, 3)
                                np.savetxt(robot_record_dir / f"{robot_out['time']:.3f}.txt", eef_world, fmt="%.6f")
                                
                                eef_points_vis = np.concatenate(eef_points_vis, axis=0)
                                eef_points_world_vis = np.concatenate(eef_points_world_vis, axis=0)
                                eef_points_world_vis = np.concatenate([eef_points_world_vis, np.ones((eef_points_world_vis.shape[0], 1))], axis=1)  # (n, 4)
                                eef_colors = eef_colors * eef_points_world_vis.shape[0]
                                
                                if self.bimanual:
                                    for point_orig, point, color, val, b2w in zip(eef_points_vis, eef_points_world_vis, eef_colors, ["left_value", "right_value"], [b2w_l, b2w_r]):
                                        e2b = robot_out[val]  # (4, 4)
                                        point = state["extr"][k] @ point
                                        point = point[:3] / point[2]
                                        point = intr @ point
                                        cv2.circle(rgbs[k], (int(point[0]), int(point[1])), 2, color, -1)
                                    
                                        # draw eef axis
                                        for axis, color in zip(eef_axis, eef_axis_colors):
                                            eef_point_axis = point_orig + 0.1 * axis
                                            eef_point_axis_world = (b2w @ e2b @ eef_point_axis).T
                                            eef_point_axis_world = state["extr"][k] @ eef_point_axis_world
                                            eef_point_axis_world = eef_point_axis_world[:3] / eef_point_axis_world[2]
                                            eef_point_axis_world = intr @ eef_point_axis_world
                                            cv2.line(rgbs[k], 
                                                (int(point[0]), int(point[1])), 
                                                (int(eef_point_axis_world[0]), int(eef_point_axis_world[1])), 
                                                color, 2)
                                else:
                                    point_orig = eef_points_vis[0]
                                    point = eef_points_world_vis[0]
                                    color = eef_colors[0]
                                    e2b = robot_out["value"]  # (4, 4)
                                    point = state["extr"][k] @ point
                                    point = point[:3] / point[2]
                                    point = intr @ point
                                    cv2.circle(rgbs[k], (int(point[0]), int(point[1])), 2, color, -1)
                                
                                    # draw eef axis
                                    for axis, color in zip(eef_axis, eef_axis_colors):
                                        eef_point_axis = point_orig + 0.1 * axis
                                        eef_point_axis_world = (b2w @ e2b @ eef_point_axis).T
                                        eef_point_axis_world = state["extr"][k] @ eef_point_axis_world
                                        eef_point_axis_world = eef_point_axis_world[:3] / eef_point_axis_world[2]
                                        eef_point_axis_world = intr @ eef_point_axis_world
                                        cv2.line(rgbs[k], 
                                            (int(point[0]), int(point[1])), 
                                            (int(eef_point_axis_world[0]), int(eef_point_axis_world[1])), 
                                            color, 2)

                row_imgs = []
                for row in range(len(self.realsense.serial_numbers)):
                    row_imgs.append(
                        np.hstack(
                            (cv2.cvtColor(rgbs[row], cv2.COLOR_BGR2RGB), 
                            cv2.applyColorMap(cv2.convertScaleAbs(depths[row], alpha=0.03), cv2.COLORMAP_JET))
                        )
                    )
                combined_img = np.vstack(row_imgs)
                combined_img = cv2.resize(combined_img, (self.screen_width,self.screen_height))
                np.copyto(
                    np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape((self.screen_height, self.screen_width, 3)), 
                    combined_img
                )

                time.sleep(max(0, 1 / fps - (time.time() - tic)))
            
            except:
                print(f"Error in robot teleop env")
                break
        
        if self.use_robot:
            teleop.stop()
        self.stop()
        print("RealEnv process stopped")

    def get_intrinsics(self):
        return self.realsense.get_intrinsics()

    def get_extrinsics(self):
        return self.state["extr"]

    @property
    def alive(self) -> bool:
        alive = self._alive.value and self.real_alive
        self._alive.value = alive
        return alive

    def start(self) -> None:
        self.start_time = time.time()
        self._alive.value = True
        self.real_start(time.time())
        self.start_image_display()
        super().start()

    def stop(self) -> None:
        self._alive.value = False
        self.real_stop()

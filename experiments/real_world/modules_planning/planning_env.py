from typing import Callable, List
from omegaconf import DictConfig, OmegaConf
from enum import Enum
import torch
import numpy as np
import multiprocess as mp
import time
import threading
import cv2
import pygame
import os
import pickle
import subprocess
from pathlib import Path
from copy import deepcopy
import open3d as o3d

from pgnd.utils import get_root, mkdir
root: Path = get_root(__file__)

from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense

from modules_planning.segment_perception import SegmentPerception
from modules_planning.planning_module import PlanningModule
from modules_planning.xarm_controller import XarmController
from modules_planning.udp_util import udpReceiver, udpSender
from modules_planning.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R
from modules_planning.common.xarm import GRIPPER_OPEN_MIN, GRIPPER_OPEN_MAX, POSITION_UPDATE_INTERVAL, COMMAND_CHECK_INTERVAL


class EnvEnum(Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2
    VERBOSE = 3


class RobotPlanningEnv(mp.Process):

    def __init__(
        self, 
        cfg: DictConfig,
        exp_root: str,
        ckpt_path: Path | str,

        realsense: MultiRealsense | SingleRealsense | None = None,
        shm_manager: mp.managers.SharedMemoryManager | None = None,
        serial_numbers: list[str] | None= None,
        resolution: tuple[int, int] = (1280, 720),
        capture_fps: int = 30,
        enable_depth: bool = True,
        enable_color: bool = True,

        record_fps: int | None = 0,
        record_time: float | None = 60 * 10,  # 10 minutes
        text_prompts: str = "white cotton rope.",
        show_annotation: bool = True,

        use_robot: bool = False,
        robot_ip: str = '192.168.1.196',
        gripper_enable: bool = False,
        calibrate_result_dir: Path = root / "log" / "latest_calibration",
        debug: bool | int | None = False,
        
        bimanual: bool = False,
        bimanual_robot_ip: List[str] = ['192.168.1.196', '192.168.1.224'],

        construct_target: bool = False,

    ) -> None:

        # Debug level
        self.debug = 0 if debug is None else (2 if debug is True else debug)

        self.cfg = cfg
        self.exp_root = Path(exp_root)

        if os.path.exists(self.exp_root.parent / "target" / "target.pcd"):
            mkdir(self.exp_root / "target", overwrite=False, resume=True)
            os.system(f"cp {self.exp_root.parent / 'target' / 'target.pcd'} {self.exp_root / 'target'}")
        else:
            assert construct_target, "target.pcd not found"
        self.construct_target = construct_target
        if construct_target:
            mkdir(self.exp_root.parent / "target", overwrite=False, resume=False)

        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.bimanual = bimanual
        if self.bimanual:
            self.bbox = np.array([[0.0, 0.7], [-0.35, 0.45 + 0.75], [-0.8, 0.0]])
        else:
            self.bbox = np.array([[0.0, 0.7], [-0.35, 0.45], [-0.8, 0.0]])
        
        self.text_prompts = text_prompts
        self.show_annotation = show_annotation

        self.capture_fps = capture_fps
        self.record_fps = record_fps

        self.state = mp.Manager().dict() # should be main explict exposed variable to the child class / process

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
    
        # manual
        self.realsense.set_exposure(exposure=60, gain=60)  # 100: bright, 60: dark
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
        mkdir(self.exp_root / "calibration", overwrite=False, resume=False)
        subprocess.run(f'cp -r {self.calibrate_result_dir}/* {str(self.exp_root)}/calibration', shell=True)

        # Perception setup
        self.perception = SegmentPerception(
            realsense=self.realsense,
            capture_fps=self.realsense.capture_fps,  # mush be the same as realsense capture fps 
            record_fps=record_fps,
            record_time=record_time,
            text_prompts=self.text_prompts,
            show_annotation=self.show_annotation,
            bbox=self.bbox,
            device=self.torch_device,
            verbose=self.debug > EnvEnum.VERBOSE.value)

        # Robot setup
        self.use_robot = use_robot
        if use_robot:
            if bimanual:
                self.left_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[0],
                    gripper_enable=gripper_enable,
                    mode='3D',
                    command_mode='cartesian',
                    robot_id=0,
                    verbose=True,
                )
                self.right_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[1],
                    gripper_enable=gripper_enable,
                    mode='3D',
                    command_mode='cartesian',
                    robot_id=1,
                    verbose=True,
                )
                self.xarm_controller = None
            else:
                self.xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip,
                    gripper_enable=gripper_enable,
                    mode='3D',
                    command_mode='cartesian',
                    verbose=True,
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

        # planning_module setup
        self.planning_module = PlanningModule(
            cfg=cfg,
            exp_root=self.exp_root,
            ckpt_path=ckpt_path,
            device=self.torch_device,
            bimanual=bimanual,
            bbox=self.bbox,
            eef_point=self.eef_point,
            debug=self.debug
        )
        self.planning_module.set_target(str(self.exp_root / "target" / "target.pcd"))
        self.command = []
        self.command_sender = None
        self.command_sender_left = None
        self.command_sender_right = None

    def real_start(self, start_time) -> None:
        self._real_alive.value = True
        print("starting real env")
        
        # Realsense camera setup
        self.realsense.start()
        self.realsense.restart_put(start_time + 1)
        time.sleep(2)

        # Perception setup
        if self.perception is not None:
            self.perception.start()
        self.perception.update_extrinsics(self.state["extr"])
    
        # Robot setup
        if self.use_robot:
            if self.bimanual:
                assert self.left_xarm_controller is not None
                assert self.right_xarm_controller is not None
                self.left_xarm_controller.start()
                self.right_xarm_controller.start()
            else:
                assert self.xarm_controller is not None
                self.xarm_controller.start()
        
        while not self.real_alive:
            self._real_alive.value = True
            print(".", end="")
            time.sleep(0.5)
        
        # get intrinsics
        intrs = self.get_intrinsics()
        intrs = np.array(intrs)
        np.save(self.exp_root / "calibration" / "intrinsics.npy", intrs)

        print("real env started")

        self.update_real_state_t = threading.Thread(name="update_real_state", target=self.update_real_state)
        self.update_real_state_t.start()

    def real_stop(self, wait=False) -> None:
        self._real_alive.value = False
        if self.use_robot:
            if self.bimanual and self.left_xarm_controller and self.left_xarm_controller.is_controller_alive:
                self.left_xarm_controller.stop()
            if self.bimanual and self.right_xarm_controller and self.right_xarm_controller.is_controller_alive:
                self.right_xarm_controller.stop()
            if not self.bimanual and self.xarm_controller and self.xarm_controller.is_controller_alive:
                self.xarm_controller.stop()
        if self.perception is not None and self.perception.alive.value:
            self.perception.stop()
        self.realsense.stop(wait=False)

        self.image_display_thread.join()
        self.update_real_state_t.join()
        print("real env stopped")

    @property
    def real_alive(self) -> bool:
        alive = self._real_alive.value
        if self.perception is not None:
            alive = alive and self.perception.alive.value
        if self.use_robot:
            controller_alive = \
                (self.bimanual and self.left_xarm_controller and self.left_xarm_controller.is_controller_alive \
                    and self.right_xarm_controller and self.right_xarm_controller.is_controller_alive) \
                or (not self.bimanual and self.xarm_controller and self.xarm_controller.is_controller_alive)
            alive = alive and controller_alive
        self._real_alive.value = alive
        return self._real_alive.value

    def _update_perception(self) -> None:
        if self.perception.alive.value and self.perception.do_process.value:
            if not self.perception.perception_q.empty():
                self.state["perception_out"] = {
                    "time": time.time(),
                    "value": self.perception.perception_q.get()
                }
        return

    def _update_robot(self) -> None:
        if self.bimanual:
            assert self.left_xarm_controller is not None
            assert self.right_xarm_controller is not None
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
            assert self.xarm_controller is not None
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
                if self.perception is not None:
                    self._update_perception()
            except BaseException as e:
                print(f"Error in update_real_state: {e.with_traceback()}")
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

        robot_record_dir = None
        if self.bimanual:
            self.command_sender_left = udpSender(port=XARM_CONTROL_PORT_L)
            self.command_sender_right = udpSender(port=XARM_CONTROL_PORT_R)
        else:
            self.command_sender = udpSender(port=XARM_CONTROL_PORT)

        # initialize images
        rgbs = []  # bgr
        depths = []
        resolution = self.realsense.resolution
        for i in range(len(self.realsense.serial_numbers)):
            rgbs.append(np.zeros((resolution[1], resolution[0], 3), np.uint8))
            depths.append(np.zeros((resolution[1], resolution[0]), np.uint16))

        fps = self.record_fps if self.record_fps and self.record_fps > 0 else self.realsense.capture_fps  # visualization fps
        idx = 0

        while self.alive:
            try:
                tic = time.time()
                state = deepcopy(self.state)
                if self.bimanual:
                    b2w_l = state["b2w_l"]
                    b2w_r = state["b2w_r"]
                    b2w = None
                else:
                    b2w = state["b2w"]
                    b2w_l = None
                    b2w_r = None

                # update images from realsense to shared memory
                perception_out = state.get("perception_out", None)
                robot_out = state.get("robot_out", None)
                gripper_out = state.get("gripper_out", None)

                # clear previous images
                if perception_out is not None:
                    self.state["perception_out"] = None

                intrinsics = self.get_intrinsics()
                state["intr"] = intrinsics
                self.state["intr"] = intrinsics
                self.perception.update_intrinsics(intrinsics)

                update_perception = True
                if update_perception and perception_out is not None:
                    print("update image display")
                    for k in range(len(perception_out['value']['color'])):
                        rgbs[k] = perception_out['value']['color'][k].copy()
                        depths[k] = perception_out['value']['depth'][k].copy()
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
                                    assert b2w_l is not None
                                    assert b2w_r is not None
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
                                    assert b2w is not None
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
                                if robot_record_dir is not None:
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
                                    point_orig, point, color, val, b2w = eef_points_vis[0], eef_points_world_vis[0], eef_colors[0], "value", b2w
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
                
                # save target
                if self.construct_target and perception_out is not None and robot_out is not None:
                    assert os.path.exists(self.exp_root.parent / "target")
                    save_dir = str(self.exp_root.parent / "target" / "target.pcd")
                    pts = perception_out['value']['pts']
                    print(f"target pts: {pts}")
                    pts = np.concatenate(pts, axis=0)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    o3d.io.write_point_cloud(save_dir, pcd)
                    
                    mkdir(self.exp_root.parent / "target" / "vis", overwrite=False, resume=True)
                    eef_points_world_vis = []
                    eef_points = np.concatenate([self.eef_point, np.ones((self.eef_point.shape[0], 1))], axis=1)  # (n, 4)
                    eef_axis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # (3, 4)
                    if self.bimanual:
                        left_eef_world_list = []
                        right_eef_world_list = []
                        assert b2w_l is not None
                        assert b2w_r is not None
                        for val, b2w, eef_world_list in zip(["left_value", "right_value"], [b2w_l, b2w_r], [left_eef_world_list, right_eef_world_list]):
                            e2b = robot_out[val]  # (4, 4)
                            eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
                            eef_points_world_vis.append(eef_points_world)
                            eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
                            eef_world = np.concatenate([eef_points_world, eef_orientation_world], axis=0)  # (n+3, 3)
                            eef_world_list.append(eef_world)
                        left_eef_world = np.concatenate(left_eef_world_list, axis=0)  # (n+3, 3)
                        right_eef_world = np.concatenate(right_eef_world_list, axis=0)  # (n+3, 3)
                        eef_world = np.concatenate([left_eef_world, right_eef_world], axis=0)  # (2n+6, 3)
                    else:
                        assert b2w is not None
                        e2b = robot_out["value"]  # (4, 4)
                        eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
                        eef_points_world_vis.append(eef_points_world)
                        eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
                        eef_world = np.concatenate([eef_points_world, eef_orientation_world], axis=0)  # (n+3, 3)
                    np.savetxt(str(self.exp_root.parent / "target" / "vis" / f"robot.txt"), eef_world, fmt="%.6f")

                    for k in range(len(perception_out['value']['color'])):
                        rgbs[k] = perception_out['value']['color'][k]
                        depths[k] = perception_out['value']['depth'][k]
                        rgb_save_dir = str(self.exp_root.parent / "target" / "vis" / f"rgb_{k}.png")
                        depth_save_dir = str(self.exp_root.parent / "target" / "vis" / f"depth_{k}.png")
                        cv2.imwrite(rgb_save_dir, rgbs[k])
                        cv2.imwrite(depth_save_dir, depths[k])
                    
                    intr_list = []
                    extr_list = []
                    for k in range(len(perception_out['value']['color'])):
                        intr = intrinsics[k]
                        extr = state["extr"][k]
                        intr_list.append(intr)
                        extr_list.append(extr)
                    np.save(str(self.exp_root.parent / "target" / "vis" / "intrinsics.npy"), np.stack(intr_list))
                    np.save(str(self.exp_root.parent / "target" / "vis" / "extrinsics.npy"), np.stack(extr_list))

                    print(f"target saved to {save_dir}")
                    time.sleep(5)
                    continue
                
                # do planning
                if perception_out is not None and robot_out is not None and gripper_out is not None:
                    if len(perception_out['value']['pts']) > 0:
                        self.perception.do_process.value = False  # pause perception
                        command = self.planning_module.get_command(state, save_dir=f'{self.exp_root}/interaction_{idx:02d}', is_first_iter=(idx == 0))
                        if self.bimanual:
                            self.command_sender_left.send(command[0])
                            self.command_sender_right.send(command[1])
                        else:
                            self.command_sender.send(command[0])
                        idx += 1
                        time.sleep(10)  # TODO execution time
                        self.perception.do_process.value = True  # resume perception
                    else:
                        print(f'no points detected in perception_out: {perception_out["value"]["pts"]}')
                else:
                    print(f'perception_out is None: {perception_out is None}', end=' ')
                    print(f'robot_out is None: {robot_out is None}', end=' ')
                    print(f'gripper_out is None: {gripper_out is None}')

                time.sleep(max(0, 1 / fps - (time.time() - tic)))

            except BaseException as e:
                print(f"Error in robot planning env: {e.with_traceback()}")
                break

        if self.bimanual:
            assert self.command_sender_left is not None
            assert self.command_sender_right is not None
            self.command_sender_left.close()
            self.command_sender_right.close()
        else:
            assert self.command_sender is not None
            self.command_sender.close()
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

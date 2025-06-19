from pathlib import Path
import open3d as o3d
from threadpoolctl import threadpool_limits
import multiprocess as mp
import threading
from threading import Lock

import os
import numpy as np
from copy import deepcopy
from functools import partial
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import transforms3d
import kornia

from pgnd.utils import get_root
root: Path = get_root(__file__)

from modules_planning.dynamics_module import DynamicsModule, fps
from utils.planning_utils import batch_chamfer_dist


class PlanningModule:

    def __init__(self, 
            cfg, 
            device,
            exp_root,
            ckpt_path,
            use_robot=False,
            bimanual=True,
            bbox=None,
            eef_point=None,
            debug=False,
        ):
        super().__init__()
        self.cfg = cfg
        self.exp_root = exp_root
        self.ckpt_path = ckpt_path
        self.bimanual = bimanual
        self.use_robot = use_robot
        self.debug = debug
        self.eef_point = eef_point
        self.torch_device = device

        assert bbox is not None
        self.bbox = torch.tensor(bbox, dtype=torch.float32, device=self.torch_device)
        
        self.repeated_action = False  # allow non-repetitive action

        self.n_look_ahead = 10  # num_steps_total
        self.n_sample = 32
        self.n_sample_chunk = 32
        self.n_chunk = np.ceil(self.n_sample / self.n_sample_chunk).astype(int)
        self.n_update_iter = 20

        self.reward_weight = 10.0

        self.target_state = torch.empty(0)

        self.state = None
        self.pts = torch.empty(0)
        self.eef_xyz = torch.empty(0)  # (n_grippers, 3)
        self.eef_rot = torch.empty(0)  # (n_grippers, 3, 3)
        self.eef_gripper = torch.empty(0)  # (n_grippers,)

        self.dynamics_module = DynamicsModule(cfg, exp_root=exp_root, ckpt_path=ckpt_path, batch_size=self.n_sample_chunk, num_steps_total=self.n_look_ahead)
        self.dynamics_module.reset_model(self.n_look_ahead)
        self.xyz_noise_level = 0.002
        self.quat_noise_level = 0.001
        self.gripper_noise_level = 0.0

    def set_target(self, pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        target_state = np.array(pcd.points)
        if len(target_state) == 0:
            print('target state is empty')
            return
        target_state = torch.tensor(target_state, dtype=torch.float32, device=self.torch_device)
        fps_idx = fps(target_state, 1000, device=self.torch_device, random_start=False)
        target_state = target_state[fps_idx]
        self.target_state = target_state.clone()
        self.dynamics_module.set_target_state(target_state)

    def model_rollout(self, act_seqs, visualize_pv=False):
        
        pts = self.pts.clone()
        n_grippers = self.cfg.sim.num_grippers

        n_sample = act_seqs.shape[0]
        eef_xyz = act_seqs[:, :, :n_grippers * 3].reshape(n_sample, self.n_look_ahead, n_grippers, 3)
        eef_rot = act_seqs[:, :, n_grippers * 3:n_grippers * (3 + 3 * 3)].reshape(n_sample, self.n_look_ahead, n_grippers, 3, 3)
        eef_gripper = act_seqs[:, :, n_grippers * (3 + 3 * 3):].reshape(n_sample, self.n_look_ahead, n_grippers, 1)

        eef_xyz_now = self.eef_xyz.clone()
        eef_rot_now = self.eef_rot.clone()
        eef_gripper_now = self.eef_gripper.clone()
        eef_xyz = torch.cat([eef_xyz_now.repeat(n_sample, 1, 1, 1), eef_xyz], dim=1)  # (n_sample, n_look_forward + 1, n_grippers, 3)
        eef_rot = torch.cat([eef_rot_now.repeat(n_sample, 1, 1, 1, 1), eef_rot], dim=1)  # (n_sample, n_look_forward + 1, n_grippers, 3, 3)
        eef_gripper = torch.cat([eef_gripper_now[:, None].repeat(n_sample, 1, 1, 1), eef_gripper], dim=1)  # (n_sample, n_look_forward + 1, n_grippers, 1)
        assert eef_xyz.shape[1] == eef_rot.shape[1] == eef_gripper.shape[1] == self.n_look_ahead + 1

        x, v = self.dynamics_module.rollout(pts, eef_xyz, eef_rot, eef_gripper, pts_his=None, visualize_pv=visualize_pv)  # (n_sample, n_look_forward, n_pts, 3)
        model_out = {
            'x': x,
        }
        return model_out


    def sample_action_seq(self, act_seq, iter_index=0):
        # get action
        n_grippers = self.cfg.sim.num_grippers
        eef_xyz = act_seq[:, :n_grippers * 3].reshape(self.n_look_ahead, n_grippers, 3)
        eef_rot = act_seq[:, n_grippers * 3:n_grippers * (3 + 3 * 3)].reshape(self.n_look_ahead, n_grippers, 3, 3)
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)
        eef_gripper = act_seq[:, n_grippers * (3 + 3 * 3):].reshape(self.n_look_ahead, n_grippers, 1)

        # add noise
        n_sample = self.n_sample_chunk
        
        if self.repeated_action:  # default: linear repeating action
            eef_xyz_delta = torch.randn((n_sample, eef_xyz.shape[1], eef_xyz.shape[2]), device=self.torch_device, dtype=torch.float32) * self.xyz_noise_level
            eef_quat_delta = torch.randn((n_sample, eef_quat.shape[1], eef_quat.shape[2]), device=self.torch_device, dtype=torch.float32) * self.quat_noise_level
            eef_gripper_delta = torch.randn((n_sample, eef_gripper.shape[1], eef_gripper.shape[2]), device=self.torch_device, dtype=torch.float32) * self.gripper_noise_level
            eef_xyz_delta = eef_xyz_delta[:, None].repeat(1, self.n_look_ahead, 1, 1)
            eef_quat_delta = eef_quat_delta[:, None].repeat(1, self.n_look_ahead, 1, 1)
            eef_gripper_delta = eef_gripper_delta[:, None].repeat(1, self.n_look_ahead, 1, 1)

        else:  # segmented repeated action
            n_parts = 4
            eef_xyz_delta_list = []
            eef_quat_delta_list = []
            eef_gripper_delta_list = []
            for p in range(n_parts):
                p_len = self.n_look_ahead // n_parts if p < n_parts - 1 else self.n_look_ahead - (n_parts - 1) * (self.n_look_ahead // n_parts)
                eef_xyz_delta = torch.randn((n_sample, eef_xyz.shape[1], eef_xyz.shape[2]), device=self.torch_device, dtype=torch.float32) * self.xyz_noise_level * (1. / (iter_index + 1) ** 1)  # TODO
                eef_quat_delta = torch.randn((n_sample, eef_quat.shape[1], eef_quat.shape[2]), device=self.torch_device, dtype=torch.float32) * self.quat_noise_level * (1. / (iter_index + 1) ** 1)
                eef_gripper_delta = torch.randn((n_sample, eef_gripper.shape[1], eef_gripper.shape[2]), device=self.torch_device, dtype=torch.float32) * self.gripper_noise_level * (1. / (iter_index + 1) ** 1)
                eef_xyz_delta = eef_xyz_delta[:, None].repeat(1, p_len, 1, 1)
                eef_quat_delta = eef_quat_delta[:, None].repeat(1, p_len, 1, 1)
                eef_gripper_delta = eef_gripper_delta[:, None].repeat(1, p_len, 1, 1)
                eef_xyz_delta_list.append(eef_xyz_delta)
                eef_quat_delta_list.append(eef_quat_delta)
                eef_gripper_delta_list.append(eef_gripper_delta)
            eef_xyz_delta = torch.cat(eef_xyz_delta_list, dim=1)
            eef_quat_delta = torch.cat(eef_quat_delta_list, dim=1)
            eef_gripper_delta = torch.cat(eef_gripper_delta_list, dim=1)

        eef_xyz_delta_cum = torch.cumsum(eef_xyz_delta, dim=1)  # (n_sample, n_look_ahead, n_grippers, 3)
        eef_quat_delta_cum = torch.cumsum(eef_quat_delta, dim=1)  # (n_sample, n_look_ahead, n_grippers, 4)
        eef_gripper_delta_cum = torch.cumsum(eef_gripper_delta, dim=1)  # (n_sample, n_look_ahead, n_grippers, 1)

        eef_xyz = eef_xyz[None] + eef_xyz_delta_cum
        eef_quat = eef_quat[None] + eef_quat_delta_cum
        eef_gripper = eef_gripper[None] + eef_gripper_delta_cum

        eef_quat = eef_quat / (torch.norm(eef_quat, dim=-1, keepdim=True) + 1e-6)  # normalize
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)

        act_seqs = torch.zeros((n_sample, self.n_look_ahead, n_grippers * (3 + 9 + 1)), device=self.torch_device, dtype=torch.float32)
        act_seqs[:, :, :n_grippers * 3] = eef_xyz.reshape(n_sample, self.n_look_ahead, -1)
        act_seqs[:, :, n_grippers * 3:n_grippers * (3 + 3 * 3)] = eef_rot.reshape(n_sample, self.n_look_ahead, -1)
        act_seqs[:, :, n_grippers * (3 + 3 * 3):] = eef_gripper.reshape(n_sample, self.n_look_ahead, -1)

        self.clip_actions(act_seqs)
        return act_seqs

    def evaluate_traj(self, model_out, act_seqs):
        target_state = self.target_state.clone()
        x = model_out['x']  # (n_sample, n_look_forward, n_pts, 3)
        chamfer_distance = batch_chamfer_dist(x[:, -1], target_state)  # (n_sample,)
        curr_chamfer_distance = batch_chamfer_dist(x[:, 0], target_state)  # (n_sample,)
        print('curr chamfer_distance:', curr_chamfer_distance.min().item(), end=' ')
        print('best chamfer_distance:', chamfer_distance.min().item())

        n_sample = self.n_sample_chunk
        n_grippers = self.cfg.sim.num_grippers
        assert act_seqs.shape[0] == n_sample
        eef_xyz = act_seqs[:, :, :n_grippers * 3].reshape(n_sample, self.n_look_ahead, n_grippers, 3)

        if eef_xyz.shape[2] == 2:
            eef_xyz_left = eef_xyz[:, :, 0]  # (n_sample, self.n_look_ahead, 3)
            eef_xyz_right = eef_xyz[:, :, 1]  # (n_sample, self.n_look_ahead, 3)
            eef_xyz_dist = torch.norm(eef_xyz_left - eef_xyz_right, dim=-1)  # (n_sample, self.n_look_ahead)
            eef_xyz_dist_penalty = (eef_xyz_dist.max(dim=1).values > 0.3).float() * 100.0  # (n_sample,)  # the smaller the better  # TODO distance threshold

            eef_height_penalty = torch.logical_or(eef_xyz_left[:, :, 2].max(dim=1).values > -0.02, eef_xyz_right[:, :, 2].max(dim=1).values > -0.02).to(torch.float32) * 100.0  # (n_sample,)  # the smaller the better
            eef_height_penalty += torch.logical_or(
                (eef_xyz_left.max(dim=1).values > (self.bbox[:, 1] - 0.02)).any(dim=-1), 
                (eef_xyz_left.min(dim=1).values < (self.bbox[:, 0] + 0.02)).any(dim=-1)
            ).to(torch.float32) * 100.0  # (n_sample,)  # the smaller the better
            eef_height_penalty += torch.logical_or(
                (eef_xyz_right.max(dim=1).values > (self.bbox[:, 1] - 0.02)).any(dim=-1), 
                (eef_xyz_right.min(dim=1).values < (self.bbox[:, 0] + 0.02)).any(dim=-1)
            ).to(torch.float32) * 100.0  # (n_sample,)  # the smaller the better

            reward = -chamfer_distance - eef_xyz_dist_penalty - eef_height_penalty  # to maximize
        else:
            assert eef_xyz.shape[2] == 1
            eef_xyz_dist_penalty = 0
            eef_height_penalty = (eef_xyz[:, :, 0, 2].max(dim=1).values > -0.02).to(torch.float32) * 100.0  # (n_sample,)  # the smaller the better
            eef_height_penalty += torch.logical_or(
                (eef_xyz[:, :, 0].max(dim=1).values > (self.bbox[:, 1] - 0.02)).any(dim=-1), 
                (eef_xyz[:, :, 0].min(dim=1).values < (self.bbox[:, 0] + 0.02)).any(dim=-1)
            ).to(torch.float32) * 100.0  # (n_sample,)  # the smaller the better
            reward = -chamfer_distance - eef_xyz_dist_penalty - eef_height_penalty  # to maximize

        print('best reward:', reward.max().item())
        eval_out = {
            'reward_seqs': reward,
        }
        return eval_out

    def optimize_action_mppi(self, act_seqs, reward_seqs):

        weight_seqs = F.softmax(reward_seqs * self.reward_weight, dim=0)  # (n_sample,)
        assert len(weight_seqs.shape) == 1 and weight_seqs.shape[0] == self.n_sample_chunk

        n_sample = self.n_sample_chunk
        n_grippers = self.cfg.sim.num_grippers
        assert act_seqs.shape[0] == n_sample
        eef_xyz = act_seqs[:, :, :n_grippers * 3].reshape(n_sample, self.n_look_ahead, n_grippers, 3)
        eef_rot = act_seqs[:, :, n_grippers * 3:n_grippers * (3 + 3 * 3)].reshape(n_sample, self.n_look_ahead, n_grippers, 3, 3)
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)
        eef_gripper = act_seqs[:, :, n_grippers * (3 + 3 * 3):].reshape(n_sample, self.n_look_ahead, n_grippers, 1)

        eef_xyz = torch.sum(weight_seqs[:, None, None, None] * eef_xyz, dim=0)  # (n_look_ahead, n_grippers, 3)
        eef_gripper = torch.sum(weight_seqs[:, None, None, None] * eef_gripper, dim=0)  # (n_look_ahead, n_grippers, 1)
        eef_quat = torch.sum(weight_seqs[:, None, None, None] * eef_quat, dim=0)  # (n_look_ahead, n_grippers, 4)

        eef_quat = eef_quat / (torch.norm(eef_quat, dim=-1, keepdim=True) + 1e-6)  # normalize
        
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)
        act_seq = torch.zeros((self.n_look_ahead, n_grippers * (3 + 9 + 1)), device=self.torch_device, dtype=torch.float32)
        act_seq[:, :n_grippers * 3] = eef_xyz.reshape(self.n_look_ahead, -1)
        act_seq[:, n_grippers * 3:n_grippers * (3 + 3 * 3)] = eef_rot.reshape(self.n_look_ahead, -1)
        act_seq[:, n_grippers * (3 + 3 * 3):] = eef_gripper.reshape(self.n_look_ahead, -1)

        act_seq = self.clip_actions(act_seq[None])[0]
        return act_seq

    def clip_actions(self, act_seqs):
        no_sample_dim = False
        if len(act_seqs.shape) == 2:
            no_sample_dim = True
            act_seqs = act_seqs[None]
        n_sample = act_seqs.shape[0]
        n_grippers = self.cfg.sim.num_grippers
        eef_xyz = act_seqs[:, :, :n_grippers * 3].reshape(n_sample, self.n_look_ahead, n_grippers, 3)
        eef_rot = act_seqs[:, :, n_grippers * 3:n_grippers * (3 + 3 * 3)].reshape(n_sample, self.n_look_ahead, n_grippers, 3, 3)
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)  # (n_sample, n_look_ahead, n_grippers, 4)
        eef_gripper = act_seqs[:, :, n_grippers * (3 + 3 * 3):].reshape(n_sample, self.n_look_ahead, n_grippers, 1)

        eef_xyz = torch.clamp(eef_xyz, self.bbox[:, 0], self.bbox[:, 1])
        eef_aa = kornia.geometry.conversions.quaternion_to_axis_angle(eef_quat)  # (n_sample, n_look_ahead, n_grippers, 3)
        max_rad = 1.0
        eef_aa_mask = torch.norm(eef_aa, dim=-1) > max_rad  # (n_sample, n_look_ahead, n_grippers)
        eef_aa[eef_aa_mask] = eef_aa[eef_aa_mask] / torch.norm(eef_aa[eef_aa_mask], dim=-1, keepdim=True) * max_rad  # cannot exceed 0.5 rad
        eef_quat = kornia.geometry.conversions.axis_angle_to_quaternion(eef_aa)
        eef_quat = eef_quat / (torch.norm(eef_quat, dim=-1, keepdim=True) + 1e-6)  # normalize
        eef_gripper = torch.clamp(eef_gripper, 0.0, 1.0)

        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)
        act_seqs[:, :, :n_grippers * 3] = eef_xyz.reshape(n_sample, self.n_look_ahead, -1)
        act_seqs[:, :, n_grippers * 3:n_grippers * (3 + 3 * 3)] = eef_rot.reshape(n_sample, self.n_look_ahead, -1)
        act_seqs[:, :, n_grippers * (3 + 3 * 3):] = eef_gripper.reshape(n_sample, self.n_look_ahead, -1)
        if no_sample_dim:
            assert act_seqs.shape[0] == 1
            act_seqs = act_seqs[0]
        return act_seqs


    def get_command(self, state, save_dir=None, is_first_iter=False):

        self.state = state

        best_act_seq = None
        best_reward_seq = None

        pts = state["perception_out"]["value"]["pts"].copy()
        pts = np.concatenate(pts, axis=0)

        # remove outliers
        rm_outliers = False
        if rm_outliers:
            pcd_rm = o3d.geometry.PointCloud()
            pcd_rm.points = o3d.utility.Vector3dVector(pts)
            outliers = None
            new_outlier = None
            rm_iter = 0
            while new_outlier is None or len(new_outlier.points) > 0:
                _, inlier_idx = pcd_rm.remove_statistical_outlier(
                    nb_neighbors = 30, std_ratio = 2.5 + rm_iter * 0.5
                )
                new_pcd = pcd_rm.select_by_index(inlier_idx)
                new_outlier = pcd_rm.select_by_index(inlier_idx, invert=True)
                if outliers is None:
                    outliers = new_outlier
                else:
                    outliers += new_outlier
                pcd_rm = new_pcd
                rm_iter += 1
            pts = np.array(pcd_rm.points)

        pts = torch.tensor(pts, dtype=torch.float32, device=self.torch_device)

        if is_first_iter:
            self.dynamics_module.reset_preprocess_meta(pts)
        self.dynamics_module.reset_downsample_indices(pts)

        self.pts = pts

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            chamfer_now = batch_chamfer_dist(pts[None], self.target_state.clone())
            print('chamfer_now:', chamfer_now.item())
            with open(Path(save_dir) / 'chamfer.txt', 'w') as f:
                f.write(str(chamfer_now.item()))

            pts_save = pts.cpu().numpy()
            o3d_pts = o3d.geometry.PointCloud()
            o3d_pts.points = o3d.utility.Vector3dVector(pts_save)
            o3d.io.write_point_cloud(str(Path(save_dir) / 'pts.ply'), o3d_pts)

            target_state_save = self.target_state.cpu().numpy()
            o3d_target_state = o3d.geometry.PointCloud()
            o3d_target_state.points = o3d.utility.Vector3dVector(target_state_save)
            o3d.io.write_point_cloud(str(Path(save_dir) / 'target_state.ply'), o3d_target_state)

        if self.bimanual:
            left_robot_out = state["robot_out"]["left_value"]
            left_gripper_out = state["gripper_out"]["left_value"]
            right_robot_out = state["robot_out"]["right_value"]
            right_gripper_out = state["gripper_out"]["right_value"]
            left_robot_out = torch.tensor(left_robot_out, dtype=torch.float32, device=self.torch_device)
            left_gripper_out = torch.tensor(left_gripper_out, dtype=torch.float32, device=self.torch_device)
            right_robot_out = torch.tensor(right_robot_out, dtype=torch.float32, device=self.torch_device)
            right_gripper_out = torch.tensor(right_gripper_out, dtype=torch.float32, device=self.torch_device)
            robot_out = None
            gripper_out = None
        else:
            robot_out = state["robot_out"]["value"]
            gripper_out = state["gripper_out"]["value"]
            robot_out = torch.tensor(robot_out, dtype=torch.float32, device=self.torch_device)
            gripper_out = torch.tensor(gripper_out, dtype=torch.float32, device=self.torch_device)
            left_robot_out = None
            left_gripper_out = None
            right_robot_out = None
            right_gripper_out = None
        
        if self.bimanual:
            b2w_l = state["b2w_l"]
            b2w_r = state["b2w_r"]
            b2w_l = torch.tensor(b2w_l, dtype=torch.float32, device=self.torch_device)
            b2w_r = torch.tensor(b2w_r, dtype=torch.float32, device=self.torch_device)
            b2w = None
        else:
            b2w = state["b2w"]
            b2w = torch.tensor(b2w, dtype=torch.float32, device=self.torch_device)
            b2w_l = None
            b2w_r = None

        # construct act_seq using current robot state
        # assert not self.cfg.sim.gripper_points
        eef_xyz = torch.zeros((self.n_look_ahead + 1, self.cfg.sim.num_grippers, 3), device=self.torch_device)
        eef_quat = torch.zeros((self.n_look_ahead + 1, self.cfg.sim.num_grippers, 4), device=self.torch_device)
        eef_quat[:, :, 0] = 1.0
        eef_gripper = torch.zeros((self.n_look_ahead + 1, 1), device=self.torch_device)
        
        # construct eef_world
        eef_points = torch.tensor([[0.0, 0.0, 0.175, 1]], device=self.torch_device)  # the eef point in the gripper frame
        eef_axis = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32, device=self.torch_device)  # (3, 4)
        if self.bimanual:
            left_eef_world_list = []
            right_eef_world_list = []
            assert left_robot_out is not None and right_robot_out is not None
            assert b2w_l is not None and b2w_r is not None
            for e2b, b2w, eef_world_list in zip([left_robot_out, right_robot_out], [b2w_l, b2w_r], [left_eef_world_list, right_eef_world_list]):
                eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
                # eef_points_vis.append(eef_points)
                # eef_points_world_vis.append(eef_points_world)
                eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
                eef_world = torch.cat([eef_points_world, eef_orientation_world], dim=0)  # (n+3, 3)
                eef_world_list.append(eef_world)
            left_eef_world = torch.cat(left_eef_world_list, dim=0)  # (n+3, 3)
            right_eef_world = torch.cat(right_eef_world_list, dim=0)  # (n+3, 3)
            eef_world = torch.cat([left_eef_world, right_eef_world], dim=0)  # (2n+6, 3)
        else:
            assert robot_out is not None
            assert b2w is not None
            e2b = robot_out  # (4, 4)
            eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
            # eef_points_vis.append(eef_points)
            # eef_points_world_vis.append(eef_points_world)
            eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
            eef_world = torch.cat([eef_points_world, eef_orientation_world], dim=0)  # (n+3, 3)
        
        if self.bimanual:
            assert left_gripper_out is not None and right_gripper_out is not None
            gripper_world = torch.tensor([left_gripper_out, right_gripper_out, 0.0], device=self.torch_device)[None, :]  # (1, 3)
        else:
            assert gripper_out is not None
            gripper_world = torch.tensor([gripper_out, 0.0, 0.0], device=self.torch_device)[None, :]  # (1, 3)
        
        eef_world = torch.cat([eef_world, gripper_world], dim=0)  # (n+4, 3) or (2n+7, 3)
        robot = eef_world

        # decode eef_world
        if len(robot.shape) > 1:  # 5 or 9
            assert robot.shape[0] in [5, 9]  # bi-manual (2 * (1 pos + 3 rot) + 1 gripper) or single arm (1 pos + 3 rot + 1 gripper or 1 pos)
            gripper = robot[-1]
            robot = robot[:-1]
            robot = robot.reshape(-1, 4, 3)
            robot_trans = robot[:, 0]  # (n, 3)
            robot_rot = robot[:, 1:]  # (n, 3, 3)
            if robot_trans.shape[0] == 1:  # single arm
                gripper = gripper[:1]  # (1,)
            else:  # bi-manual
                gripper = gripper[:2]  # (2,)
        else:
            assert len(robot.shape) == 1 and robot.shape[0] == 3
            robot_trans = robot
            robot_rot = torch.eye(3).to(self.torch_device).to(torch.float32)
            gripper = torch.tensor([0.0], device=self.torch_device).to(torch.float32)
        robot_trans = robot_trans #  + torch.tensor([0, 0, -0.01], device=self.torch_device)  # offset
        gripper = torch.clamp(gripper / 800.0, 0, 1)  # 1: open, 0: close

        # init eef variables in world frame
        eef_xyz = robot_trans.reshape(-1, 3)
        eef_rot = robot_rot.reshape(-1, 3, 3)  # (n_grippers, 3, 3)
        eef_gripper = gripper.reshape(-1)  # (n_grippers,)
        assert eef_xyz.shape[0] == eef_rot.shape[0] == eef_gripper.shape[0]
        n_grippers = eef_xyz.shape[0]

        self.eef_xyz = eef_xyz
        self.eef_rot = eef_rot
        self.eef_gripper = eef_gripper

        # init act_seq in world frame
        act_seq = torch.zeros((self.n_look_ahead, n_grippers * (3 + 9 + 1)), device=self.torch_device)
        act_seq[:, :n_grippers * 3] = eef_xyz.reshape(-1)
        act_seq[:, n_grippers * 3:n_grippers * (3 + 3 * 3)] = eef_rot.reshape(-1)
        act_seq[:, n_grippers * (3 + 3 * 3):] = eef_gripper.reshape(-1)

        res_all = []
        for ci in range(self.n_chunk):

            best_act_seq = act_seq
            for ti in range(self.n_update_iter):

                print(f'chunk: {ci}/{self.n_chunk}, iter: {ti}/{self.n_update_iter}')
                
                with torch.no_grad():
                    act_seqs = self.sample_action_seq(act_seq, iter_index=ti)  # support iteration-dependent noise
                    model_out = self.model_rollout(act_seqs, visualize_pv=False)  # TODO
                    eval_out = self.evaluate_traj(model_out, act_seqs)
                    reward_seqs = eval_out["reward_seqs"]  # (n_sample,)
                    act_seq = self.optimize_action_mppi(act_seqs, reward_seqs)

                    best_reward_idx = torch.argmax(reward_seqs)

                    if ti == 0:
                        best_act_seq = act_seqs[best_reward_idx]
                        best_reward_seq = reward_seqs[best_reward_idx]
                    elif reward_seqs[best_reward_idx] > best_reward_seq:
                        best_act_seq = act_seqs[best_reward_idx]
                        best_reward_seq = reward_seqs[best_reward_idx]
                    
                    # model_out = self.model_rollout(best_act_seq[None].repeat(self.n_sample_chunk, 1, 1), visualize_pv=True)  # TODO

            torch.cuda.empty_cache()
            # model_out = self.model_rollout(best_act_seq[None].repeat(self.n_sample_chunk, 1, 1), visualize_pv=True)  # TODO

            res = {
                "best_act_seq": best_act_seq,  # (n_look_ahead, n_grippers * (3 + 9 + 1))
                "best_reward_seq": best_reward_seq,
            }
            res_all.append(res)

        reward_list = [res["best_reward_seq"].item() for res in res_all]
        best_idx = np.argmax(reward_list)
        best_act_seq = res_all[best_idx]['best_act_seq']  # (n_look_ahead, n_grippers * (3 + 9 + 1))

        torch.cuda.empty_cache()

        eef_xyz = best_act_seq[:, :n_grippers * 3].reshape(self.n_look_ahead, n_grippers, 3)
        eef_rot = best_act_seq[:, n_grippers * 3:n_grippers * (3 + 3 * 3)].reshape(self.n_look_ahead, n_grippers, 3, 3)
        eef_gripper = best_act_seq[:, n_grippers * (3 + 3 * 3):].reshape(self.n_look_ahead, n_grippers, 1)

        # transform back into robot coordinates
        eef_points = torch.tensor([[0.0, 0.0, 0.175, 1]], device=self.torch_device)  # the eef point in the gripper frame
        eef_axis = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32, device=self.torch_device)  # (3, 4)

        # initialize command
        command = [[] for _ in range(eef_xyz.shape[-2])]  # (n_grippers,)
        e2b_command = [[] for _ in range(eef_xyz.shape[-2])]  # (n_grippers,)

        look_ahead_range = range(self.n_look_ahead)

        for li in look_ahead_range:
            
            if self.bimanual:
                assert b2w_l is not None and b2w_r is not None

                left_eef_xyz = eef_xyz[li:li+1, 0]  # (1, 3)
                right_eef_xyz = eef_xyz[li:li+1, 1]  # (1, 3)

                left_eef_rot = eef_rot[li, 0]  # (3, 3)
                right_eef_rot = eef_rot[li, 1]  # (3, 3)

                e2b_l = torch.eye(4, device=self.torch_device)
                e2b_r = torch.eye(4, device=self.torch_device)

                for b2w, e2b, eef_points_world, eef_orientation_world in \
                    zip([b2w_l, b2w_r], [e2b_l, e2b_r], [left_eef_xyz, right_eef_xyz], [left_eef_rot, right_eef_rot]):

                    eef_orientation_world = eef_orientation_world.T  # (3, 3) = b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T
                    eef_orientation_world = b2w[:3, :3].T @ eef_orientation_world  # (3, 3) = e2b[:3, :3] @ eef_axis[:, :3].T
                    e2b[:3, :3] = eef_orientation_world @ eef_axis[:, :3]  # (3, 3) = e2b[:3, :3]

                    eef_points_world = eef_points_world.T  # (3, n) = b2w_R @ (e2b_R @ eef_points[:, :3].T + e2b_t) + b2w_t
                    eef_points_world = eef_points_world - b2w[:3, 3].reshape(-1, 1)  # (3, n) = b2w_R @ (e2b_R @ eef_points[:, :3].T + e2b_t)
                    eef_points_world = b2w[:3, :3].T @ eef_points_world  # (3, n) = e2b_R @ eef_points[:, :3].T + e2b_t
                    e2b[:3, 3:4] = eef_points_world - e2b[:3, :3] @ eef_points[:, :3].T  # (3, n) = e2b_t

                e2b_list = [e2b_l, e2b_r]

            else:
                assert b2w is not None

                eef_points_world = eef_xyz[li:li+1, 0]  # (n, 3)
                eef_orientation_world = eef_rot[li, 0]

                e2b = torch.eye(4, device=self.torch_device)

                eef_orientation_world = eef_orientation_world.T  # (3, 3) = b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T
                eef_orientation_world = b2w[:3, :3].T @ eef_orientation_world  # (3, 3) = e2b[:3, :3] @ eef_axis[:, :3].T
                e2b[:3, :3] = eef_orientation_world @ eef_axis[:, :3]  # (3, 3) = e2b[:3, :3]

                eef_points_world = eef_points_world.T  # (3, n) = b2w_R @ (e2b_R @ eef_points[:, :3].T + e2b_t) + b2w_t
                eef_points_world = eef_points_world - b2w[:3, 3].reshape(-1, 1)  # (3, n) = b2w_R @ (e2b_R @ eef_points[:, :3].T + e2b_t)
                eef_points_world = b2w[:3, :3].T @ eef_points_world  # (3, n) = e2b_R @ eef_points[:, :3].T + e2b_t
                e2b[:3, 3:4] = eef_points_world - e2b[:3, :3] @ eef_points[:, :3].T  # (3, n) = e2b_t

                e2b_list = [e2b]


            for gripper_id in range(eef_xyz.shape[-2]):

                fk_trans_mat = e2b_list[gripper_id].cpu().numpy()

                cur_xyzrpy = np.zeros(6)
                cur_xyzrpy[:3] = fk_trans_mat[:3, 3] * 1000
                cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])
                cur_xyzrpy[3:] = cur_xyzrpy[3:] / np.pi * 180

                gripper = eef_gripper[li, gripper_id].item()
                gripper = gripper * 800.0

                single_command = list(cur_xyzrpy) + [gripper]
                command[gripper_id].append(single_command)

                # debug
                e2b_command[gripper_id].append(e2b_list[gripper_id].cpu().numpy())

        return command

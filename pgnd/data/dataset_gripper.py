from pathlib import Path
from typing import Union, Optional
from omegaconf import DictConfig

import os
import torch
import time
import shutil
import json
import yaml
import random
import glob
import kornia
import numpy as np
import pickle as pkl
import open3d as o3d
from torch.utils.data import Dataset
from dgl.geometry import farthest_point_sampler
from sklearn.neighbors import NearestNeighbors


def fps(x, n, device, random_start=False):
    start_idx = random.randint(0, x.shape[0] - 1) if random_start else 0
    fps_idx = farthest_point_sampler(x[None], n, start_idx=start_idx)[0]
    fps_idx = fps_idx.to(x.device)
    return fps_idx


def read_splat(splat_file):
    with open(splat_file, "rb") as f:
        data = f.read()
    pts = []
    colors = []
    scales = []
    quats = []
    opacities = []
    for i in range(0, len(data), 32):
        v = np.frombuffer(data[i : i + 12], dtype=np.float32)
        s = np.frombuffer(data[i + 12 : i + 24], dtype=np.float32)
        c = np.frombuffer(data[i + 24 : i + 28], dtype=np.uint8) / 255
        q = np.frombuffer(data[i + 28 : i + 32], dtype=np.uint8)
        q = (q * 1.0 - 128) / 128
        pts.append(v)
        scales.append(s)
        colors.append(c[:3])
        quats.append(q)
        opacities.append(c[3:])
    return np.array(pts), np.array(colors), np.array(scales), np.array(quats), np.array(opacities)


class RealGripperDataset(Dataset):

    def __init__(self, cfg: DictConfig, device: torch.device, train=False) -> None:
        super().__init__()

        pts, colors, scales, quats, opacities = read_splat('experiments/log/gs/ckpts/gripper_new.splat')
        self.device = device
        self.uniform = cfg.sim.uniform or not train
        self.n_particles = 500

        R = np.array(
            [[1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]]
        )
        eef_global_T = np.array([cfg.model.eef_t[0], cfg.model.eef_t[1], cfg.model.eef_t[2]])  # 1018_sloth: 0.185, 1018_rope_short: 0.013)
        pts = pts + eef_global_T
        pts = pts @ R.T
        scale = cfg.sim.preprocess_scale
        pts = pts * scale

        axis = np.array([0, 1, 0])
        angle = -27  # degree
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi / 180 * angle)
        translation = np.array([-0.015, 0.06, 0])

        pts = pts @ R.T
        pts = pts + translation

        R = np.array(
            [[0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]]
        )
        pts = pts @ R.T

        self.gripper_pts = torch.from_numpy(pts).to(torch.float32)

    def __len__(self) -> int:
        return 10000000

    def __getitem__(self, index):
        x = self.gripper_pts
        if self.uniform:
            downsample_indices = fps(x, self.n_particles, self.device, random_start=True)
        else:
            downsample_indices = torch.randperm(x.shape[0])[:self.n_particles]
        x = x[downsample_indices]
        return x, downsample_indices

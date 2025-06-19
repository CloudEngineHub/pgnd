import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def chamfer(x, y):  # x: (B, N, D), y: (B, M, D)
    x = x[:, None].repeat(1, y.shape[1], 1, 1)  # (B, M, N, D)
    y = y[:, :, None].repeat(1, 1, x.shape[2], 1)  # (B, M, N, D)
    dis = torch.norm(x - y, 2, dim=-1)  # (B, M, N)
    dis_xy = torch.mean(dis.min(dim=2).values, dim=1)  # dis_xy: mean over N
    dis_yx = torch.mean(dis.min(dim=1).values, dim=1)  # dis_yx: mean over M
    return dis_xy + dis_yx


def batch_chamfer_dist(xyz, xyz_gt):
    # xyz: (B, N, 3)
    # xyz_gt: (M, 3)

    # mean aligning
    # chamfer = (xyz.mean(dim=1) - xyz_gt.mean(dim=0)).norm(dim=1)  # (B,)

    # chamfer distance
    xyz_gt = xyz_gt[None]  # (1, M, 3)
    dist1 = torch.sqrt(torch.sum((xyz[:, :, None] - xyz_gt[:, None]) ** 2, dim=3))  # (B, N, M)
    dist2 = torch.sqrt(torch.sum((xyz_gt[:, None] - xyz[:, :, None]) ** 2, dim=3))  # (B, M, N)
    chamfer = torch.mean(torch.min(dist1, dim=1).values, dim=1) + torch.mean(torch.min(dist2, dim=1).values, dim=1)  # (B,)
    return chamfer


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


def clip_actions(action, action_lower_lim, action_upper_lim):
    action_new = action.clone()
    # action_new[..., 2] = angle_normalize(action[..., 2])
    action_new.data.clamp_(action_lower_lim, action_upper_lim)
    return action_new

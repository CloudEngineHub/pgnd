from typing import Optional

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .network.pointnet import PointNetEncoder
from .network.nerf import CondNeRFModel
from .network.utils import get_grid_locations, fill_grid_locations


class PGNDModel(nn.Module):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.feature_dim = 64
        self.radius = cfg.model.material.radius
        self.n_history = cfg.sim.n_history
        self.num_grids_list = cfg.sim.num_grids[:3]
        self.dx = cfg.sim.num_grids[3]
        self.inv_dx = 1 / self.dx
        self.requires_grad = cfg.model.material.requires_grad
        self.pe_num_func = int(np.log2(self.inv_dx)) + cfg.model.material.pe_num_func_res
        self.pe_dim = 3 + self.pe_num_func * 6
        self.output_scale = cfg.model.material.output_scale
        self.input_scale = cfg.model.material.input_scale
        self.absolute_y = cfg.model.material.absolute_y

        self.encoder = PointNetEncoder(
            global_feat=(cfg.model.material.radius <= 0), 
            feature_transform=False, 
            feature_dim=self.feature_dim, 
            channel=6 * (1 + self.n_history),
        )
        self.decoder = CondNeRFModel(
            xyz_dim=self.pe_dim,
            condition_dim=self.feature_dim,
            out_channel=3,
            num_layers=2,
            hidden_size=64,
            skip_connect_every=4,
        )

    def positional_encoding(self, tensor):
        num_encoding_functions = self.pe_num_func
        if num_encoding_functions == 0:
            assert include_input
            return tensor

        encoding = [tensor]  # include the input itself
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

    def forward(self, x: Tensor, v: Tensor, x_his: Tensor, v_his: Tensor, enabled: Tensor) -> Tensor:
        # x: (bsz, num_particles, 3)
        # v: (bsz, num_particles, 3)
        bsz = x.shape[0]
        num_particles = x.shape[1]
        v = v * self.input_scale
        v_his = v_his * self.input_scale

        x_his = x_his.reshape(bsz, num_particles, self.n_history, 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history, 3)
        x_his = x_his.detach()
        v_his = v_his.detach()
        
        x_grid, grid_idxs = get_grid_locations(x, self.num_grids_list, self.dx)
        x_grid = x_grid.detach()
        grid_idxs = grid_idxs.detach()

        # centering
        x_center = x.mean(1, keepdim=True)
        if self.absolute_y:
            x_center[:, :, 1] = 0  # only centering x and z
        x = x - x_center
        x_his = x_his - x_center[:, :, None]
        if self.training:
            # random azimuth
            theta = torch.rand(bsz, 1, device=x.device) * 2 * np.pi
            rot = torch.stack([
                torch.cos(theta), torch.zeros_like(theta), torch.sin(theta),
                torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta),
                -torch.sin(theta), torch.zeros_like(theta), torch.cos(theta),
            ], dim=-1).reshape(bsz, 3, 3)
            inv_rot = rot.transpose(1, 2)
        else:
            rot = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(bsz, 1, 1)
            inv_rot = rot.transpose(1, 2)
        x = torch.einsum('bij,bjk->bik', x, rot)
        x_his = torch.einsum('bcij,bjk->bcik', x_his, rot)  # (bsz, num_particles, n_history, 3)
        v = torch.einsum('bij,bjk->bik', v, rot)
        v_his = torch.einsum('bcij,bjk->bcik', v_his, rot)  # (bsz, num_particles, n_history, 3)

        x_grid = x_grid - x_center
        x_grid = x_grid @ rot
        x_his = x_his.reshape(bsz, num_particles, self.n_history * 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history * 3)

        feat = torch.cat([x, v, x_his, v_his], dim=-1)  # (bsz, num_particles, 6 * (1 + n_history))
        feat = feat.permute(0, 2, 1)  # (bsz, 6 * (1 + n_history), num_particles)
        feat, trans, trans_feat = self.encoder(feat, enabled)  # feat: (bsz, feature_dim, num_particles)

        if self.radius > 0:
            # aggregate neighborhood
            # x.shape: (bsz, num_particles, 3)
            # x_grid.shape: (bsz, num_grids_total, 3)
            dist_pt_grid = torch.cdist(x_grid, x, p=2)
            mask = dist_pt_grid < self.radius  # (bsz, num_grids_total, num_particles)
            mask_normed = mask / (mask.sum(dim=-1, keepdim=True) + 1e-5)  # for each grid, normalize the weights
            mask_normed = mask_normed.detach()
            feat = mask_normed @ feat.permute(0, 2, 1)  # (bsz, num_grids_total, feature_dim)
        else:
            # global max pooling
            feat = feat[:, None, :].repeat(1, x_grid.shape[1], 1)  # (bsz, num_grids_total, feature_dim)

        # positional encoding and decoder
        feat = feat.reshape(-1, self.feature_dim)
        x_grid = x_grid.reshape(-1, 3)
        x_grid = self.positional_encoding(x_grid)
        feat = self.decoder(x_grid, feat)
        feat = feat * self.output_scale
        feat = feat.reshape(bsz, -1, feat.shape[-1])
        feat = torch.bmm(feat, inv_rot)

        # expand to full grid
        feat = fill_grid_locations(feat, grid_idxs, self.num_grids_list)
        return feat

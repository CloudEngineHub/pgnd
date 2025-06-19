from typing import Optional, Union, Sequence, Any
from omegaconf import DictConfig
import numpy as np
import torch
from torch import Tensor
import warp as wp

from .utils import ConstantBatch, StaticsBatch, CollidersBatch, GridDataBatch, ParticleDataBatch


class SimModelBatch:

    def __init__(self, constant: ConstantBatch, device: wp.context.Devicelike = None, batch_size: int = 1, requires_grad: bool = False) -> None:
        self.constant = constant
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad
        self.batch_size = batch_size

    def update_friction(self, friction) -> None:
        self.constant.update_friction(friction, self.requires_grad)

    @staticmethod
    @wp.kernel
    def grid_op_batch(
            constant: ConstantBatch,
            colliders: CollidersBatch,
            grid_curr: GridDataBatch,
            grid_next: GridDataBatch,
        ) -> None:

        b, px, py, pz = wp.tid()

        v = grid_curr.v[b, px, py, pz]

        friction = constant.friction[b, 0]

        if px < constant.bound and v[0] < 0.0:
            impulse = wp.vec3(v[0], 0.0, 0.0)
            v = wp.vec3(0.0, v[1], v[2])
            friction_impulse = wp.float32(-1.) * friction * wp.length(impulse) * wp.normalize(v)
            friction_impulse *= wp.min(1.0, wp.length(v) / (wp.length(friction_impulse) + 1e-10))
            v = v + friction_impulse
        if py < constant.bound and v[1] < 0.0:
            impulse = wp.vec3(0.0, v[1], 0.0)
            v = wp.vec3(v[0], 0.0, v[2])
            friction_impulse = wp.float32(-1.) * friction * wp.length(impulse) * wp.normalize(v)
            friction_impulse *= wp.min(1.0, wp.length(v) / (wp.length(friction_impulse) + 1e-10))
            v = v + friction_impulse
        if pz < constant.bound and v[2] < 0.0:
            impulse = wp.vec3(0.0, 0.0, v[2])
            v = wp.vec3(v[0], v[1], 0.0)
            friction_impulse = wp.float32(-1.) * friction * wp.length(impulse) * wp.normalize(v)
            friction_impulse *= wp.min(1.0, wp.length(v) / (wp.length(friction_impulse) + 1e-10))
            v = v + friction_impulse
        if px > int(constant.num_grids_list[0]) - constant.bound and v[0] > 0.0:
            impulse = wp.vec3(v[0], 0.0, 0.0)
            v = wp.vec3(0.0, v[1], v[2])
            friction_impulse = wp.float32(-1.) * friction * wp.length(impulse) * wp.normalize(v)
            friction_impulse *= wp.min(1.0, wp.length(v) / (wp.length(friction_impulse) + 1e-10))
            v = v + friction_impulse
        if py > int(constant.num_grids_list[1]) - constant.bound and v[1] > 0.0:
            impulse = wp.vec3(0.0, v[1], 0.0)
            v = wp.vec3(v[0], 0.0, v[2])
            friction_impulse = wp.float32(-1.) * friction * wp.length(impulse) * wp.normalize(v)
            friction_impulse *= wp.min(1.0, wp.length(v) / (wp.length(friction_impulse) + 1e-10))
            v = v + friction_impulse
        if pz > int(constant.num_grids_list[2]) - constant.bound and v[2] > 0.0:
            impulse = wp.vec3(0.0, 0.0, v[2])
            v = wp.vec3(v[0], v[1], 0.0)
            friction_impulse = wp.float32(-1.) * friction * wp.length(impulse) * wp.normalize(v)
            friction_impulse *= wp.min(1.0, wp.length(v) / (wp.length(friction_impulse) + 1e-10))
            v = v + friction_impulse

        for gripper_id in range(colliders.gripper_centers.shape[1]):

            dx = wp.float32(px) * constant.dx - colliders.gripper_centers[b, gripper_id][0]
            dy = wp.float32(py) * constant.dx - colliders.gripper_centers[b, gripper_id][1]
            dz = wp.float32(pz) * constant.dx - colliders.gripper_centers[b, gripper_id][2]

            grid_from_gripper = wp.vec3(dx, dy, dz)
            grid_from_gripper_norm = wp.length(grid_from_gripper)

            gripper_add_radii = 0.0
            if grid_from_gripper_norm < colliders.gripper_radii[b, gripper_id] + gripper_add_radii and colliders.gripper_open[b, gripper_id] < 0.5:
                if wp.length(colliders.gripper_quat[b, gripper_id]) > 0:  # filter out
                    gripper_vel = colliders.gripper_vels[b, gripper_id]
                    gripper_quat_vel = colliders.gripper_quat_vel[b, gripper_id]

                    gripper_angular_vel = wp.length(gripper_quat_vel)
                    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)

                    grid_from_gripper_axis = grid_from_gripper - \
                        wp.dot(gripper_quat_axis, grid_from_gripper) * gripper_quat_axis
                    v = wp.cross(gripper_quat_vel, grid_from_gripper_axis) + gripper_vel

                else:
                    v = colliders.gripper_vels[b, gripper_id]

        grid_next.v[b, px, py, pz] = v

    @staticmethod
    @wp.kernel
    def g2p_batch(
            constant: ConstantBatch,
            statics: StaticsBatch,
            particle_curr: ParticleDataBatch,
            grid_next: GridDataBatch,
            particle_next: ParticleDataBatch,
        ) -> None:

        b, p = wp.tid()

        if statics.enabled[b, p] == 0:
            return

        p_x = particle_curr.x[b, p] * constant.inv_dx
        base_x = int(p_x[0] - 0.5)
        base_y = int(p_x[1] - 0.5)
        base_z = int(p_x[2] - 0.5)
        f_x = p_x - wp.vec3(
            float(base_x),
            float(base_y),
            float(base_z))

        # quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx,fx-1,fx-2]
        wa = wp.vec3(1.5) - f_x
        wb = f_x - wp.vec3(1.0)
        wc = f_x - wp.vec3(0.5)

        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )

        new_v = wp.vec3(0.0)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]

                    v = grid_next.v[b, base_x + i, base_y + j, base_z + k]
                    new_v = new_v + weight * v

        particle_next.v[b, p] = new_v

        bound = statics.clip_bound[b, p] * constant.dx + constant.eps
        new_x = particle_curr.x[b, p] + constant.dt * new_v
        new_x = wp.vec3(
            wp.clamp(new_x[0], 0.0 + bound, float(constant.num_grids_list[0]) * constant.dx - bound),
            wp.clamp(new_x[1], 0.0 + bound, float(constant.num_grids_list[1]) * constant.dx - bound),
            wp.clamp(new_x[2], 0.0 + bound, float(constant.num_grids_list[2]) * constant.dx - bound),
        )
        particle_next.x[b, p] = new_x


def build_model(
        cfg: DictConfig,
        batch_size: int,
        device: wp.context.Devicelike = None,
        requires_grad: bool = False
    ) -> SimModelBatch:

    dt: float = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
    bound: int = cfg.sim.bound
    eps: float = cfg.sim.eps
    assert len(cfg.sim.num_grids) == 4
    num_grids_list: list = cfg.sim.num_grids[:3]
    dx: float = cfg.sim.num_grids[3]
    inv_dx: float = 1 / dx

    constant = ConstantBatch()
    constant.init()
    constant.dt = dt
    constant.bound = bound
    constant.dx = dx
    constant.inv_dx = inv_dx
    constant.eps = eps
    friction = np.array([cfg.model.friction.value], dtype=np.float32)[None].repeat(batch_size, axis=0)
    constant.friction = wp.from_numpy(friction, dtype=float)  # [bsz, 1]
    constant.num_grids_list = wp.vec3(*num_grids_list)

    model = SimModelBatch(constant, device, batch_size, requires_grad)
    return model

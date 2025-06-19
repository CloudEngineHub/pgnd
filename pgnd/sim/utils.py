from typing import Optional, Union, Sequence, Any

import numpy as np
import torch
from torch import Tensor
import warp as wp


@wp.struct
class GridDataBatch(object):

    batch_size: int
    v: wp.array(dtype=wp.vec3, ndim=4)

    def init(self, shape: Sequence[int], device: wp.context.Devicelike = None, requires_grad: bool = False) -> None:
        self.batch_size = shape[0]
        self.v = wp.zeros(shape=shape, dtype=wp.vec3, ndim=4, device=device, requires_grad=requires_grad)

    def clear(self) -> None:
        self.v.zero_()

    def zero_grad(self) -> None:
        if self.v.requires_grad:
            self.v.grad.zero_()
    
    def from_torch(self, v: Optional[Tensor] = None) -> None:
        if v is not None:
            self.v = wp.from_torch(v.contiguous(), dtype=wp.vec3, requires_grad=self.v.requires_grad)
        
    def to_torch(self) -> Tensor:
        v = wp.to_torch(self.v).requires_grad_(self.v.requires_grad)
        return v
    
    def to_torch_grad(self) -> Tensor:
        grad_v = wp.to_torch(self.v.grad) if self.v.grad is not None else None
        return grad_v
    
    def from_torch_grad(self, grad_v: Optional[Tensor] = None) -> None:
        if grad_v is not None:
            self.v.grad = wp.from_torch(grad_v.contiguous(), dtype=wp.vec3)


@wp.struct
class ParticleDataBatch(object):

    x: wp.array(dtype=wp.vec3, ndim=2)
    v: wp.array(dtype=wp.vec3, ndim=2)

    def init(self, shape: Union[Sequence[int], int], device: wp.context.Devicelike = None, requires_grad: bool = False) -> None:
        self.x = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.v = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=requires_grad)

    def clear(self) -> None:
        self.x.zero_()
        self.v.zero_()
    
    def zero_grad(self) -> None:
        if self.x.requires_grad:
            self.x.grad.zero_()
        if self.v.requires_grad:
            self.v.grad.zero_()

    def from_torch(self, x: Optional[Tensor] = None, v: Optional[Tensor] = None) -> None:
        if x is not None:
            self.x = wp.from_torch(x.contiguous(), dtype=wp.vec3, requires_grad=self.x.requires_grad)
        if v is not None:
            self.v = wp.from_torch(v.contiguous(), dtype=wp.vec3, requires_grad=self.v.requires_grad)

    def to_torch(self) -> tuple[Tensor, Tensor]:
        x = wp.to_torch(self.x).requires_grad_(self.x.requires_grad)
        v = wp.to_torch(self.v).requires_grad_(self.v.requires_grad)
        return x, v

    def from_torch_grad(self, grad_x: Optional[Tensor] = None, grad_v: Optional[Tensor] = None) -> None:
        if grad_x is not None:
            self.x.grad = wp.from_torch(grad_x.contiguous(), dtype=wp.vec3)
        if grad_v is not None:
            self.v.grad = wp.from_torch(grad_v.contiguous(), dtype=wp.vec3)

    def to_torch_grad(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
        grad_x = wp.to_torch(self.x.grad) if self.x.grad is not None else None
        grad_v = wp.to_torch(self.v.grad) if self.v.grad is not None else None
        return grad_x, grad_v


@wp.struct
class StaticsBatch(object):

    clip_bound: wp.array(dtype=float, ndim=2)
    enabled: wp.array(dtype=int, ndim=2)

    def init(self, shape: Union[Sequence[int], int], device: wp.context.Devicelike = None) -> None:
        self.clip_bound = wp.zeros(shape=(shape[0], shape[1]), dtype=float, device=device, requires_grad=False)
        self.enabled = wp.zeros(shape=(shape[0], shape[1]), dtype=int, device=device, requires_grad=False)

    @staticmethod
    @wp.kernel
    def set_float(x: wp.array(dtype=float, ndim=2), value: wp.array(dtype=float)) -> None:
        b, p = wp.tid()
        x[b, p] = value[b]

    @staticmethod
    @wp.kernel
    def set_int_per_particle(x: wp.array(dtype=int, ndim=2), value: wp.array(dtype=int, ndim=2)) -> None:
        b, p = wp.tid()
        x[b, p] = value[b, p]

    def update_clip_bound(self, clip_bound: Tensor) -> None:
        clip_bound = wp.from_numpy(clip_bound.numpy(), dtype=float)
        wp.launch(self.set_float, dim=self.clip_bound.shape, inputs=[self.clip_bound, clip_bound], device=self.clip_bound.device)

    def update_enabled(self, enabled: Tensor) -> None:
        enabled = wp.from_numpy(enabled.numpy(), dtype=int)
        wp.launch(self.set_int_per_particle, dim=self.enabled.shape, inputs=[self.enabled, enabled], device=self.enabled.device)


@wp.struct
class CollidersBatch(object):

    gripper_centers: wp.array(dtype=wp.vec3, ndim=2)
    gripper_vels: wp.array(dtype=wp.vec3, ndim=2)
    gripper_radii: wp.array(dtype=float, ndim=2)
    gripper_quat: wp.array(dtype=wp.quat, ndim=2)
    gripper_quat_vel: wp.array(dtype=wp.vec3, ndim=2)
    gripper_open: wp.array(dtype=float, ndim=2)

    def init(self, shape: Sequence[int], device: wp.context.Devicelike = None, use_quat: bool = False) -> None:
        self.gripper_centers = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=False)
        self.gripper_vels = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=False)
        self.gripper_radii = wp.zeros(shape=shape, dtype=float, device=device, requires_grad=False)
        self.gripper_quat = wp.zeros(shape=shape, dtype=wp.quat, device=device, requires_grad=False)
        self.gripper_quat_vel = wp.zeros(shape=shape, dtype=wp.vec3, device=device, requires_grad=False)
        self.gripper_open = wp.zeros(shape=shape, dtype=float, device=device, requires_grad=False)

    @staticmethod
    @wp.kernel
    def set_int(x: wp.array(dtype=wp.int32, ndim=2), value: wp.array(dtype=wp.int32, ndim=2)) -> None:
        b, p = wp.tid()
        x[b, p] = value[b, p]

    @staticmethod
    @wp.kernel
    def set_float(x: wp.array(dtype=wp.float32, ndim=2), value: wp.array(dtype=wp.float32, ndim=2)) -> None:
        b, p = wp.tid()
        x[b, p] = value[b, p]
    
    @staticmethod
    @wp.kernel
    def set_vec3(x: wp.array(dtype=wp.vec3, ndim=2), value: wp.array(dtype=wp.vec3, ndim=2)) -> None:
        b, p = wp.tid()
        x[b, p] = value[b, p]

    @staticmethod
    @wp.kernel
    def add_vec3(x: wp.array(dtype=wp.vec3, ndim=2), value: wp.array(dtype=wp.vec3, ndim=2)) -> None:
        b, p = wp.tid()
        wp.atomic_add(x, b, p, value[b, p])
    
    @staticmethod
    @wp.kernel
    def set_quat(x: wp.array(dtype=wp.quat, ndim=2), value: wp.array(dtype=wp.quat, ndim=2)) -> None:
        b, p = wp.tid()
        x[b, p] = value[b, p]
    
    def update_grippers(self, gripper: Tensor) -> None:
        assert gripper.shape[-1] == 15  # quat, quat_vel (axis-angle)
        center, vel, quat, quat_vel, _, openness = torch.split(gripper, [3, 3, 4, 3, 1, 1], dim=-1)

        center_wp = wp.from_torch(center.contiguous(), dtype=wp.vec3)
        wp.launch(self.set_vec3, dim=self.gripper_centers.shape, inputs=[self.gripper_centers, center_wp], device=self.gripper_centers.device)

        vel_wp = wp.from_torch(vel.contiguous(), dtype=wp.vec3)
        wp.launch(self.set_vec3, dim=self.gripper_vels.shape, inputs=[self.gripper_vels, vel_wp], device=self.gripper_vels.device)

        quat_wp = wp.from_torch(quat.contiguous(), dtype=wp.quat)
        wp.launch(self.set_quat, dim=self.gripper_quat.shape, inputs=[self.gripper_quat, quat_wp], device=self.gripper_quat.device)

        quat_vel_wp = wp.from_torch(quat_vel.contiguous(), dtype=wp.vec3)
        wp.launch(self.set_vec3, dim=self.gripper_quat_vel.shape, inputs=[self.gripper_quat_vel, quat_vel_wp], device=self.gripper_quat_vel.device)

        openness_wp = wp.from_torch(openness.squeeze(-1).contiguous(), dtype=wp.float32)
        wp.launch(self.set_float, dim=self.gripper_open.shape, inputs=[self.gripper_open, openness_wp], device=self.gripper_open.device)

    def initialize_grippers(self, gripper: Tensor) -> None:
        assert gripper.shape[-1] == 15  # quat, quat_vel (axis-angle)
        center, vel, quat, quat_vel, radius, openness = torch.split(gripper, [3, 3, 4, 3, 1, 1], dim=-1)

        center_wp = wp.from_torch(center.contiguous(), dtype=wp.vec3)
        wp.launch(self.set_vec3, dim=self.gripper_centers.shape, inputs=[self.gripper_centers, center_wp], device=self.gripper_centers.device)

        vel_wp = wp.from_torch(vel.contiguous(), dtype=wp.vec3)
        wp.launch(self.set_vec3, dim=self.gripper_vels.shape, inputs=[self.gripper_vels, vel_wp], device=self.gripper_vels.device)

        quat_wp = wp.from_torch(quat.contiguous(), dtype=wp.quat)
        wp.launch(self.set_quat, dim=self.gripper_quat.shape, inputs=[self.gripper_quat, quat_wp], device=self.gripper_quat.device)

        quat_vel_wp = wp.from_torch(quat_vel.contiguous(), dtype=wp.vec3)
        wp.launch(self.set_vec3, dim=self.gripper_quat_vel.shape, inputs=[self.gripper_quat_vel, quat_vel_wp], device=self.gripper_quat_vel.device)

        radius_wp = wp.from_torch(radius.squeeze(-1).contiguous(), dtype=wp.float32)
        wp.launch(self.set_float, dim=self.gripper_radii.shape, inputs=[self.gripper_radii, radius_wp], device=self.gripper_radii.device)

        openness_wp = wp.from_torch(openness.squeeze(-1).contiguous(), dtype=wp.float32)
        wp.launch(self.set_float, dim=self.gripper_open.shape, inputs=[self.gripper_open, openness_wp], device=self.gripper_open.device)

    def export(self):
        data = {}
        if self.gripper_centers.shape[1] > 0:
            gripper_centers = self.gripper_centers.numpy()
            grippers = np.zeros((gripper_centers.shape[0], gripper_centers.shape[1], 15))
            grippers[:, :, :3] = gripper_centers
            grippers[:, :, 3:6] = self.gripper_vels.numpy()
            grippers[:, :, 6:10] = self.gripper_quat.numpy()
            grippers[:, :, 10:13] = self.gripper_quat_vel.numpy()
            grippers[:, :, 13] = self.gripper_radii.numpy()
            grippers[:, :, 14] = self.gripper_open.numpy()
            data['grippers'] = grippers
        return data


@wp.struct
class ConstantBatch(object):

    num_grids_list: wp.vec3
    dt: float
    bound: int
    gravity: wp.vec3
    dx: float
    inv_dx: float
    eps: float
    friction: wp.array(dtype=wp.float32, ndim=2)

    def init(self):
        # self.num_grids = 0
        self.num_grids_list = wp.vec3(0, 0, 0)
        self.dt = 0.0
        self.bound = 0
        self.gravity = wp.vec3(0.0, 0.0, 0.0)
        self.dx = 0.0
        self.inv_dx = 0.0
        self.eps = 0.0
        self.friction = wp.zeros(shape=(1, 1), dtype=wp.float32)
    
    def update_friction(self, friction, requires_grad=False) -> None:
        self.friction = wp.from_torch(friction[..., None].to(torch.float32), dtype=wp.float32)
        self.friction.requires_grad = requires_grad
    
    def zero_grad(self) -> None:
        if self.friction.requires_grad:
            self.friction.grad.zero_()
    
    def to_torch_grad(self) -> Optional[Tensor]:
        return wp.to_torch(self.friction.grad) if self.friction.grad is not None else None

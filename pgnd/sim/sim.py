from typing import Optional
from omegaconf import DictConfig
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
import warp as wp

from .utils import StaticsBatch, CollidersBatch, ParticleDataBatch, GridDataBatch
from .model import SimModelBatch, build_model
from pgnd.utils import Tape, CondTape


class SimFunctionWithFrictionBatch(autograd.Function):

    @staticmethod
    def forward(
            ctx: autograd.function.FunctionCtx,
            model: SimModelBatch,
            statics: StaticsBatch,
            colliders: CollidersBatch,
            particles_curr: ParticleDataBatch,
            particles_next: ParticleDataBatch,
            grid_curr: GridDataBatch,
            grid_next: GridDataBatch,
            friction: Tensor,
            x: Tensor,
            v: Tensor,
            pred: Tensor) -> tuple[Tensor, Tensor]:

        tape: Tape = Tape()
        model.update_friction(friction)
        particles_curr.from_torch(x=x, v=v)
        grid_curr.clear()
        grid_curr.zero_grad()
        grid_curr.from_torch(v=pred)
        grid_next.clear()
        grid_next.zero_grad()

        device = model.device
        constant = model.constant
        batch_size = model.batch_size

        num_grids_list = [int(constant.num_grids_list[0]), int(constant.num_grids_list[1]), int(constant.num_grids_list[2])]
        num_particles = particles_curr.x.shape[1]

        with CondTape(tape, model.requires_grad):
            wp.launch(model.grid_op_batch, 
                dim=[batch_size] + [num_grids_list[0], num_grids_list[1], num_grids_list[2]],
                inputs=[constant, colliders, grid_curr], 
                outputs=[grid_next],
                device=device
            )
            wp.launch(model.g2p_batch, 
                dim=[batch_size, num_particles], 
                inputs=[constant, statics, particles_curr, grid_next], 
                outputs=[particles_next],
                device=device
            )

        x_next, v_next = particles_next.to_torch()

        ctx.tape = tape
        ctx.particles_curr = particles_curr
        ctx.particles_next = particles_next
        ctx.grid_curr = grid_curr

        return x_next, v_next

    @staticmethod
    def backward(
            ctx: autograd.function.FunctionCtx,
            grad_x_next: Tensor,
            grad_v_next: Tensor,
            ) -> tuple[None, None, None, None, None, None, None, None, Tensor, Tensor, Tensor]:

        tape: Tape = ctx.tape
        particles_curr: ParticleDataBatch = ctx.particles_curr
        particles_next: ParticleDataBatch = ctx.particles_next
        grid_curr: GridDataBatch = ctx.grid_curr

        if torch.isnan(grad_x_next).any() or torch.isnan(grad_v_next).any():
            import ipdb; ipdb.set_trace()

        tape.backward(
            grads={
                particles_next.x: wp.from_torch(grad_x_next.contiguous(), dtype=wp.vec3),
                particles_next.v: wp.from_torch(grad_v_next.contiguous(), dtype=wp.vec3),
            }
        )

        grad_x, grad_v = particles_curr.to_torch_grad()
        grad_pred = grid_curr.to_torch_grad()

        if grad_x is not None:
            torch.nan_to_num_(grad_x, 0.0, 0.0, 0.0)
        if grad_v is not None:
            torch.nan_to_num_(grad_v, 0.0, 0.0, 0.0)
        if grad_pred is not None:
            torch.nan_to_num_(grad_pred, 0.0, 0.0, 0.0)

        return None, None, None, None, None, None, None, None, grad_x, grad_v, grad_pred


class CacheDiffSimWithFrictionBatch(nn.Module):

    def __init__(self, cfg: DictConfig, num_steps: int, batch_size: int, device: wp.context.Devicelike = None, requires_grad: bool = False) -> None:
        super().__init__()
        self.model = build_model(cfg=cfg, batch_size=batch_size, device=device, requires_grad=requires_grad)
        self.curr_particles = [None for _ in range(num_steps)]
        self.next_particles = [None for _ in range(num_steps)]
        self.curr_grids = [None for _ in range(num_steps)]
        self.next_grids = [None for _ in range(num_steps)]

    def forward(self, statics: StaticsBatch, colliders: CollidersBatch, step: int, x: Tensor, v: Tensor, friction: Tensor, pred: Tensor) -> tuple[Tensor, Tensor]:
        shape = (x.size(0), x.size(1))
        if self.curr_particles[step] is None:
            self.curr_particles[step] = ParticleDataBatch()
            self.curr_particles[step].init(shape, self.model.device, self.model.requires_grad)
        if self.next_particles[step] is None:
            self.next_particles[step] = ParticleDataBatch()
            self.next_particles[step].init(shape, self.model.device, self.model.requires_grad)
        particles_curr = self.curr_particles[step]
        particles_next = self.next_particles[step]

        shape = (pred.size(0), pred.size(1), pred.size(2), pred.size(3))
        if self.curr_grids[step] is None:
            self.curr_grids[step] = GridDataBatch()
            self.curr_grids[step].init(shape, self.model.device, self.model.requires_grad)
        grid_curr = self.curr_grids[step]
        if self.next_grids[step] is None:
            self.next_grids[step] = GridDataBatch()
            self.next_grids[step].init(shape, self.model.device, self.model.requires_grad)
        grid_next = self.next_grids[step]

        return SimFunctionWithFrictionBatch.apply(self.model, statics, colliders, particles_curr, particles_next, grid_curr, grid_next, friction, x, v, pred)

import torch
import numpy as np
import time
import kornia


def interpolate_motions(bones, motions, relations, xyz, rot=None, quat=None, weights=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3)
    # motions: (n_bones, 3)
    # relations: (n_bones, k)
    # indices: (n_bones,)
    # xyz: (n_particles, 3)
    # rot: (n_particles, 3, 3)
    # quat: (n_particles, 4)
    # weights: (n_particles, n_bones)

    t0 = time.time()
    n_bones, _ = bones.shape
    n_particles, _ = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]
    
    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)
    
    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)
    
    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        import ipdb; ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        import ipdb; ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    if weights is None:
        weights = torch.ones((n_particles, n_bones), device=device)
        dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
        dist = torch.clamp(dist, min=1e-4)
        weights = 1 / dist
        weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    
    # Compute the transformed particles
    xyz_transformed = torch.zeros((n_particles, n_bones, 3), device=device)

    xyz_transformed = xyz[:, None] - bones[None]  # (n_particles, n_bones, 3)
    xyz_transformed = torch.einsum('ijk,jkl->ijl', xyz_transformed, bone_transforms[:, :3, :3].permute(0, 2, 1))  # (n_particles, n_bones, 3)
    xyz_transformed = xyz_transformed + bone_transforms[:, :3, 3][None] + bones[None]  # (n_particles, n_bones, 3)
    xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)

    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)
        rot = quaternion_multiply(quats, quat)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights
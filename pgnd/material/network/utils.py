import torch


def get_grid_locations(x, num_grids_list, dx):
    bsz = x.shape[0]
    x_grid = torch.stack(torch.meshgrid(
        torch.linspace(0, (num_grids_list[0] - 1) * dx, num_grids_list[0], device=x.device),
        torch.linspace(0, (num_grids_list[1] - 1) * dx, num_grids_list[1], device=x.device),
        torch.linspace(0, (num_grids_list[2] - 1) * dx, num_grids_list[2], device=x.device),
    ), dim=-1).reshape(-1, 3)
    grid_idxs = torch.stack(torch.meshgrid(
        torch.linspace(0, (num_grids_list[0] - 1), num_grids_list[0], device=x.device),
        torch.linspace(0, (num_grids_list[1] - 1), num_grids_list[1], device=x.device),
        torch.linspace(0, (num_grids_list[2] - 1), num_grids_list[2], device=x.device),
    ), dim=-1).reshape(-1, 3)

    grid_hits = torch.zeros(bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], device=x.device)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                grid_hits[:, \
                    ((x[:, :, 0] / dx - 0.5).int() + i).clamp(0, num_grids_list[0] - 1), \
                    ((x[:, :, 1] / dx - 0.5).int() + j).clamp(0, num_grids_list[1] - 1), \
                    ((x[:, :, 2] / dx - 0.5).int() + k).clamp(0, num_grids_list[2] - 1)] = 1
    grid_hits = grid_hits.sum(0) > 0
    
    x_grid = x_grid[grid_hits.reshape(-1)].reshape(1, -1, 3).repeat(bsz, 1, 1)
    grid_idxs = grid_idxs[grid_hits.reshape(-1)]
    return x_grid, grid_idxs


def fill_grid_locations(feat, grid_idxs, num_grids_list):  
    # feat: (bsz, num_active_grids, feature_dim)
    # grid_idxs: (num_active_grids, 3)
    bsz = feat.shape[0]
    feat_filled = torch.zeros(bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], 3, device=feat.device)
    grid_idxs_1d = grid_idxs[:, 0] * num_grids_list[1] * num_grids_list[2] + grid_idxs[:, 1] * num_grids_list[2] + grid_idxs[:, 2]
    feat_filled = feat_filled.reshape(bsz, -1, 3)
    feat_filled[:, grid_idxs_1d.long()] = feat.clone()
    feat_filled = feat_filled.reshape(bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], 3)
    return feat_filled

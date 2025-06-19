from pathlib import Path
import argparse
import random
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm, trange
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from datetime import datetime
import numpy as np
from PIL import Image
import warp as wp
import matplotlib.pyplot as plt
import multiprocess as mp
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import logging

from pgnd.utils import get_root, mkdir
from modules_planning.planning_env import RobotPlanningEnv

root: Path = get_root(__file__)
logging.basicConfig(level=logging.WARNING)


def main(args):
    mp.set_start_method('spawn')

    with open(root / args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    cfg = OmegaConf.create(config)

    cfg.sim.num_steps = 1000
    cfg.sim.gripper_forcing = False
    cfg.sim.uniform = True

    iteration = args.iteration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_path = (root / args.config).parent / 'ckpt' / f'{iteration:06d}.pt'
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # path
    datetime_now = datetime.now().strftime('%y%m%d-%H%M%S')
    exp_root: Path = root / 'log' / cfg.train.name / 'plan' / datetime_now
    mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)

    env = RobotPlanningEnv(
        cfg,
        exp_root=exp_root,
        ckpt_path=ckpt_path,
        resolution=(848, 480),
        capture_fps=30,
        record_fps=0,
        text_prompts=args.text_prompts,
        show_annotation=(not args.no_annotation),
        use_robot=True,
        bimanual=args.bimanual,
        gripper_enable=True,
        debug=True,
        construct_target=args.construct_target,
    )

    env.start()
    env.join()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='log/cloth/train/hydra.yaml')
    arg_parser.add_argument('--iteration', type=str, default=100000)
    arg_parser.add_argument('--text_prompts', type=str, default='green towel.')
    arg_parser.add_argument('--seed', type=int, default=42)
    arg_parser.add_argument('--no_annotation', action='store_true')
    arg_parser.add_argument('--bimanual', action='store_true')
    arg_parser.add_argument('--construct_target', action='store_true')
    args = arg_parser.parse_args()

    with torch.no_grad():
        main(args)

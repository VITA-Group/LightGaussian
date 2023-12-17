#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from icecream import ic  
import random
import copy
import gc
from os import makedirs
from prune import prune_list, uniform_prune, HDBSCAN_prune
import csv
import numpy as np

def save_imp_score(dataset, opt, pipe, checkpoint, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iteration = opt.iterations
    ic("Prune iteration: "+str(iteration))
    ic(len(gaussians.get_xyz))
    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background) 
    volume = torch.prod(gaussians.get_scaling, dim = 1)
    index = int(len(volume) * 0.9)
    sorted_volume, sorted_indices = torch.sort(volume, descending=True, dim=0)
    kth_percent_largest = sorted_volume[index]
    v_list = torch.pow(volume/kth_percent_largest, args.v_pow)
    v_list = v_list * imp_list
    v_list = v_list.detach().cpu().numpy()
    np.savez(os.path.join(scene.model_path,"imp_score"), v_list)
    # If you want to check the imp_score:
    if args.show_imp_score:
        data = np.load(os.path.join(scene.model_path,"imp_score.npz"))
        lst = data.files
        for item in lst:
            ic(item)
            ic(data[item].shape)

    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--show_imp_score", action='store_true', default=False)
    parser.add_argument("--get_fps",action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--v_pow", type=float, default=None)


    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    save_imp_score(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args)
    # All done
    print("\nTraining complete.")

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
from prune import prune_list, calculate_v_imp_score
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
    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background) 
    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
    np.savez(os.path.join(scene.model_path,"imp_score"), v_list)
    
    # If you want to print the imp_score:
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
    parser.add_argument("--v_pow", type=float, default=0.1)


    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    save_imp_score(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args)
    # All done
    print("\nTraining complete.")

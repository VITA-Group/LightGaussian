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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from os import makedirs
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import gaussian_poses
from icecream import ic 
import random
import copy
import json
import numpy as np
from utils.logger_utils import prepare_output_and_logger, training_report
from torch.optim.lr_scheduler import ExponentialLR

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
        

to_tensor = lambda x: x.to("cuda") if isinstance(
    x, torch.Tensor) else torch.Tensor(x).to("cuda")
img2mse = lambda x, y: torch.mean((x - y)**2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(to_tensor([10.]))

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, new_max_sh):
    first_iter = 0
    old_sh_degree = dataset.sh_degree
    dataset.sh_degree = new_max_sh
    tb_writer = prepare_output_and_logger(dataset)    
    with torch.no_grad():
        teacher_gaussians = GaussianModel(old_sh_degree) 
        # teacher_gaussians.training_setup(opt)
    
    student_gaussians = GaussianModel(old_sh_degree)
    student_scene = Scene(dataset, student_gaussians)

    if checkpoint:
        (teacher_model_params, _) = torch.load(args.teacher_model)
        (model_params, first_iter) = torch.load(checkpoint)
        teacher_gaussians.restore(teacher_model_params, copy.deepcopy(opt))
        student_gaussians.restore(model_params, opt)
        student_gaussians.max_sh_degree = new_max_sh
        student_gaussians.onedownSHdegree()
    student_gaussians.training_setup(opt)
    student_gaussians.scheduler = ExponentialLR(student_gaussians.optimizer, gamma=0.90)
    # if !args.enable
    if (not args.enable_covariance):
        student_gaussians._scaling.requires_grad = False
        student_gaussians._rotation.requires_grad = False
    if (not args.enable_opacity):
        student_gaussians._opacity.requires_grad = False
        
    teacher_gaussians.optimizer = None
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # os.makedirs(student_scene.model_path + "/vis_data", exist_ok=True)
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, student_gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        student_gaussians.update_learning_rate(iteration)

        # Every 500 iterations step in scheduler
        if iteration % 500 == 0:
            # student_gaussians.oneupSHdegree()
            student_gaussians.scheduler.step()
        
        if not viewpoint_stack:
            viewpoint_stack = student_scene.getTrainCameras().copy()
        viewpoint_cam_org = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = copy.deepcopy(viewpoint_cam_org)

        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        if args.augmented_view:
            viewpoint_cam = gaussian_poses(viewpoint_cam, mean= 0, std_dev_translation=0.05, std_dev_rotation=0)
            student_render_pkg = render(viewpoint_cam, student_gaussians, pipe, background)
            student_image = student_render_pkg["render"]
            teacher_render_pkg = render(viewpoint_cam, teacher_gaussians, pipe, background)        
            teacher_image = teacher_render_pkg["render"].detach() 
        else:
            render_pkg = render(viewpoint_cam, student_gaussians, pipe, background)
            student_image = render_pkg["render"]
            teacher_image = render(viewpoint_cam, teacher_gaussians, pipe, background)["render"].detach() 
        Ll1 = l1_loss(student_image, teacher_image)
        # Ll1 = img2mse(student_image, teacher_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(student_image, teacher_image))
        loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                ic(student_gaussians._features_rest.detach().shape)
                student_scene.save(iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, student_scene, render, (pipe, background))

            # Optimizer step
            if iteration < opt.iterations:
                student_gaussians.optimizer.step()
                student_gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(student_scene.model_path):
                    os.makedirs(student_scene.model_path)
                torch.save((student_gaussians.capture(), iteration), student_scene.model_path + "/chkpnt" + str(iteration) + ".pth")



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[35_001, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--new_max_sh", type=int, default = 2)
    parser.add_argument("--augmented_view", action="store_true")
    parser.add_argument("--enable_covariance", action="store_true")
    parser.add_argument("--enable_opacity", action="store_true")
    parser.add_argument("--opacity_prune", type=float, default = 0)
    parser.add_argument("--teacher_model", type=str)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.new_max_sh)

    # All done
    print("\nTraining complete.")

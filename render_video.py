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

import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from icecream import ic
import copy

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, spherify_poses
# import stepfun 



def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w



def get_focal(camera):
    focal = camera.FoVx
    return focal

def poses_avg_fixed_center(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = [1, 0, 0]
    up = [0, 0, 1]
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt



def render_path_spiral(views, focal=1.5, zrate= 0, rots=1, N=600, tilt_angle=20):
    poses = []
    focal = 0
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        focal += get_focal(views[0])
    poses = np.stack(poses, 0)


    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
    render_poses = []

    rads = np.array(list(rads) + [1.0])
    focal /= len(views)

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta),-np.sin(theta * zrate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))

        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(z, up, c)
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses


def get_circular_view(viewpoint_cam, radius, angle=0.0):
    translate_x = radius * np.cos(angle)
    translate_y = radius * np.sin(angle)
    translate_z = 0

    translate = np.array([translate_x, translate_y, translate_z])
    viewpoint_cam.world_view_transform = torch.tensor(getWorld2View2(viewpoint_cam.R, viewpoint_cam.T, translate)).transpose(0, 1).cuda()
    viewpoint_cam.full_proj_transform = (viewpoint_cam.world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))).squeeze(0)
    viewpoint_cam.camera_center = viewpoint_cam.world_view_transform.inverse()[3, :3]

    return viewpoint_cam

# xy circular 
def render_circular_video(model_path, iteration, views, gaussians, pipeline, background, radius=0.5, num_points=240): 
    render_path = os.path.join(model_path, 'circular', "ours_{}".format(iteration))
    os.makedirs(render_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    # view = views[0]
    for idx in range(num_points):
        view = copy.deepcopy(views[13])
        angle = 2 * np.pi * idx / num_points
        cam = get_circular_view(view, radius, angle)
        rendering = render(cam, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def gaussain_position(viewpoint_cam, mean =0 , std_dev = 0.03):
    translate_x = np.random.normal(mean, std_dev)
    translate_y = np.random.normal(mean, std_dev)
    translate_z = np.random.normal(mean, std_dev)
    translate = np.array([translate_x, translate_y, translate_z])
    viewpoint_cam.world_view_transform = torch.tensor(getWorld2View2(viewpoint_cam.R, viewpoint_cam.T, translate)).transpose(0, 1).cuda()
    viewpoint_cam.full_proj_transform = (viewpoint_cam.world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))).squeeze(0)
    viewpoint_cam.camera_center = viewpoint_cam.world_view_transform.inverse()[3, :3]
    return viewpoint_cam

def render_video(model_path, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]
    # render_path_spiral
    # render_path_spherical
    for idx, pose in enumerate(tqdm(generate_ellipse_path(views,n_frames=600), desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_path_spherical(views, focal=1.5, zrate= -0.05, azimuthal_rots=1, polar_rots=0.75, N=10):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        focal = get_focal(view)
    poses = np.stack(poses, 0)
    # ic(min_focal, max_focal)
    
    c2w = poses_avg(poses)  
    up = normalize(poses[:, :3, 1].sum(0))  
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
    rads = np.array(list(rads) + [1.0])
    ic(rads)
    render_poses = []
    focal_range = np.linspace(0.5, 3, N **2+1)
    index = 0
    # Modify this loop to include phi
    for theta in np.linspace(0.0, 2.0 * np.pi * azimuthal_rots, N + 1)[:-1]:
        for phi in np.linspace(0.0, np.pi * polar_rots, N + 1)[:-1]:
            
            # Modify these lines to use spherical coordinates for c
            c = np.dot(
                c2w[:3, :4],
                rads * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                    1.0
                ])
            )
            
            z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal_range[index], 1.0])))
            render_pose = np.eye(4)
            render_pose[:3] = viewmatrix(z, up, c)  
            
            render_pose[:3, 1:3] *= -1
            render_poses.append(np.linalg.inv(render_pose))
            index += 1

    return render_poses





def gaussain_position(viewpoint_cam, mean =0 , std_dev = 0.03):
    # mean = 0
    # std_dev = 0.03
    translate_x = np.random.normal(mean, std_dev)
    translate_y = np.random.normal(mean, std_dev)
    translate_z = np.random.normal(mean, std_dev)
    translate = np.array([translate_x, translate_y, translate_z])
    viewpoint_cam.world_view_transform = torch.tensor(getWorld2View2(viewpoint_cam.R, viewpoint_cam.T, translate)).transpose(0, 1).cuda()
    viewpoint_cam.full_proj_transform = (viewpoint_cam.world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))).squeeze(0)
    viewpoint_cam.camera_center = viewpoint_cam.world_view_transform.inverse()[3, :3]
    return viewpoint_cam


def gaussian_render(model_path, iteration, views, gaussians, pipeline, background, args):
    views = views[:10] #take the first 10 views and check gaussian view point 
    render_path = os.path.join(model_path, 'video', "gaussians_{}_std{}".format(iteration, args.std))
    makedirs(render_path, exist_ok=True)

    for i, view in enumerate(views):
        rendering = render(view, gaussians, pipeline, background)["render"]
        sub_path = os.path.join(render_path,"view_"+str(i))
        makedirs(sub_path ,exist_ok=True)
        torchvision.utils.save_image(rendering, os.path.join(sub_path, "gt"+'{0:05d}'.format(i) + ".png"))
        for j in range(10):
            n_view = copy.deepcopy(view)
            g_view = gaussain_position(n_view, args.mean, args.std)
            rendering = render(g_view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(sub_path, '{0:05d}'.format(j) + ".png"))



    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, video: bool, circular:bool, radius: float, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        if circular:
            render_circular_video(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        # by default generate ellipse path, other options include spiral, circular, sphrical. Modify it in render_video's enumerate 
        if video:
            render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        #sample virtual view 
        if args.gaussians:
            gaussian_render(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--radius", default=5, type=float)
    parser.add_argument("--gaussians", action="store_true")
    parser.add_argument("--mean", default=0, type=float)
    parser.add_argument("--std", default=0.03, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.video, args.circular, args.radius, args)
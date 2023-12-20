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
from gaussian_renderer import render, count_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2
from icecream import ic
import random
import copy
import gc
import numpy as np
from collections import defaultdict

# from cuml.cluster import HDBSCAN


# def HDBSCAN_prune(gaussians, score_list, prune_percent):
#     # Ensure the tensor is on the GPU and detached from the graph
#     s, d = gaussians.get_xyz.shape
#     X_gpu = cp.asarray(gaussians.get_xyz.detach().cuda())

#     scores_gpu = cp.asarray(score_list.detach().cuda())
#     hdbscan = HDBSCAN(min_cluster_size = 100)
#     cluster_labels = hdbscan.fit_predict(X_gpu)
#     points_by_centroid = {}
#     ic("cluster_labels")
#     ic(cluster_labels.shape)
#     ic(cluster_labels)
#     for i, label in enumerate(cluster_labels):
#         if label not in points_by_centroid:
#             points_by_centroid[label] = []
#         points_by_centroid[label].append(i)
#     points_to_prune = []

#     for centroid_idx, point_indices in points_by_centroid.items():
#         # Skip noise points with label -1
#         if centroid_idx == -1:
#             continue
#         num_to_prune = int(cp.ceil(prune_percent * len(point_indices)))
#         if num_to_prune <= 3:
#             continue
#         point_indices_cp = cp.array(point_indices)
#         distances = scores_gpu[point_indices_cp].squeeze()
#         indices_to_prune = point_indices_cp[cp.argsort(distances)[:num_to_prune]]
#         points_to_prune.extend(indices_to_prune)
#     points_to_prune = np.array(points_to_prune)
#     mask = np.zeros(s, dtype=bool)
#     mask[points_to_prune] = True
#     # points_to_prune now contains the indices of the points to be pruned
#     return mask


# def uniform_prune(gaussians, k, score_list, prune_percent, sample = "k_mean"):
#     # get the farthest_point
#     D, I = None, None
#     s, d = gaussians.get_xyz.shape

#     if sample == "k_mean":
#         ic("k_mean")
#         n_iter = 200
#         verbose = False
#         kmeans = faiss.Kmeans(d, k=k, niter=n_iter, verbose=verbose, gpu=True)
#         kmeans.train(gaussians.get_xyz.detach().cpu().numpy())
#         # The cluster centroids can be accessed as follows
#         centroids = kmeans.centroids
#         D, I = kmeans.index.search(gaussians.get_xyz.detach().cpu().numpy(), 1)
#     else:
#         point_idx = farthest_point_sampler(torch.unsqueeze(gaussians.get_xyz, 0), k)
#         centroids = gaussians.get_xyz[point_idx,: ]
#         centroids = centroids.squeeze(0)
#         index = faiss.IndexFlatL2(d)
#         index.add(centroids.detach().cpu().numpy())
#         D, I = index.search(gaussians.get_xyz.detach().cpu().numpy(), 1)
#     points_to_prune = []
#     points_by_centroid = defaultdict(list)
#     for point_idx, centroid_idx in enumerate(I.flatten()):
#         points_by_centroid[centroid_idx.item()].append(point_idx)
#     for centroid_idx in points_by_centroid:
#         points_by_centroid[centroid_idx] = np.array(points_by_centroid[centroid_idx])
#     for centroid_idx, point_indices in points_by_centroid.items():
#         # Find the number of points to prune
#         num_to_prune = int(np.ceil(prune_percent * len(point_indices)))
#         if num_to_prune <= 3:
#             continue
#         distances = score_list[point_indices].squeeze().cpu().detach().numpy()
#         indices_to_prune = point_indices[np.argsort(distances)[:num_to_prune]]
#         points_to_prune.extend(indices_to_prune)
#     # Convert the list to an array
#     points_to_prune = np.array(points_to_prune)
#     mask = np.zeros(s, dtype=bool)
#     mask[points_to_prune] = True
#     return mask

def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """
    :param gaussians: A data structure containing Gaussian components with a get_scaling method.
    :param imp_list: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(gaussians.get_scaling, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list




def prune_list(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )

    # ic(dataset.model_path)
    for iteration in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
        gc.collect()
    return gaussian_list, imp_list

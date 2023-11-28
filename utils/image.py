import os, sys
import math, random, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio
from pdb import set_trace as st


mse2psnr = (
    lambda x: -10.0 * torch.log(x) / torch.log(torch.tensor([10.0], device=x.device))
)


def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x * mask - y * mask) ** 2) / (torch.sum(mask) + 1e-5)


def img2mae(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x * mask - y * mask)) / (torch.sum(mask) + 1e-5)

# LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

[Project Page](https://lightgaussian.github.io) | [Video](https://youtu.be/470hul75bSM)
<div>
<img src="https://lightgaussian.github.io/static/images/teaser.png" height="250"/>
</div>

Our complete codebase will be released within two weeks.

## Setup
#### Local Setup
Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate lightgaussian
```
note: we modified the "diff-gaussian-rasterization" in submodule to get the Global Significant Score.



## Prunning

After preparing the datasets, users can initiate training from scratch using the following command (ensure to modify the script's path accordingly):
```
bash scripts/run_train_densify_prune.sh
```
This process will generate a point cloud file that is approximately 35% of the size of the original 3D Gaussian splatting, while ensuring a comparable quality level

If you have a trained point cloud already you can start the prune process with command:
```
bash scripts/run_prune_finetune.sh
```



## TODO
- [x] Upload module 1: prune
- [ ] Upload module 2: distillation
- [ ] Upload module 3: Vectree Quantization
- [ ] Upload docker image 


This repo is based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

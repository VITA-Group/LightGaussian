# LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

[Project Page](https://lightgaussian.github.io) | [Video](https://youtu.be/470hul75bSM) |[Paper](https://lightgaussian.github.io/static/paper/LightGaussian_arxiv.pdf)
<div>
<img src="https://lightgaussian.github.io/static/images/teaser.png" height="250"/>
</div>

Our complete codebase will be released within two weeks.

## Setup
#### Local Setup
The codebase is based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

For installation:
```shell
conda env create --file environment.yml
conda activate lightgaussian
```
note: we modified the "diff-gaussian-rasterization" in the submodule to get the Global Significant Score.


## Training

Lightgaussian includes **3 stages** to compress 3D Gaussian

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). 

#### Stage 1 Prune & Recovery
After preparing the datasets, users can initiate training from scratch using the following command (ensure to modify the script's path accordingly):
```
bash scripts/run_train_densify_prune.sh
```
This process will train the Gaussian model for 20,000 iterations and then prune it twice. The resulting point cloud file is approximately 35% of the size of the original 3D Gaussian splatting while ensuring a comparable quality level.

If you have a trained point cloud already you can start the pruning process with the command:
```
bash scripts/run_prune_finetune.sh
```
#### Stage 2 SH distillation
#### Stage 3 Vectree Quantization




## TODO List
- [x] Upload module 1: Prune & recovery 
- [ ] Upload module 2: SH distillation
- [ ] Upload module 3: Vectree Quantization
- [ ] Upload docker image 


## BibTeX
If you find our work useful for your project, please consider citing the following paper.


```
@misc{fan2023lightgaussian, 
title={LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS}, 
author={Zhiwen Fan and Kevin Wang and Kairun Wen and Zehao Zhu and Dejia Xu and Zhangyang Wang}, 
year={2023},
eprint={2311.17245},
archivePrefix={arXiv},
primaryClass={cs.CV} }
```

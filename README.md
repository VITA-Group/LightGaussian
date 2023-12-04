# LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

[Project Page](https://lightgaussian.github.io) | [Video](https://youtu.be/470hul75bSM) |[Paper](https://lightgaussian.github.io/static/paper/LightGaussian_arxiv.pdf)
<div>
<img src="https://lightgaussian.github.io/static/images/teaser.png" height="250"/>
</div>

Our complete codebase will be released within two weeks.

## Setup
#### Local Setup
The codebase is based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

The used datasets, MipNeRF360 and Tank & Temple, are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). 

For installation:
```shell
conda env create --file environment.yml
conda activate lightgaussian
```
note: we modified the "diff-gaussian-rasterization" in the submodule to get the Global Significant Score.


## Compress to Compact Representation

Lightgaussian includes **3 ways** to make the 3D Gaussians be compact
<!-- #### Option 0 Run all (currently Prune + SH distillation) -->


#### Option 1 Prune & Recovery
Users can directly prune a trained 3D-GS checkpoint using the following command (default setting):
```
bash scripts/run_prune_finetune.sh
```

One can also train from scratch and jointly prune redundant Gaussians in training using the following command (different setting from the paper):
```
bash scripts/run_train_densify_prune.sh
```
note: 3D-GS is trained for 20,000 iterations and then prune it. The resulting ply file is approximately 35% of the size of the original 3D-GS while ensuring a comparable quality level.


#### Option 2 SH distillation
#### Option 3 VecTree Quantization


## Render

```
# Render with trajectory. By default ellipse, you can change it to spiral or others trajectory by changing to corresponding function.
python render_video.py --source_path PATH/TO/DATASET --model_path PATH/TO/MODEL --skip_train --skip_test --video 
```


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

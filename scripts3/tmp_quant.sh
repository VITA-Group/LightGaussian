#!/bin/bash

# SCENES=(bicycle bonsai counter garden kitchen room stump train truck)
SCENES=(drone)
VQ_RATIO=0.6
CODEBOOK_SIZE=8192

for SCENE in "${SCENES[@]}"   # Add more scenes as needed
do
    IMP_PATH=./output_drone/
    INPUT_PLY_PATH=./output_drone/point_cloud/iteration_10000/point_cloud.ply
    SAVE_PATH=./vectree/output/drone

    CMD="CUDA_VISIBLE_DEVICES=0 python vectree/vectree.py \
    --important_score_npz_path ${IMP_PATH} \
    --input_path ${INPUT_PLY_PATH} \
    --save_path ${SAVE_PATH} \
    --vq_ratio ${VQ_RATIO} \
    --codebook_size ${CODEBOOK_SIZE} \
    "
    eval $CMD
done
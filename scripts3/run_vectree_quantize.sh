#!/bin/bash

# SCENES=(bicycle bonsai counter garden kitchen room stump train truck)

get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}
# 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.55 0.60 0.65 0.7 0.75 0.8 0.85 0.9
# 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.60 0.65 0.7 0.75 0.8 0.85 0.9
SCENES=( 
    "bicycle"
    "bonsai"
    "counter"
    "kitchen"
    "room"
    "stump"
    "garden"
    "train"
    "truck"
    "flowers"
    "treehill")  # Add more scenes as needed

VQ_RATIOS=(0.6)  # Add more VQ_RATIO values as needed
# 
CODEBOOK_SIZE=8192
# output_3_distill/bicycle--augmented_view_2
# LightGaussian/output_1_org/room
# LightGaussian/output_3_distill/room--augmented_view_2_0.4

for SCENE in "${SCENES[@]}"
do
    for VQ_RATIO in "${VQ_RATIOS[@]}"
    do
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Starting vectree quantize with dataset '$SCENE' and VQ_RATIO '$VQ_RATIO'"
                IMP_PATH=./output_3_distill/${SCENE}--augmented_view_2_0.4
                INPUT_PLY_PATH=./output_3_distill/${SCENE}--augmented_view_2_0.4/point_cloud/iteration_42000/point_cloud.ply
                SAVE_PATH=./output_4_qaunt_xyz/b_${SCENE}_vq_ratio_${VQ_RATIO}
                CUDA_VISIBLE_DEVICES=$gpu_id python vectree/vectree.py \
                --important_score_npz_path ${IMP_PATH} \
                --input_path ${INPUT_PLY_PATH} \
                --save_path ${SAVE_PATH} \
                --vq_ratio ${VQ_RATIO} \
                --sh_degree 2 \
                --codebook_size ${CODEBOOK_SIZE} > "logs_quant/${SCENE}_${VQ_RATIO}.log" 2>&1 &
                sleep 30
                break
            else
                echo "No GPU available at the moment. Retrying in 1 minute."
                sleep 60
            fi
        done
    done
done
wait
echo "All vectree.py runs completed."


# evaluation 
#!/bin/bash
# Function to get an available GPU
get_available_gpu() {
    local mem_threshold=500
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}
# ($1 == 0 ) && 

# Declare the array of arguments
declare -a run_args=(
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
    "treehill"
    # "chair"
    # "drums"
    # "ficus"
    # "hotdog"
    # "lego"
    # "mic"
    # "materials"
    # "ship"
)

declare -a prune_percents=(0.6)
#  0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9

# LightGaussian/output_2_prune/room_0.1
for arg in "${run_args[@]}"; do
    for i in "${!prune_percents[@]}"; do
        prune_percent="${prune_percents[i]}"
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Starting get_fps.py for $arg."
                CUDA_VISIBLE_DEVICES=$gpu_id python get_fps.py -s "/ssd1/zhiwen/datasets/kevin/nerf360/$arg" \
                -m "/ssd1/zhiwen/projects/LightGaussian/output2_baseline/$arg/" \
                --sh_degree 3 \
                --eval  > "logs_result/fps_${arg}_${prune_percent}.log" 2>&1 
                sleep 30
                break
            else
                echo "No available GPU. Sleeping for 60 seconds."
                sleep 60
            fi
        done
    done
done
# --loop

#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=2000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6145

# Datasets
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
)

declare -a virtue_view_arg=(
  ""
  # ""
)
# LightGaussian/output_2_prune
# compress_gaussian/output5_prune_final_result/bicycle_v_important_score_oneshot_prune_densify0.67_vpow0.1_try3_decay1
# compress_gaussian/output2
# b_kitchen_vq_ratio_0.6 
for arg in "${run_args[@]}"; do
  for view in "${virtue_view_arg[@]}"; do
    # Wait for an available GPU
    while true; do
      gpu_id=$(get_available_gpu)
      if [[ -n $gpu_id ]]; then
        echo "GPU $gpu_id is available. Starting quant_train.py with dataset '$arg' and options '$view' on port $port"
        CUDA_VISIBLE_DEVICES=$gpu_id python quant_finetune.py \
          -s "/ssd2/zhiwen/datasets/kevin/nerf360/$arg" \
          --iteration 35500 \
          --eval \
          -m "/ssd2/kevin2/projects/LightGaussian/output_4_qaunt_xyz/b_${arg}_vq_ratio_0.6" \
          --load_vq \
          --sh_degree 2 \
          --position_lr_max_steps 35000 \
          --port $port > "logs_quant_fixed/${arg}.log" 2>&1  &
# background &
        # Increment the port number for the next run
        ((port++))
        # Allow some time for the process to initialize and potentially use GPU memory
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
echo "All quant_train.py runs completed."

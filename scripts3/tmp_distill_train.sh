#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=10000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6025

# Datasets
declare -a run_args=(
    # "bicycle"
    # "bonsai"
    # "counter"
    # "kitchen"
    # "room"
    # "stump"
    # "garden"
    #  "train"
    # "truck"
    "drone"
)


# activate psudo view, else using train view for distillation 
declare -a virtue_view_arg=(
  "--virtual_view"
)


for arg in "${run_args[@]}"; do
  for view in "${virtue_view_arg[@]}"; do
    # Wait for an available GPU
    while true; do
      gpu_id=$(get_available_gpu)
      if [[ -n $gpu_id ]]; then
        echo "GPU $gpu_id is available. Starting distill_train.py with dataset '$arg' and options '$view' on port $port"
        CUDA_VISIBLE_DEVICES=$gpu_id nohup python distill_train.py \
          -s "/ssd1/zhiwen/datasets/drone/UTTOWERDATA" \
          --start_checkpoint "output_drone/chkpnt5000.pth" \
          --iteration 10000 \
          --eval \
          -m "output_drone" \
          --teacher_model "output_drone/chkpnt5000.pth" \
          --new_max_sh 2 \
          --enable_covariance \
          $view \
          --port $port > "logs_distill/distill_${arg}${view}.log" 2>&1 &

        # Increment the port number for the next run
        ((port++))
        # Allow some time for the process to initialize and potentially use GPU memory
        sleep 60
        break
      else
        echo "No GPU available at the moment. Retrying in 1 minute."
        sleep 60
      fi
    done
  done
done
wait
echo "All distill_train.py runs completed."

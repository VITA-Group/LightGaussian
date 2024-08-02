#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=10000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
   $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6041

# Only one dataset specified here, but you could run multiple
declare -a run_args=(
    # "bicycle"
    # "bonsai"
    # "counter"
    # "kitchen"
    # "room"
    # "stump"
    # "garden"
    # "train"
    # "truck"
    # "flowers"
    # "treehill"
    # "chair"
    # "drums"
    # "ficus"
    # "hotdog"
    # "lego"
    # "mic"
    # "materials"
    # "ship"
  )


# Prune percentages and corresponding decays, volume power
declare -a prune_percents=(0.66)
# decay rate for the following prune. The 2nd prune would prune out 0.5 x 0.6 = 0.3 of the remaining gaussian
declare -a prune_decays=(1)
# The volumetric importance power. The higher it is the more weight the volume is in the Global significant
declare -a v_pow=(0.1)

# prune type, by default the Global significant listed in the paper, but there are other option that you can play with
declare -a prune_types=(
  "v_important_score"
  # "important_score"
  # "count"
  )
  # --start_checkpoint "/ssd1/zhiwen/projects/LightGaussian/output2_baseline/$arg/chkpnt30000.pth" \

# Check that prune_percents, prune_decays, and v_pow arrays have the same length
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ] || [ "${#prune_percents[@]}" -ne "${#v_pow[@]}" ]; then
  echo "The lengths of prune_percents, prune_decays, and v_pow arrays do not match."
  exit 1
fi

# Loop over the arguments array
for arg in "${run_args[@]}"; do
  for i in "${!prune_percents[@]}"; do
    prune_percent="${prune_percents[i]}"
    prune_decay="${prune_decays[i]}"
    vp="${v_pow[i]}"

    for prune_type in "${prune_types[@]}"; do
      # Wait for an available GPU
      while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting prune_finetune.py with dataset '$arg', prune_percent '$prune_percent', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"
          
          CUDA_VISIBLE_DEVICES=$gpu_id nohup python prune_finetune.py \
            -s "/ssd1/zhiwen/datasets/kevin/nerf360/$arg/" \
            -m "output_2_prune/${arg}_${prune_percent}" \
            --eval \
            --port $port \
            --start_pointcloud "/ssd1/kevin2/projects/LightGaussian/3dgs_output/${arg}/point_cloud/iteration_30000/point_cloud.ply" \
            --iteration 6000 \
            --save_iterations 6000 \
            --test_iterations 1 6000 \
            --checkpoint_iterations 6000\
            --prune_percent $prune_percent \
            --prune_type $prune_type \
            --prune_decay $prune_decay \
            --position_lr_init 0.000005 \
            --position_lr_max_steps 6500 \
            --v_pow $vp > "logs_prune/${arg}_${prune_percent}prunned.log" 2>&1 &

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
done
wait
echo "All prune_finetune.py runs completed."

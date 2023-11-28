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

# Only one dataset specified here
declare -a run_args=(
    "bicycle"
    # "bonsai"
    # "counter"
    # "kitchen"
    # "room"
    # "stump"
    # "garden"
    # "train"
    # "truck"
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
declare -a prune_percents=(0.40)
declare -a prune_decays=(1)
declare -a v_pow=(0.1)

declare -a prune_types=(
  "v_important_score"
  # "important_score"
  # "count"
  )


# Check that prune_percents and prune_decays arrays have the same length
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ]; then
  echo "The number of prune percents does not match the number of prune decays."
  exit 1
fi

# Loop over the arguments array
for arg in "${run_args[@]}"; do
  for i in "${!prune_percents[@]}"; do
    prune_percent="${prune_percents[i]}"
    prune_decay="${prune_decays[i]}"
    
    for prune_type in "${prune_types[@]}"; do
      for vp in "${v_pow[@]}"; do  # Loop over v_pow values
        
        # Wait for an available GPU
        while true; do
          gpu_id=$(get_available_gpu)
          if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting prune_finetune.py with dataset '$arg', prune_percent '$prune_percent', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"
            
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python prune_finetune.py \
              -s "PATH/TO/DATASET/$arg" \
              -m "OUTPUT/PATH/${arg}_${prune_percent}" \
              --eval \
              --port $port \
              --start_checkpoint "PATH/TO/CHECKPOINT/$arg/chkpnt30000.pth" \
              --iteration 35000 \
              --position_lr_init 0.00016 \
              --position_lr_final 0.0000016 \
              --prune_percent $prune_percent \
              --prune_type $prune_type \
              --prune_decay $prune_decay \
              --feature_lr 0.005 \
              --v_pow $vp > "logs_prune/${arg}${prune_percent}prunned.log" 2>&1 &

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
        
      done  # End loop over v_pow values
    done
  done
done
wait
echo "All prune_finetune.py runs completed."
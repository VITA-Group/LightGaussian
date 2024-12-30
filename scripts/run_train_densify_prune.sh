#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -v threshold="$mem_threshold" -F', ' '
      $2 < threshold { print $1; exit }
    '
}

port=6035

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
)

# prune percentage for the first prune
declare -a prune_percents=(0.6)

# decay rate for the following prune
declare -a prune_decays=(0.6)

# The volumetric importance power
declare -a v_pow=(0.1)

# Prune types
declare -a prune_types=(
  "v_important_score"
)

# Check that prune_percents and prune_decays arrays have the same length
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ]; then
  echo "The number of prune_percents does not match the number of prune_decays."
  exit 1
fi

# Loop over datasets
for arg in "${run_args[@]}"; do
  # Loop over each index in prune_percents/decays/v_pow
  for i in "${!prune_percents[@]}"; do
    prune_percent="${prune_percents[i]}"
    prune_decay="${prune_decays[i]}"
    vp="${v_pow[i]}"

    # Loop over each prune type
    for prune_type in "${prune_types[@]}"; do

      # Wait for an available GPU
      while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting train_densify_prune.py with dataset '$arg', prune_percent '$prune_percent', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"

          CUDA_VISIBLE_DEVICES=$gpu_id nohup python train_densify_prune.py \
            -s "PATH/TO/DATASET/$arg" \
            -m "OUTPUT/PATH/${arg}" \
            --prune_percent "$prune_percent" \
            --prune_decay "$prune_decay" \
            --prune_iterations 20000 \
            --v_pow "$vp" \
            --eval \
            --port "$port" \
            > "logs/train_${arg}.log" 2>&1 &

          # you need to create the log folder first if it doesn't exist
          ((port++))

          # Give the process time to start using GPU memory
          sleep 60
          break
        else
          echo "No GPU available at the moment. Retrying in 1 minute."
          sleep 60
        fi
      done

    done  # end for prune_type
  done    # end for i
done      # end for arg

# Wait for all background processes to finish
wait
echo "All train_densify_prune.py runs completed."

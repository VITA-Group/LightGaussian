# Declare the array of arguments
declare -a run_args=(
    "bicycle"
    "counter"
    "kitchen"
    "room"
    "stump"
    "garden"
    "bonsai"
    "flowers"
    "treehill"
)

# --iteration
# output_d5/bicycle--virtual_view_psudo_distill_sh2
# output5/bicycle_v_important_score_oneshot_prune_densify0.67_vpow0.1_try3_decay1
# Function to get available GPU
get_available_gpu() {
    local mem_threshold=5000
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}
# -m "/ssd1/zhiwen/projects/LightGaussian/output2_baseline/${arg}" \
# LightGaussian/output_3_distill/bonsai--augmented_view_2_0.4
# output_prunned_out/bicycle_prunned1
# output2/garden
# Loop through the arguments and run the Python script on an available GPU
for arg in "${run_args[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if  [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting render_video_${arg}_2"
            CUDA_VISIBLE_DEVICES=$gpu_id python render_video.py -s "/ssd1/zhiwen/datasets/kevin/nerf360/${arg}" \
            -m "output_3_distill/${arg}--augmented_view_2_0.4" \
            --skip_train \
            --skip_test \
            --video \
            --sh_degree 2 \
            --iteration 42000  > "logs/${arg}_render_video1" 2>&1 &
            sleep 60
            break
        else
            echo "No available GPU. Sleeping for 60 seconds."
            sleep 60
        fi
    done
done
wait
echo "All render_video.py runs completed."
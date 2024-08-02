get_available_gpu() {
  local mem_threshold=30000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.60 0.65 0.7 0.75 0.8 0.85 0.9
SCENES=(
    "bicycle"
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
    )  # Add more scenes as needed

# /ssd1/kevin2/projects/LightGaussian/output_4_qaunt/bicycle_vq_ratio_0.1/point_cloud/iteration_40001/point_cloud.ply
# output_3_distill/bicycle--augmented_view_2
# output_4_qaunt/bicycle_vq_ratio_0.1
port=6082
# output_4_qaunt/bicycle_vq_ratio_0.1/point_cloud/iteration_40001/point_cloud.ply
# 0.1
VQ_RATIOS=(0.6)  # Add more VQ_RATIO values as needed
# 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.60 0.65 0.7 0.75 0.8 0.85 0.9 
# 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.60 0.65 0.7 0.75 0.8 0.85 0.9
                # -s "/ssd1/zhiwen/datasets/kevin/nerf360/$SCENES" \
                # --start_pointcloud "./output_4_qaunt/${SCENES}_vq_ratio_${VQ_RATIO}/point_cloud/iteration_40001/point_cloud.ply" \
                # -m "output_4_qaunt/${SCENES}_vq_ratio_${VQ_RATIO}" \

 # -s "/ssd1/zhiwen/datasets/kevin/nerf360/$SCENES" \
# --start_pointcloud "./output_4_qaunt/${SCENES}_vq_ratio_${VQ_RATIO}/point_cloud/iteration_40001/point_cloud.ply" \
# -m "output_4_qaunt/${SCENES}_vq_ratio_${VQ_RATIO}" \
# LightGaussian/output_1_org/room
                # --start_pointcloud "./output_4_qaunt/${SCENES}_vq_ratio_${VQ_RATIO}/point_cloud/iteration_40001/point_cloud.ply" \
                # --sh_degree 3 \

for SCENE in "${SCENES[@]}"
do
    for VQ_RATIO in "${VQ_RATIOS[@]}"
    do
        while true; do
            gpu_id=$(get_available_gpu)
            if [[ -n $gpu_id ]]; then
                echo "GPU $gpu_id is available. Starting vectree quantize with dataset '$SCENE' and VQ_RATIO '$VQ_RATIO'"
                CUDA_VISIBLE_DEVICES=$gpu_id python temp_eval.py \
                -s "/ssd1/zhiwen/datasets/kevin/nerf360/$SCENE" \
                --start_pointcloud "./output_4_qaunt_xyz/b_${SCENE}_vq_ratio_${VQ_RATIO}/point_cloud/iteration_40001/point_cloud.ply" \
                -m "output_4_qaunt_xyz/b_${SCENE}_vq_ratio_${VQ_RATIO}" \
                --sh_degree 2 \
                --iteration 2 \
                --eval \
                --test_iterations 1  \
                --port $port > "logs_result/${SCENE}${VQ_RATIO}_eval.log" 2>&1 &
                # IMP_PATH=./output_3_distill/${SCENE}--augmented_view_2
                # INPUT_PLY_PATH=./output_3_distill/${SCENE}--augmented_view_2/point_cloud/iteration_40000/point_cloud.ply
                # SAVE_PATH=./output_4_qaunt/${SCENE}_vq_ratio_${VQ_RATIO}
                # CUDA_VISIBLE_DEVICES=$gpu_id python vectree/vectree.py \
                # --important_score_npz_path ${IMP_PATH} \
                # --input_path ${INPUT_PLY_PATH} \
                # --save_path ${SAVE_PATH} \
                # --vq_ratio ${VQ_RATIO} \
                # --codebook_size ${CODEBOOK_SIZE} > "logs_quant/${SCENE}_${VQ_RATIOS}.log" 2>&1 &
                 ((port++))
                sleep 20
                break
            else
                echo "No GPU available at the moment. Retrying in 1 minute."
                sleep 60
            fi
        done
    done
done
wait
echo "All evaluation.py runs completed."

# Create an output directory if it doesn't exist
mkdir -p concate_folder

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
# Function to get available GPU



for arg in "${run_args[@]}"; do
# Loop through all PNG images in folder1
    # mkdir -p "concate_folder/${arg}_2"
    # for img in output2/$arg/video/ours_30000/*.png; do
    #     # Extract just the filename without the folder path
    #     img_name=$(basename "$img")
    #     # Concatenate the images side by side
    #     ffmpeg -i "output2/${arg}/video/ours_30000/$img_name" \
    #     -i "output_prunned_out/${arg}_prunned1/video/ours_30002/$img_name" \
    #     -i "output5/${arg}_v_important_score_oneshot_prune_densify0.67_vpow0.1_try3_decay1/video/ours_35000/$img_name" \
    #     -filter_complex "[0:v][1:v][2:v]hstack=inputs=3" "concate_folder/${arg}_2/$img_name"
    # done
    
    ffmpeg -r 60 -i "concate_folder/${arg}/%05d.png" -y -vcodec libx264 -crf 16 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "concate_folder/videos/${arg}_video.mp4" &

    # ffmpeg -r 60 -i "output_d5/garden--virtual_view_psudo_distill_sh2/circular/ours_40001/%05d.png" -y -vcodec libx264 -crf 0 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "output_d5/garden--virtual_view_psudo_distill_sh2/circular/ours_40001/${arg}_video_clip.mp4" &
    # ffmpeg -r 60 -i concate_folder/${arg}/%05d.png -y -vcodec h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p concate_folder/videos/${arg}_video_comp3.mp4 &
    # ffmpeg -start_number 133 -r 60 -i "concate_folder/${arg}/%05d.png" -frames:v 111 -y -vcodec libx264 -crf 1 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "concate_folder/videos/${arg}_video_clip.mp4" &

    # ffmpeg -r 30 -i "concate_folder/${arg}_3/%05d.png" -c:v huffyuv "concate_folder/${arg}_3/${arg}_output_video.avi"

done

#         -i "output_d5/${arg}--virtual_view_psudo_distill_sh2/video/ours_40001/$img_name" \
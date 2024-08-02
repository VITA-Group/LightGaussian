import pandas as pd
import os
import numpy as np

scenes = (
    "bicycle",
    "room",
    "kitchen",
    "stump",
    "bonsai",
    "counter",
    "garden",
    "flowers",
    "treehill",
    # "train",
    # "truck"
    )

#   "train",
#   "truck"
  # "chair",
  # "drums",
  # "ficus",
  # "hotdog",
  # "lego",
  # "mic",
  # "materials",
  # "ship"
#   "kitchen",
#   "stump",
#   "bonsai",
#   "counter",
#   "train",
#   "truck"

# output_d5/bicycle--virtual_view_psudo_distill_sh2/vq_metric1.csv
# _v_important_score_oneshot_prune_densify0.67_vpow0.1_try3_decay1
# output_d5/room--virtual_view_psudo_distill_sh2
# "--virtual_view_psudo_distill_sh2"
# output_d5/bicycle--virtual_view_psudo_distill_sh2
# output_final_quant
# output_d5/bicycle--virtual_view_psudo_distill_sh2/final2_metric.csv
# output_d5/chair--virtual_view--virtual_view_psudo_distill_sh2
# output_prune_nerf/chair_0.50
# output_prune_nerf/drums_0.40/vq3_metric1.csv
# output_d5_final_result/bicycle--virtual_view_psudo_distill_sh2/vq3_metric1.csv
# output_4_qaunt/b_bicycle_vq_ratio_0.6
folder = "output_4_qaunt_xyz"
# f"b_{s}_vq_ratio_0.6
# f"{s}_0.66"
sum_array = None
for i, s in enumerate(scenes):
    path = os.path.join(folder, f"b_{s}_vq_ratio_0.6" , "metric.csv")
    # path = os.path.join(folder, s+ "_v_important_score_oneshot_prune_0.67_vpow0.1_decay1", "metric.csv")
    csv_data = np.genfromtxt(path, delimiter=',')
    if sum_array is None:
        sum_array = np.zeros_like(csv_data)
    sum_array += csv_data

average_array = sum_array / len(scenes)
np.savetxt(folder+"/mip360_v2.csv", average_array, delimiter=",",fmt='%.6f')
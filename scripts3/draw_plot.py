import csv
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import json

def load_fps_from_json(file_path):
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the FPS value
    fps = data['fps']
    return fps


def read_last_line(csv_file):
    with open(csv_file, 'r') as file:
        last_line = file.readlines()[-1]
    return last_line
font = {'family':'sans-serif',
        'size'   : 19}
plt.rc('font', **font)
# Using a different style for aesthetic improvements
style.use('ggplot')
# room_0.1_metric
# output_2_quantize/bicycle_0.1
prune_ratios = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
psnr_values = []
ssim_values = []
lpips_values = []
fpss = []
input_path = "./prune_room_result2/"
output_path = "./room_result/"
for i, ratio in enumerate(prune_ratios):
    file_name = f"room_{ratio}_metric.csv"  # Adjust if file naming is different
    # file_name = f"bicycle_{ratio}/metric.csv"
    last_line = read_last_line(os.path.join(input_path, file_name))
    data = last_line.strip().split(',')
    psnr_values.append(float(data[3]))
    ssim_values.append(float(data[4]))
    lpips_values.append(float(data[5]))
    fpss.append(load_fps_from_json(os.path.join(input_path, f"room_{ratio}_fps.json")))
print(lpips_values)
# print(ssim_values)
# plt.figure(figsize=(10, 6))
# # Plot the remaining data points and lines
# plt.plot(prune_ratios, ssim_values, 'o-', label='GNT', color='#d62728', linewidth=2)
# # Add title and labels
# # plt.title('PSNR vs Noise Levels', fontsize=24)
# plt.xlabel('Gasussian Prune Ratio', fontsize=22, fontweight='bold')
# plt.ylabel('SSIM Values', fontsize=22, fontweight='bold')
# # plt.title('SSIM vs Prune Ratio')
# plt.tick_params(axis='both', which='major', labelsize=18)
# # Setting the y-axis limit
# # plt.ylim(5, None)
# plt.ylim([0.7, 1])  # Set the y-axis range for SSIM
# # Add legend
# # plt.legend(loc='lower left', fontsize=22)
# # Add grid for better readability of the plot
# plt.grid(True, linestyle='--', alpha=0.7)
# # Save the plot as a PNG file
# # plt.savefig('exp/psnr_comparison.svg', bbox_inches='tight')
# plt.savefig(os.path.join(output_path, "prune_SSIM.png"), bbox_inches='tight')

ax1 = plt.gca()  # Get current axis
ax1.plot(prune_ratios, lpips_values, "o-", label="LPIPS", color="#d62728", linewidth=2)
ax1.set_xlabel('Gaussian Prune Ratio', fontsize=22, fontweight='bold')
ax1.set_ylabel("LPIPS Values", fontsize=22, fontweight="bold", color="#d62728")
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.tick_params(axis='y', labelcolor='#d62728')
ax1.set_ylim([0.15, 0.30])  # Set the y-axis range for SSIM
# [0.85, 0.95] for ssim
ax2 = ax1.twinx()
ax2.plot(prune_ratios, fpss, 's-', label='FPS', color='blue', linewidth=2)
ax2.set_ylabel('FPS', fontsize=22, fontweight='bold', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='both', which='major', labelsize=18)

# Set the title and layout
plt.title("LPIPS and FPS vs Gaussian Prune Ratio", fontsize=24, pad=20)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig(
    os.path.join(output_path, "zoom_in_prune_LPIPS_FPS.png"), bbox_inches="tight"
)
plt.savefig(
    os.path.join(output_path, "zoom_in_prune_LPIPS_FPS.svg"), bbox_inches="tight"
)

plt.show()

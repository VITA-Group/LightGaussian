import csv
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import json


# Define file path


font = {'family':'sans-serif',
        'size'   : 19}
plt.rc('font', **font)
# Using a different style for aesthetic improvements
style.use('ggplot')

# Initialize lists
psnr_values = []
ssim_values = []
lpips_values = []
quantize_ratios = []
vq_ratios = []


output_path = "./room_result/"
csv_file_path = './output_4_qaunt/b_all_metric.csv'  # Replace with your actual file path


# Read the CSV file
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        vq_ratios.append(float(row['vq_ratio']))
        psnr_values.append(float(row['psnr']))
        ssim_values.append(float(row['ssim']))
        lpips_values.append(float(row['lpips']))
        quantize_ratios.append(float(row['quantize_ratio']))


ax1 = plt.gca()  # Get current axis
ax1.plot(vq_ratios, lpips_values, "o-", label="LPIPS", color="#d62728", linewidth=2)
ax1.set_xlabel('Gaussian VQ Ratio', fontsize=22, fontweight='bold')
ax1.set_ylabel("LPIPS Values", fontsize=22, fontweight="bold", color="#d62728")
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.tick_params(axis='y', labelcolor='#d62728')
ax1.set_ylim([0.15, 0.30])  # Set the y-axis range for SSIM

ax2 = ax1.twinx()
ax2.plot(vq_ratios, quantize_ratios, 's-', label='Quantize Ratio', color='blue', linewidth=2)
ax2.set_ylabel('Quantize Ratio', fontsize=22, fontweight='bold', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='both', which='major', labelsize=18)

# Set the title and layout
plt.title("LPIPS and Quantize Ratio vs Gaussian VQ Ratio", fontsize=24, pad=20)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig(
    os.path.join(output_path, "zoom_in_quant_LPIPS_vq.png"), bbox_inches="tight"
)
plt.savefig(
    os.path.join(output_path, "zoom_in_quant_LPIPS_vq.svg"), bbox_inches="tight"
)

plt.show()

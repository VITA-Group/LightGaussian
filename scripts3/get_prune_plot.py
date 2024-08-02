import csv
import matplotlib.pyplot as plt
import os
def read_last_line(csv_file):
    with open(csv_file, 'r') as file:
        last_line = file.readlines()[-1]
    return last_line

# output_2_quantize/bicycle_0.1
quantize_ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
psnr_values = []
ssim_values = []
lpips_values = []
# output_4_qaunt/bicycle_vq_ratio_0.1
# path = "./output_2_prune"
path = "./output_4_qaunt"
for i, ratio in enumerate(quantize_ratios):
    file_name = f"bicycle_vq_ratio_{ratio}/metric.csv"  # Adjust if file naming is different
    # file_name = f"bicycle_{ratio}/metric.csv"
    last_line = read_last_line(os.path.join(path, file_name))
    data = last_line.strip().split(',')

    psnr_values.append(float(data[3]))
    ssim_values.append(float(data[4]))
    lpips_values.append(float(data[5]))

plt.figure(figsize=(10, 6))

plt.plot(quantize_ratios, psnr_values, label='PSNR', marker='o')
plt.plot(quantize_ratios, ssim_values, label='SSIM', marker='o')
plt.plot(quantize_ratios, lpips_values, label='LPIPS', marker='o')

plt.xlabel('quantize_ration')
plt.ylabel('Values')
plt.xticks(quantize_ratios)  # Set x-ticks to quantize ratios
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(path, "plot.png"))

plt.figure(figsize=(10, 6))
plt.plot(quantize_ratios, psnr_values, label='PSNR', marker='o', color='blue')
plt.xlabel('Quantize Ratio')
plt.ylabel('PSNR Values')
plt.title('PSNR vs Quantize Ratio')
plt.xticks(quantize_ratios)
# plt.ylim([min(psnr_values)-0.1, max(psnr_values)+1])  # Set the y-axis range for PSNR
plt.grid(True)
plt.savefig(os.path.join(path, "psnr.png"))


plt.figure(figsize=(10, 6))
plt.plot(quantize_ratios, ssim_values, label='SSIM', marker='o', color='green')
plt.xlabel('Quantize Ratio')
plt.ylabel('SSIM Values')
plt.title('SSIM vs Quantize Ratio')
plt.xticks(quantize_ratios)
# plt.ylim([min_ssim-0.3, max_ssim ])  # Set the y-axis range for SSIM
plt.grid(True)
plt.savefig(os.path.join(path, "ssim.png"))

plt.figure(figsize=(10, 6))
plt.plot(quantize_ratios, lpips_values, label='LPIPS', marker='o', color='red')
plt.xlabel('Quantize Ratio')
plt.ylabel('LPIPS Values')
plt.title('LPIPS vs Quantize Ratio')
plt.xticks(quantize_ratios)
plt.grid(True)
plt.savefig(os.path.join(path, "lpips.png"))


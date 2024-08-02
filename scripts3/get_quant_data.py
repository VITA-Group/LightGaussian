import os
import pandas as pd

# Parameters
base_folder = "output_4_qaunt"  # Change this to your base folder path
quantize_ratio_denominator = 140.16848945617676 * 1024 * 1024  # 350 MB in bytes

# Prepare the output DataFrame
all_metrics = pd.DataFrame(columns=['vq_ratio', 'psnr', 'ssim', 'lpips', 'quantize_ratio'])
# LightGaussian/output_4_qaunt
quantize_ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

for i in quantize_ratios:
    # Calculate the ratio for folder name
    ratio = i
    folder_name = f"b_room_vq_ratio_{ratio}"
    
    # Construct the folder path
    folder_path = os.path.join(base_folder, folder_name)
    
    # Read the metric.csv file
    metric_file_path = os.path.join(folder_path, 'metric.csv')
    if os.path.isfile(metric_file_path):
        metric_data = pd.read_csv(metric_file_path)
        
        # Get the last row of the CSV
        last_row = metric_data.iloc[-1]
        
        # Compute the quantize_ratio
        zip_file_path = os.path.join(folder_path, 'extreme_saving.zip')
        if os.path.isfile(zip_file_path):
            file_size = os.path.getsize(zip_file_path)
            quantize_ratio = 1-file_size / quantize_ratio_denominator
        else:
            quantize_ratio = None
        
        # Append the data to the DataFrame
        all_metrics = all_metrics.append({
            'vq_ratio': round(ratio, 4),
            'psnr': round(last_row['psnr'], 4),
            'ssim': round(last_row['ssim'], 4),
            'lpips': round(last_row['lpips'], 4),
            'quantize_ratio': round(quantize_ratio, 4) if quantize_ratio is not None else None
        }, ignore_index=True)

    else:
        print(f"metric.csv not found in {folder_name}")

# Save the combined data to all_metric.csv
all_metrics.to_csv(os.path.join(base_folder, 'b_all_metric.csv'), index=False)

import os
import shutil

scene = 'room'
destination_folder = f'./prune_{scene}_result2'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

source_folders = [f'./output_2_prune/{scene}_{i/20}' for i in range(2, 19, 1)]  # generates numbers from 0.1 to 0.9 with 0.05 increment

for folder in source_folders:
    scene_number = folder.split('_')[-1]  # Extracts the number from the folder name
    
    # Handle metric.csv file
    metric_source_file = os.path.join(folder, 'metric.csv')
    if os.path.exists(metric_source_file):
        shutil.copy(metric_source_file, destination_folder)
        metric_file_in_dest = os.path.join(destination_folder, 'metric.csv')
        new_metric_file_name = f'{scene}_{scene_number}_metric.csv'
        new_metric_file_path = os.path.join(destination_folder, new_metric_file_name)
        os.rename(metric_file_in_dest, new_metric_file_path)
    
    # Handle fps.json file
    fps_source_file = os.path.join(folder, 'fps.json')
    if os.path.exists(fps_source_file):
        shutil.copy(fps_source_file, destination_folder)
        fps_file_in_dest = os.path.join(destination_folder, 'fps.json')
        new_fps_file_name = f'{scene}_{scene_number}_fps.json'
        new_fps_file_path = os.path.join(destination_folder, new_fps_file_name)
        os.rename(fps_file_in_dest, new_fps_file_path)

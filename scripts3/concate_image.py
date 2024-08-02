import cv2
import numpy as np
import glob
import os
from multiprocessing import Pool
from argparse import ArgumentParser
import sys

def create_mask(rows, cols):
    # Vectorized mask creation
    i = np.arange(rows).reshape(rows, 1)
    j = np.arange(cols).reshape(1, cols)
    mask = (j <= cols * i / rows).astype(np.uint8) * 255
    return mask

def process_image(img_path, path1, path2, output_path, text_top, text_bottom):
    img_name = os.path.basename(img_path)
    img1 = cv2.imread(path1 + '/' + img_name)
    img2 = cv2.imread(path2 + '/' + img_name)
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    rows, cols, _ = img1.shape
    mask = create_mask(rows, cols)
    mask_inv = cv2.bitwise_not(mask)

    mask_3d = cv2.merge([mask, mask, mask])
    mask_inv_3d = cv2.merge([mask_inv, mask_inv, mask_inv])

    img1_part = cv2.bitwise_and(img1, mask_3d)
    img2_part = cv2.bitwise_and(img2_resized, mask_inv_3d)

    result = cv2.add(img1_part, img2_part)

    # Draw a white diagonal line
    cv2.line(result, (0, 0), (cols, rows), (255, 255, 255), 2)


    # text_top = "Ours (68MB, 0.7684)"
    # text_bottom = "3D-GS (1081MB, 0.7703)"
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2  # Increase font size
    font_color = (255, 255, 255)  # White text
    font_thickness = 3
    padding = 20  # Increase padding for larger box

    # Calculate the average color of the image for the box
    # average_color_per_row = np.average(result, axis=0)
    # average_color = np.average(average_color_per_row, axis=0)
    # average_color = [int(x) for x in average_color]  # Convert to integer values

    background_color = (0, 0, 0)


    # Calculate text size & position
    (text_width_top, text_height_top), baseline_top = cv2.getTextSize(text_top, font, font_scale, font_thickness)
    (text_width_bottom, text_height_bottom), baseline_bottom = cv2.getTextSize(text_bottom, font, font_scale, font_thickness)
    x_top = (result.shape[1] - text_width_top) // 2
    y_top = text_height_top + padding
    x_bottom = (result.shape[1] - text_width_bottom) // 2
    y_bottom = result.shape[0] - padding

    # Draw background rectangle for top text
    cv2.rectangle(result, (x_top - padding, y_top - text_height_top - padding), 
                (x_top + text_width_top + padding, y_top + baseline_top), background_color, -1)

    # Draw background rectangle for bottom text
    cv2.rectangle(result, (x_bottom - padding, y_bottom - text_height_bottom - baseline_bottom), 
                (x_bottom + text_width_bottom + padding, y_bottom + padding), background_color, -1)

    # Put the text on the image
    cv2.putText(result, text_top, (x_top, y_top), font, font_scale, font_color, font_thickness)
    cv2.putText(result, text_bottom, (x_bottom, y_bottom), font, font_scale, font_color, font_thickness)
    # Save the result
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(output_path + '/' + img_name, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def concat_image(path1, path2, output_path, text_top, text_bottom):
    pattern = path1 + '/*.png'
    img_paths = glob.glob(pattern)

    # Parallel processing
    with Pool() as pool:
        pool.starmap(process_image, [(img_path, path1, path2, output_path,text_top, text_bottom) for img_path in img_paths])




if __name__ == "__main__":
    # parser = ArgumentParser(description="concate_images")
    # parser.add_argument('--path1', type=str, default = "output2/stump/video/ours_30000")
    # parser.add_argument('--path2', type=str, default = "output_d5/stump--virtual_view_psudo_distill_sh2/video/ours_40001")
    # parser.add_argument('--output_path', type=str, default = "concate_folder/stump_1")
    # args = parser.parse_args(sys.argv[1:])
    scenes = (
    "bicycle", 
    "flowers",
    "garden",
    "stump",
    "room", 
    "treehill",
    "counter",
    "kitchen",
    "bonsai",
    # "truck",
    # "train"
  )
    lightgs_text = (
        "Ours (78MB, 0.745)",
        "Ours (43MB, 0.566)",
        "Ours (77MB, 0.845)",
        "Ours (62MB, 0.775)",
        "Ours (20MB, 0.923)",
        "Ours (41MB, 0.627)",
        "Ours (16MB, 0.900)",
        "Ours (24MB, 0.921)",
        "Ours (17MB, 0.938)",
        # "Ours (35MB, 0.862)",
        # "Ours (13MB, 0.772)"
    )

    gs_text = (
        "3D-GS (1334MB, 0.746)",
        "3D-GS (748MB, 0.574)",
        "3D-GS (1334MB, 0.856)",
        "3D-GS (1081MB, 0.770)",
        "3D-GS (353MB, 0.926)",
        "3D-GS (707MB, 0.630)",
        "3D-GS (276MB, 0.914)",
        "3D-GS (412MB, 0.932)",
        "3D-GS (275MB, 0.946)",
        # "3D-GS (551MB, 0.863)",
        # "3D-GS (209MB, 0.781)"
    )
    
# LightGaussian/output_3_distill/flower--augmented_view_2_0.4
# /ssd1/zhiwen/projects/LightGaussian/output2_baseline/bicycle/video/ours_30000/00161.png
    for i, s in enumerate(scenes):
        path1 = os.path.join("/ssd1/zhiwen/projects/LightGaussian/output2_baseline", s, "video/ours_30000")
        path2 = os.path.join("output_3_distill", s + "--augmented_view_2_0.4", "video/ours_42000")
        output_path = os.path.join("concate_folder", s )
        concat_image(path1, path2, output_path,lightgs_text[i], gs_text[i])
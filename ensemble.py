import pandas as pd
import cv2
import matplotlib.pyplot as plt
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

def read_txt_to_df(txt_path):
    with open(txt_path, 'r') as file:
        data = file.read()
    # Split the data into rows and columns
    rows = [line.split() for line in data.strip().split('\n')]

    # Define column names
    columns = ["image", "class_id", "x_center", "y_center", "width", "height", "score"]

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Convert numeric columns to the correct data type
    numeric_columns = ["class_id", "x_center", "y_center", "width", "height", "score"]
    df[numeric_columns] = df[numeric_columns].astype(float)

    return df

def calculate_x1_y1_x2_y2(row):
    # print(row["width"].type)
    x1 = row["x_center"] - row["width"] / 2
    y1 = row["y_center"] - row["height"] / 2
    x2 = row["x_center"] + row["width"] / 2
    y2 = row["y_center"] + row["height"] / 2
    return pd.Series([x1, y1, x2, y2])

df_Co_DETR = read_txt_to_df('predict_codetr_001.txt')
df_YOLO = read_txt_to_df('results_yolo_augment_005.txt')
df_Intern = read_txt_to_df('predict_Intern_full.txt')
df_RYOLO = read_txt_to_df('predict_YOLOR_full_10e.txt')

df_Co_DETR[["x1", "y1", "x2", "y2"]] = df_Co_DETR.apply(calculate_x1_y1_x2_y2, axis=1)
df_Intern[["x1", "y1", "x2", "y2"]] = df_Intern.apply(calculate_x1_y1_x2_y2, axis=1)
df_YOLO[["x1", "y1", "x2", "y2"]] = df_YOLO.apply(calculate_x1_y1_x2_y2, axis=1)
df_RYOLO[["x1", "y1", "x2", "y2"]] = df_RYOLO.apply(calculate_x1_y1_x2_y2, axis=1)




def apply_wbf_to_image(image_name, df1, df2, df3, df4, iou_thr=0.5, skip_box_thr=0.0001):
    # Filter the bounding boxes for the current image from both DataFrames
    bboxes_df1 = df1[df1['image'] == image_name]
    bboxes_df2 = df2[df2['image'] == image_name]
    bboxes_df3 = df3[df3['image'] == image_name]
    bboxes_df4 = df4[df4['image'] == image_name]
    
    # Prepare the input for WBF (list of box coordinates and scores for each model)
    boxes_list = [
        bboxes_df1[['x1', 'y1', 'x2', 'y2']].values.tolist(),
        bboxes_df2[['x1', 'y1', 'x2', 'y2']].values.tolist(),
        bboxes_df3[['x1', 'y1', 'x2', 'y2']].values.tolist(),
        bboxes_df4[['x1', 'y1', 'x2', 'y2']].values.tolist()
    ]
    scores_list = [
        bboxes_df1['score'].values.tolist(),
        bboxes_df2['score'].values.tolist(),
        bboxes_df3['score'].values.tolist(),
        bboxes_df4['score'].values.tolist()
    ]
    labels_list = [
        bboxes_df1['class_id'].values.tolist(),
        bboxes_df2['class_id'].values.tolist(),
        bboxes_df3['class_id'].values.tolist(),
        bboxes_df4['class_id'].values.tolist()
    ]
    
    # Normalize box coordinates (if not already normalized)
    boxes_list = [[
        [x1, y1, x2, y2] for x1, y1, x2, y2 in boxes
    ] for boxes in boxes_list]
    
    # Perform Weighted Box Fusion
    # print(scores_list)
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    
    # print(boxes)
    # Return the fused results
    return boxes, scores, labels

def save_wbf_results_to_txt(output_file, df1, df2, df3, df4):
    # Get all unique image names
    image_names = pd.concat([df1['image'], df2['image'], df3['image'], df4['image']]).unique()
    
    with open(output_file, 'w') as f:
        for image_name in tqdm(image_names):
            # Apply WBF to each image
            boxes, scores, labels = apply_wbf_to_image(image_name, df1, df2, df3, df4)
            
            # Write results to the file in the required format
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                x_center = (x1 + x2)/2
                y_center = (y1 + y2)/2
                width = (x2-x1)
                height = (y2-y1)
                
                f.write(f"{image_name} {int(label)} {x_center} {y_center} {width} {height} {score}\n")

# Example usage
# df1 = pd.read_csv('model1_predictions.csv')  # Replace with your actual DataFrame
# df2 = pd.read_csv('model2_predictions.csv')  # Replace with your actual DataFrame

output_file = 'test.txt'
save_wbf_results_to_txt(output_file, df_Co_DETR, df_RYOLO, df_Intern, df_YOLO)
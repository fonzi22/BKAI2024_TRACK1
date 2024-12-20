import os
import json
import cv2
import argparse
from tqdm import tqdm
import random


def get_image_Id(img_name):
    """
        Calculate imageId from image file
        - Params:
            img_name: image file
        
        - Returns:
            imageId: the id of the image
    """
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId

def get_subpath_after_dataset(full_path, folder_name='dataset'):
    # Split the path based on the given folder name
    if folder_name in full_path:
        return full_path.split(folder_name + os.sep, 1)[-1]
    return None

def yolo_2_coco(images_dirs, labels_dirs, output_file, use_fisheye8k_id=False, use_conf=False, is_submission=False):
    """
        Convert YOLO dataset to COCO json format
        - Params:
            images_dir          :
            labels_dir          :
            output_file         :
            use_fisheye8k_id    :
    """
    categories = []
    images = []
    annotations = []

    # Add categories' ids
    categories.append({"id": 0, "name": "Bike"})
    categories.append({"id": 1, "name": "Car"})
    categories.append({"id": 2, "name": "Bus"})
    categories.append({"id": 3, "name": "Truck"})
    # categories.append({"id": 4, "name": "Truck"})

    image_id = 0
    annotation_id = 0
    for i in tqdm(range(len(images_dirs))):
        # Loop through the image directory
        images_dir = images_dirs[i]
        _, _, images_list = next(os.walk(images_dir))
        labels_dir = labels_dirs[i]
        # if i == len(images_dirs) - 2:
        #     images_list = random.sample(images_list, 1300)
        # if i == len(images_dirs) - 1:
        #     images_list = random.sample(images_list, 900)
        images_list = random.sample(images_list, 10)
        

        for image_file in tqdm(images_list):
            image_path = os.path.join(images_dir, image_file)
            img = cv2.imread(image_path)
            img_h, img_w, img_c = img.shape
            
            if use_fisheye8k_id:
                id = get_image_Id(image_file)
            else:
                id = image_id
                image_id += 1

            images.append({
                "id": id,
                "file_name": get_subpath_after_dataset(images_dir) + '/' + image_file,
                "width": img_w,
                "height": img_h
            })
        
            label_file = image_file[:-4] + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            # if not os.path.exists(label_path):
            #     continue
            with open(label_path, "r") as f:
                bboxes = f.readlines()

            for bbox in bboxes:
                args = bbox.split(" ")
                category_id = int(args[0])
                center_x = int(float(args[1]) * img_w)
                if category_id < 0:
                    print(image_file)
                center_y = int(float(args[2]) * img_h)
                bbox_w = int(float(args[3]) * img_w)
                bbox_h = int(float(args[4]) * img_h)
                if use_conf:
                    score = round(float(args[5]), 6)

                left = int(center_x - bbox_w/2)
                top = int(center_y - bbox_h/2)
                
                annotation_dict = {
                        "id": annotation_id,
                        "category_id": category_id,
                        "image_id": id,
                        "bbox": [left, top, bbox_w, bbox_h],
                        "segmentation": [],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0
                    }
            
                if use_conf:
                    annotation_dict["score"] = score
                
                if is_submission:
                    assert(use_conf == True)
                    del annotation_dict["id"]
                    del annotation_dict["segmentation"]
                    del annotation_dict["area"]
                    del annotation_dict["iscrowd"]
                
                annotations.append(annotation_dict)
                annotation_id += 1
        
    data_dict = {}
    data_dict["categories"] = categories
    data_dict["images"] = images
    data_dict["annotations"] = annotations

    if is_submission:
        with open(output_file, "w") as f:
            json.dump(annotations, f, indent = 4)
    else:
        with open(output_file, "w") as f:
            json.dump(data_dict, f, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset from YOLO format to COCO format")
    parser.add_argument("--images_dir", type=str, default="../visdrone", help="Path to images directory")
    parser.add_argument("--labels_dir", type=str, default="../visdrone", help="Path to labels directory")
    parser.add_argument("--output", type=str, default="./output.json", help="Path to the output json file")
    parser.add_argument("--is_fisheye8k", type=bool, default=False, help="Whether to use the Fisheye8k imageId")
    parser.add_argument("--conf", type=bool, default=False, help="Whether the text files contain confidence scores")
    parser.add_argument("--submission", type=bool, default=False, help="Whether to generate the AI City 2024 submission format")

    args = parser.parse_args()
    # images_dir = args.images_dir
    # labels_dir = args.labels_dir
    # images_dir = ['/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/daytime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/nighttime']
    # labels_dir = ['/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/label_daytime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/label_nighttime']
    images_dir = ['/home/s48gb/Desktop/GenAI4E/test_OD/dataset/style_transfer_images','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train/copy-paste/daytime/images','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train/copy-paste/nighttime/images','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/daytime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/nighttime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/augmented_train/daytime/rain_effect','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/augmented_train/nighttime/rain_effect']
    labels_dir = ['/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/label_daytime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train/copy-paste/label_daytime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train/copy-paste/label_nighttime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/label_daytime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/label_nighttime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/label_daytime','/home/s48gb/Desktop/GenAI4E/test_OD/dataset/train_all/label_nighttime']
    output = 'train_augment_test.json'
    is_fisheye8k = args.is_fisheye8k
    use_conf = args.conf
    is_submission = args.submission

    print(is_fisheye8k)

    yolo_2_coco(images_dir, labels_dir, output, is_fisheye8k, use_conf, is_submission)
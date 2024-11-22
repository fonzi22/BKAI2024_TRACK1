import sys
from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import json
from tqdm import tqdm
import cv2


config_file = 'train/CO-DETR/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py'
checkpoint_file = 'train/CO-DETR/work_dirs/train_all/codetr_augment_1e.pth'

# Khởi tạo mô hình
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Đường dẫn tới thư mục hình ảnh
img_folder = '/dataset/public test'
imgs = [os.path.join(img_folder, img) for img in os.listdir(img_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Danh sách các lớp mà bạn quan tâm
target_classes = ['Bike', 'Car', 'Bus', 'Truck']
mapping = {4: 0, 3: 1, 6: 2, 8: 3}

# Danh sách kết quả cuối cùng
final_results = []


# Duyệt qua từng hình ảnh
for index in tqdm(range(0, len(imgs), 4),desc='Running Inference'):
    images = imgs[index:index+4]
    # print(images)
    # img = cv2.imread(images[0])
    results = inference_detector(model, images)
    # print(results)
    # break
    for j,result in enumerate(results):
        img = images[j] 
        img_name = os.path.basename(img)
        img_id = os.path.splitext(img_name)[0]
    
        img_data = mmcv.imread(img)
        height, width, _ = img_data.shape
        

        # Danh sách các đối tượng trong hình ảnh
        objects = []
        for i, class_name in enumerate(result.pred_instances.labels):
            # print(i, bboxes)
            bboxes = result.pred_instances.bboxes[i]
            score = result.pred_instances.scores[i].item()
            class_id = class_name.item()
            if class_id < 4:
                # for bbox in bboxes:
                # classs_id = mapping[class_id]
                x1, y1, x2, y2 = bboxes.tolist()
                if score >= 0.01:  
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height 
                    obj = {
                        'class_id': class_id,
                        'class_name': target_classes[class_id],
                        'bbox': [x_center, y_center, bbox_width, bbox_height],
                        'score': float(score)
                    }
                    objects.append(obj)
        img_result = {
            'image_id': img_name,
            'objects': objects
        }

        final_results.append(img_result)

with open('results_co-detr.json', 'w') as f:
    json.dump(final_results, f, indent=4)
    
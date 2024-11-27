#**SOICT Hackathon 2024 - Track Traffic Vehicle Detection**

** Chạy docker Co-DETR**
- Pull docker contrainer về từ docker-hub
docker pull anhthuy/co-detr:tagname
- Vào bash và sử dung GPU
```bash
sudo docker run --rm -it --runtime=nvidia --gpus all anhthuy/co-detr:tagname /bin/bash
```

- Inference: Inference co-detr với out put ở /Co-dert/results_test.json đã có file mẫu vì đã run rồi.
```bash 
python inference
```
- Training: Kết quả checkpoint lưu vào /Co-dert/projects/work_dir/train_all/....
```bash
tools/dist_train.sh projects/CO-DETR/configs/codino/train_all.py 1
```

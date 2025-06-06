import os
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET
import torch
import multiprocessing

def main():
    # 检查GPU是否可用
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")

    images_dir = "dataset/images"
    annotations_dir = "dataset/labels"
    output_labels_dir = "dataset/outputs"

    os.makedirs(output_labels_dir, exist_ok=True)

    class_map = {"fire": 0, "smoke": 1}

    def convert_xml_to_yolo(xml_file, output_dir):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        image_w = int(size.find('width').text)
        image_h = int(size.find('height').text)

        label_lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            class_id = class_map.get(name, -1)
            if class_id == -1:
                continue

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            x_center = ((xmin + xmax) / 2) / image_w
            y_center = ((ymin + ymax) / 2) / image_h
            width = (xmax - xmin) / image_w
            height = (ymax - ymin) / image_h

            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        filename = os.path.splitext(os.path.basename(xml_file))[0] + ".txt"
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write("\n".join(label_lines))

    # 转换所有XML文件
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith(".xml"):
            convert_xml_to_yolo(os.path.join(annotations_dir, xml_file), output_labels_dir)

    # 创建目录结构
    Path('yolo_dataset/images/train').mkdir(parents=True, exist_ok=True)
    Path('yolo_dataset/labels/train').mkdir(parents=True, exist_ok=True)
    Path('yolo_dataset/images/val').mkdir(parents=True, exist_ok=True)
    Path('yolo_dataset/labels/val').mkdir(parents=True, exist_ok=True)

    # 分割文件 - 修改为9:1分割
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)
    split_index = int(0.9 * len(image_files))  # 修改为0.9，即90%为训练集
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"总图像数量: {len(image_files)}")
    print(f"训练集数量: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"验证集数量: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")

    # 移动训练文件
    for img_file in train_files:
        label_file = img_file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(images_dir, img_file), f'yolo_dataset/images/train/{img_file}')
        if os.path.exists(os.path.join(output_labels_dir, label_file)):
            shutil.copy(os.path.join(output_labels_dir, label_file), f'yolo_dataset/labels/train/{label_file}')

    # 移动验证文件
    for img_file in val_files:
        label_file = img_file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(images_dir, img_file), f'yolo_dataset/images/val/{img_file}')
        if os.path.exists(os.path.join(output_labels_dir, label_file)):
            shutil.copy(os.path.join(output_labels_dir, label_file), f'yolo_dataset/labels/val/{label_file}')

    # 创建数据配置文件
    data_yaml = """
    path: yolo_dataset
    train: images/train
    val: images/val

    names:
      0: fire
      1: smoke
    """

    with open('yolo_dataset/data.yaml', 'w') as f:
        f.write(data_yaml)

    from ultralytics import YOLO

    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 也可以使用 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'

    # 使用GPU进行训练
    model.train(
        data='yolo_dataset/data.yaml', 
        epochs=100, 
        imgsz=640,
        device=0,  # 使用第一个GPU，可以设置为0,1,2等指定GPU，或'cpu'用CPU训练
        batch=8,  # 根据GPU内存大小调整批次大小
        workers=4,  # 数据加载的工作线程数
        patience=20,  # 早停参数
        save=True,  # 保存模型
        project='fire_smoke_detection',  # 项目名称
        name='yolov8n_training',  # 运行名称
        exist_ok=True,  # 覆盖现有运行
        pretrained=True,  # 使用预训练权重
        optimizer='auto',  # 优化器选择
        augment=True,  # 使用数据增强
        verbose=True,  # 详细输出
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
    )

    # 使用训练好的模型进行预测
    results = model.predict(
        source='yolo_dataset/images/val',  # 验证集图像目录
        save=True,  # 保存带注释的图像
        conf=0.25,  # 置信度阈值
        device=0,  # 使用GPU进行推理
        project='fire_smoke_detection',  # 项目名称
        name='predictions',  # 运行名称
        exist_ok=True,  # 覆盖现有结果
    )

    # 绘制预测结果
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    # 获取预测结果路径
    pred_folder = "yolo_dataset/images/val"

    pred_images = os.listdir(pred_folder)
    pred_images = [img for img in pred_images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 随机选择5张图像显示预测结果
    if len(pred_images) > 5:
        pred_images = random.sample(pred_images, 5)

    # 可视化预测结果
    fig, axs = plt.subplots(len(pred_images), 1, figsize=(12, 4*len(pred_images)))
    if len(pred_images) == 1:
        axs = [axs]

    for i, img_name in enumerate(pred_images):
        img_path = os.path.join(pred_folder, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axs[i].imshow(img_rgb)
        axs[i].set_title(f"预测结果: {img_name}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

    # 打印模型性能指标
    performance = model.metrics
    print("\n模型性能指标:")
    for metric_name, metric_value in performance.items():
        if isinstance(metric_value, (int, float, np.number)):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

if __name__ == '__main__':
    # 添加Windows多进程支持
    multiprocessing.freeze_support()
    main()

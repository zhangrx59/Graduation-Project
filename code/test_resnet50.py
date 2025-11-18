# -*- coding: utf-8 -*-
"""
Test ResNet50 on HAM10000 with pretrained weights
"""

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from PIL import Image
from glob import glob

from ResNet50 import HAM10000Dataset  # 复用已有的Dataset类


def load_model(weight_path, num_classes=7):
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location='cuda'))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ====== 路径设置 ======
    weight_path = "best_resnet50_ham10000.pth"  # 你的权重文件
    root_dir = '../kaggle/input'
    data_dir = os.path.join(root_dir, 'skin-cancer-mnist-ham10000')

    # ====== 数据映射 ======
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['dx']).codes

    # ====== 只使用 test_set ======
    _, df_test = train_test_split(df_original, test_size=0.2, random_state=101)

    # ====== transforms 与训练一致 ======
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_set = HAM10000Dataset(df_test, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # ====== 加载模型 ======
    model = load_model(weight_path).to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.max(1)[1]
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nTest Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)


if __name__ == "__main__":
    main()

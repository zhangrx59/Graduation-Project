# -*- coding: utf-8 -*-
"""
Test ResNet50 on HAM10000 with saved weights
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from ResNet50 import HAM10000Dataset  # 复用 Dataset 类


def load_model(weight_path, num_classes=7, device=torch.device("cpu")):
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ====== 路径设置 ======
    weight_path = "best_resnet50_ham10000.pth"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    root_dir = '../kaggle/input'
    data_dir = os.path.join(root_dir, 'skin-cancer-mnist-ham10000')

    # ====== 统一的标签顺序，与训练脚本一致 ======
    dx_categories = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
    dx_to_idx = {dx: i for i, dx in enumerate(dx_categories)}

    # ====== 数据映射 ======
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type_idx'] = df_original['dx'].map(dx_to_idx)

    print("df_original head:\n", df_original.head())
    print("dx value_counts:\n", df_original['dx'].value_counts())

    # ====== 复现训练脚本里的 df_val / df_test 划分逻辑 ======

    # 1) 按 lesion_id 去重，获取 unduplicated 列表
    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # 标记 duplicates
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        return 'unduplicated' if x in unique_list else 'duplicated'

    df_original['duplicates'] = df_original['lesion_id'].apply(get_duplicates)
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # 2) 从 unduplicated 中划出一部分做 val/test 基底
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    # 3) 用 image_id 把原始数据划分成 train / val
    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df_original['train_or_val'] = df_original['image_id'].apply(get_val_rows)
    df_val = df_original[df_original['train_or_val'] == 'val']

    # 4) 再把 df_val 一分为二 → val + test，与训练脚本一致
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=101)
    df_test = df_test.reset_index(drop=True)

    print("Reconstructed df_test length:", len(df_test))
    print("df_test dx counts:\n", df_test['dx'].value_counts())

    # ====== transforms 与训练一致 ======
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_set = HAM10000Dataset(df_test, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=(device.type == "cuda"))

    # ====== 加载模型 ======
    model = load_model(weight_path, num_classes=len(dx_categories), device=device).to(device)
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
    print(classification_report(y_true, y_pred, target_names=dx_categories))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)


if __name__ == "__main__":
    main()

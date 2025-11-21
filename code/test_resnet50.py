# -*- coding: utf-8 -*-
"""
Test ResNet50 on HAM10000 with saved weights
可视化并保存：
1. Confusion Matrix 热力图
2. 每类疾病的 ROC 曲线（七条曲线一张图）
3. 每类疾病的 Precision-Recall 曲线（七条曲线一张图）
4. 文本版分类报告（打印 + 保存为 txt）
"""

import os
import torch.nn as nn
from ResNet50 import HAM10000Dataset, plot_confusion_matrix, CBAMBlock
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

import matplotlib.pyplot as plt

# 从训练代码复用 Dataset 类和绘制混淆矩阵函数
from ResNet50 import HAM10000Dataset, plot_confusion_matrix


def load_model(weight_path, num_classes=7, device=torch.device("cpu")):
    # 1. 先建一个“干净的” resnet50
    model = resnet50(weights=None)

    # 2. 按训练脚本里的方式，在 layer4 后面接上 CBAM，并包一层 Sequential
    #    这样名字结构就会变成 layer4.0.* 和 layer4.1.ca.*，和保存权重里的 key 一致
    model.layer4 = nn.Sequential(
        model.layer4,                     # 原来的 layer4 (Sequential of Bottleneck blocks)
        CBAMBlock(in_planes=2048, reduction=16, kernel_size=7)
    )

    # 3. 替换最后的全连接层，和训练时保持一致
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 4. 加载权重
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)  # strict=True 确保完全匹配

    return model



def ensure_dir(path):
    """自动创建目录（若不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    # ====== 路径管理 ======
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
    CODE_DIR = os.path.join(BASE_DIR, "code")
    PICS_DIR = os.path.join(BASE_DIR, "pics")
    ensure_dir(PICS_DIR)  # 创建 pics 文件夹

    weight_path = os.path.join(BASE_DIR, "best_resnet50_ham10000_cbam_focal.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ====== 统一类别顺序 ======
    dx_categories = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
    dx_to_idx = {dx: i for i, dx in enumerate(dx_categories)}
    n_classes = len(dx_categories)

    # ====== 数据路径 ======
    root_dir = '../kaggle/input'
    data_dir = os.path.join(root_dir, 'skin-cancer-mnist-ham10000')

    # ====== 加载数据 ======
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type_idx'] = df_original['dx'].map(dx_to_idx)

    # 按训练逻辑复现 df_test
    df_undup = df_original.groupby('lesion_id').apply(lambda x: x.sample(1, random_state=101))
    df_undup = df_undup.reset_index(drop=True)
    df_undup.reset_index(inplace=True)

    def get_duplicates(x):
        return 'unduplicated' if x in list(df_undup['lesion_id']) else 'duplicated'

    df_original['duplicates'] = df_original['lesion_id'].apply(get_duplicates)
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    def get_val_rows(x):
        return 'val' if str(x) in list(df_val['image_id']) else 'train'

    df_original['train_or_val'] = df_original['image_id'].apply(get_val_rows)
    df_val = df_original[df_original['train_or_val'] == 'val']

    _, df_test = train_test_split(df_val, test_size=0.5, random_state=101)

    # ====== 数据预处理 ======
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_set = HAM10000Dataset(df_test, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # ====== 模型加载 ======
    model = load_model(weight_path, num_classes=n_classes, device=device).to(device)
    model.eval()

    y_true, y_pred, y_scores = [], [], []

    # ====== 推理 ======
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_scores.append(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.concatenate(y_scores, axis=0)

    # ====== 文本版报告（整体准确率 + classification_report） ======
    test_acc = (y_true == y_pred).mean()
    print("\n=================== Text Report (Test Set) ===================")
    print(f"Overall test accuracy: {test_acc:.4f}\n")

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=dx_categories,
        digits=4
    )
    print("Classification report on test set:\n")
    print(cls_report)

    # 同时把报告保存成 txt 文件，方便论文/报告使用
    report_path = os.path.join(PICS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Test set classification report\n")
        f.write(f"Overall accuracy: {test_acc:.4f}\n\n")
        f.write(cls_report)
    print(f"[SAVED] Text classification report -> {report_path}")
    print("==============================================================\n")

    # ====== Confusion Matrix 可视化并保存 ======
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, dx_categories, title='Test Confusion Matrix')
    cm_path = os.path.join(PICS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"[SAVED] Confusion Matrix -> {cm_path}")

    # ====== ROC 曲线 ======
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f"{dx_categories[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(PICS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.show()
    print(f"[SAVED] ROC curves -> {roc_path}")

    # ====== Precision-Recall 曲线 ======
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=1.5, label=f"{dx_categories[i]} (AP={ap:.2f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.tight_layout()
    pr_path = os.path.join(PICS_DIR, "pr_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.show()
    print(f"[SAVED] Precision-Recall curves -> {pr_path}")

    # ====== 最后的文字汇总输出 ======
    print("\n=================== Finished ===================")
    print("✅ Text classification report has been printed above.")
    print(f"   And saved to: {report_path}")
    print("✅ All images (Confusion Matrix, ROC Curves, PR Curves) have been saved to:")
    print(f"   📁 {PICS_DIR}")
    print("You can open them for visualization or include them in reports/papers.")
    print("=================================================\n")


if __name__ == "__main__":
    main()

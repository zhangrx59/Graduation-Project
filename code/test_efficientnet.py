# test_efficientnet_b3.py
# -*- coding: utf-8 -*-
"""
Test EfficientNet-B3 on HAM10000 with saved weights.

特点：
1. 按训练脚本（ResNet/MobileNet/EfficientNet）的逻辑重建数据划分：
   - lesion_id 去重
   - 从 undup 中抽 20% 作为 val 基底
   - 用 image_id 标记 train / val
   - 训练集 = train_or_val == 'train'
   - 测试集 = train_or_val == 'val'（不再二次切半）
2. 测试集保持原始不均衡分布，但每一类的测试样本数量比之前多。
3. 评估内容：
   - Overall accuracy
   - classification_report（打印 + 保存为 txt）
   - 混淆矩阵（图）
   - 多类 ROC 曲线（图）
   - 多类 Precision-Recall 曲线（图）
4. 所有输出保存到项目根目录的 pic3/ 文件夹。
"""

import os
from glob import glob
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


# ====================== 随机种子 ======================

np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)


# ====================== Dataset 定义 ======================

class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.loc[index, 'path']
        X = Image.open(img_path).convert("RGB")
        y = torch.tensor(int(self.df.loc[index, 'cell_type_idx']))
        if self.transform:
            X = self.transform(X)
        return X, y


# ====================== EfficientNet-B3 初始化 ======================

def initialize_efficientnet_b3(num_classes, feature_extract=False, use_pretrained=False):
    """
    构建与训练时相同结构的 EfficientNet-B3。
    测试阶段不需要预训练权重，因此 use_pretrained 通常为 False，
    直接 load_state_dict 覆盖即可。
    """
    weights = EfficientNet_B3_Weights.DEFAULT if use_pretrained else None
    model_ft = models.efficientnet_b3(weights=weights)

    if feature_extract:
        for p in model_ft.parameters():
            p.requires_grad = False

    in_features = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(in_features, num_classes)

    return model_ft


# ====================== 工具：混淆矩阵绘图 ======================

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    打印并绘制混淆矩阵。
    normalize=True 显示比例，否则显示计数。
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# ====================== 主流程 ======================

def main():
    print(">>> This is test_efficientnet_b3.py")
    print("CUDA available:", torch.cuda.is_available())

    # 假设 test_efficientnet_b3.py 在 code/ 目录下，项目根目录是上一级
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CODE_DIR)
    PIC3_DIR = os.path.join(BASE_DIR, "pic3")
    ensure_dir(PIC3_DIR)

    weight_path = os.path.join(BASE_DIR, "best_efficientnet_b3_ham10000.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"未找到模型权重文件：{weight_path}\n"
            f"请确认已在训练脚本中使用 torch.save(model.state_dict(), 'best_efficientnet_b3_ham10000.pth') 保存。"
        )

    root_dir = '../kaggle/input'
    print("root_dir content:", os.listdir(root_dir))
    data_dir = os.path.join(root_dir, 'skin-cancer-mnist-ham10000')
    print("data_dir content:", os.listdir(data_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # -------- 2. 构建 df_original（与训练脚本一致） --------
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path
    }

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    print("df_original head:")
    print(df_original.head())

    num_classes = df_original['cell_type_idx'].nunique()

    # 建立 idx -> dx 的映射（用出现最多的 dx 作为该 idx 的标签）
    idx_to_dx = (
        df_original
        .groupby('cell_type_idx')['dx']
        .agg(lambda x: x.value_counts().idxmax())
        .sort_index()
    )
    class_indices = list(idx_to_dx.index)
    target_names = list(idx_to_dx.values)

    print("Class indices:", class_indices)
    print("Target names:", target_names)

    # -------- 3. 按训练脚本逻辑重建 train / test 划分 --------
    # 3.1 找出每个 lesion_id 只出现一次的样本（unduplicated）
    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    print("df_undup head:")
    print(df_undup.head())

    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    df_original['duplicates'] = df_original['lesion_id'].apply(get_duplicates)
    print("duplicates value_counts:")
    print(df_original['duplicates'].value_counts())

    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    print("df_undup shape:", df_undup.shape)

    # 3.2 从 undup 中 stratified 抽 20% 作为 val 基底（与训练一致）
    y = df_undup['cell_type_idx']
    _, df_val_base = train_test_split(
        df_undup,
        test_size=0.2,
        random_state=101,
        stratify=y
    )
    print("df_val_base shape:", df_val_base.shape)
    print("df_val_base cell_type_idx counts:\n",
          df_val_base['cell_type_idx'].value_counts())

    # 3.3 用 image_id 标记 train / val
    def get_val_rows(x):
        val_list = list(df_val_base['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df_original['train_or_val'] = df_original['image_id'].apply(get_val_rows)

    df_train = df_original[df_original['train_or_val'] == 'train']
    df_val_pool = df_original[df_original['train_or_val'] == 'val']

    print("df_train size:", len(df_train))
    print("df_val_pool size (作为测试全集):", len(df_val_pool))
    print("df_val_pool cell_type_idx counts:\n",
          df_val_pool['cell_type_idx'].value_counts().sort_index())

    # 注意：这里不再做 df_val, df_test 的二次划分，
    # 而是直接把所有非训练样本 df_val_pool 当作测试集。
    df_test = df_val_pool.reset_index(drop=True)

    print("Final df_test length:", len(df_test))
    print("df_test cell_type_idx counts:\n",
          df_test['cell_type_idx'].value_counts().sort_index())

    # -------- 4. transform & DataLoader --------
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]

    input_size = 300  # 和训练 EfficientNet-B3 一致
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_set = HAM10000(df_test, transform=test_transform)
    test_loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    # -------- 5. 构建 EfficientNet-B3 并加载权重 --------
    model = initialize_efficientnet_b3(
        num_classes=num_classes,
        feature_extract=False,
        use_pretrained=False  # 这里不需要预训练，因为马上 load_state_dict
    )
    model = model.to(device)

    print("Model structure:")
    print(model)

    print("Loading weights from:", weight_path)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # -------- 6. 在测试集上推理 --------
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_scores.append(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.concatenate(y_scores, axis=0)

    # -------- 7. 文本报告：accuracy + classification_report --------
    test_acc = (y_true == y_pred).mean()
    print("\n=================== EfficientNet-B3 Test Report (Imbalanced Test Set) ===================")
    print(f"Overall test accuracy: {test_acc:.4f}\n")

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )
    print("Classification report on imbalanced test set:\n")
    print(cls_report)

    report_path = os.path.join(PIC3_DIR, "efficientnet_b3_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Test set classification report (EfficientNet-B3, imbalanced test set)\n")
        f.write(f"Overall accuracy: {test_acc:.4f}\n\n")
        f.write(cls_report)
    print(f"[SAVED] Text classification report -> {report_path}")
    print("=================================================================\n")

    # -------- 8. 混淆矩阵 --------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(
        cm,
        classes=target_names,
        normalize=False,
        title='Test Confusion Matrix (EfficientNet-B3, Imbalanced Test Set)'
    )
    cm_path = os.path.join(PIC3_DIR, "efficientnet_b3_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"[SAVED] Confusion Matrix -> {cm_path}")

    # -------- 9. ROC 曲线 --------
    # 将 y_true 转成 one-hot，用于多类 ROC/PR
    y_true_bin = label_binarize(y_true, classes=class_indices)

    plt.figure(figsize=(8, 6))
    for i, cls_idx in enumerate(class_indices):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, cls_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5,
                 label=f"{target_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves (EfficientNet-B3, Imbalanced Test Set)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(PIC3_DIR, "efficientnet_b3_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.show()
    print(f"[SAVED] ROC curves -> {roc_path}")

    # -------- 10. Precision-Recall 曲线 --------
    plt.figure(figsize=(8, 6))
    for i, cls_idx in enumerate(class_indices):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i],
            y_scores[:, cls_idx]
        )
        ap = average_precision_score(y_true_bin[:, i], y_scores[:, cls_idx])
        plt.plot(recall, precision, lw=1.5,
                 label=f"{target_names[i]} (AP={ap:.2f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curves (EfficientNet-B3, Imbalanced Test Set)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    pr_path = os.path.join(PIC3_DIR, "efficientnet_b3_pr_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.show()
    print(f"[SAVED] Precision-Recall curves -> {pr_path}")

    # -------- 11. 结束提示 --------
    print("\n=================== Finished (EfficientNet-B3 Test) ===================")
    print("✅ 测试集按照“不均衡配比 + 最大可用样本数”构建（所有非训练样本都用来测试）。")
    print(f"✅ 文本报告保存到: {report_path}")
    print("✅ 所有可视化(Confusion Matrix / ROC / PR)保存到:")
    print(f"   📁 {PIC3_DIR}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()

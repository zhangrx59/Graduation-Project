# -*- coding: utf-8 -*-
"""
HAM10000 ResNet50 training script (Windows safe, GPU enabled)
"""

# ====== 基本库 ======
import os
import cv2
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# ====== PyTorch ======
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# ====== Sklearn ======
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# =========== 一些工具函数和类 ===========

def set_seed(seed: int = 10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_img_mean_std(image_paths):
    """
    计算整个数据集的三通道 mean/std（不一定要每次运行）
    """
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print("Stacked image shape:", imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR -> RGB
    means.reverse()
    stdevs.reverse()

    print("normMean =", means)
    print("normStd  =", stdevs)
    return means, stdevs


class AverageMeter(object):
    """用于统计 loss / acc 的平均值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


# pytorch Dataset
class HAM10000Dataset(Dataset):
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


def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name: str, num_classes: int,
                     feature_extract: bool, use_pretrained: bool = True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        # 使用新写法，避免 pretrained 警告
        if use_pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
        model_ft = resnet50(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        raise ValueError("Invalid model name: {}".format(model_name))

    return model_ft, input_size


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device,
                    total_loss_train, total_acc_train):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        # 关键：把数据搬到 GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prediction = outputs.max(1, keepdim=True)[1]
        acc = prediction.eq(labels.view_as(prediction)).sum().item() / N

        train_acc.update(acc, N)
        train_loss.update(loss.item(), N)

        if (i + 1) % 100 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)

    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion, epoch, device):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, labels = data
            N = images.size(0)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            acc = prediction.eq(labels.view_as(prediction)).sum().item() / N
            val_acc.update(acc, N)

            loss = criterion(outputs, labels).item()
            val_loss.update(loss, N)

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# =========== 主流程封装到 main() ===========

def main():
    # ========== 0. 随机种子 ==========
    set_seed(10)

    # ========== 1. 数据路径 ==========
    root_dir = '../kaggle/input'
    print("Root dir list:", os.listdir(root_dir))

    data_dir = os.path.join(root_dir, 'skin-cancer-mnist-ham10000')
    print("Data dir list:", os.listdir(data_dir))

    # ========== 2. 构造 image_id -> 路径映射 ==========
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # ========== 3. 使用预先算好的 mean/std ==========
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]

    # ========== 4. 读取 metadata 并构建 DataFrame ==========
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    print("df_original head:\n", df_original.head())

    # ========== 5. 统计每个 lesion_id 的图像数量 ==========
    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    print("df_undup head:\n", df_undup.head())

    # 标记重复 / 不重复
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    df_original['duplicates'] = df_original['lesion_id'].apply(get_duplicates)
    print("df_original with duplicates:\n", df_original.head())
    print("duplicates value_counts:\n", df_original['duplicates'].value_counts())

    # 只保留 unduplicated
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    print("df_undup shape:", df_undup.shape)

    # ========== 6. 从 unduplicated 中划出一部分做 val/test 基底 ==========
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    print("Initial df_val shape:", df_val.shape)
    print("df_val cell_type_idx counts:\n", df_val['cell_type_idx'].value_counts())

    # ========== 7. 用 image_id 把原始数据划分成 train / val ==========
    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df_original['train_or_val'] = df_original['image_id'].apply(get_val_rows)
    df_train = df_original[df_original['train_or_val'] == 'train']
    print("len(df_train):", len(df_train))
    print("len(df_val):", len(df_val))
    print("df_train cell_type_idx counts:\n", df_train['cell_type_idx'].value_counts())
    print("df_val cell_type counts:\n", df_val['cell_type'].value_counts())

    # ========== 8. 过采样平衡各类别 ==========
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    augmented_frames = [df_train]
    for i, rate in enumerate(data_aug_rate):
        if rate > 1:
            df_i = df_train[df_train['cell_type_idx'] == i]
            augmented_frames.append(pd.concat([df_i] * (rate - 1), ignore_index=True))

    df_train = pd.concat(augmented_frames, ignore_index=True)
    print("After oversampling, df_train cell_type counts:\n", df_train['cell_type'].value_counts())

    # ========== 9. 再把 df_val 一分为二 → val + test ==========
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=101)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print("Final df_test length:", len(df_test))
    print("df_test cell_type counts:\n", df_test['cell_type'].value_counts())

    # ========== 10. 初始化模型 ==========
    model_name = "resnet"
    num_classes = 7
    feature_extract = False

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # 设备（这里确定用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True  # 提升卷积性能

    model = model_ft.to(device)

    # ========== 11. 定义 transforms ==========
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # ========== 12. Dataset & DataLoader ==========
    training_set = HAM10000Dataset(df_train, transform=train_transform)
    validation_set = HAM10000Dataset(df_val, transform=val_transform)
    test_set = HAM10000Dataset(df_test, transform=test_transform)

    # CUDA 下推荐 pin_memory=True
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(training_set, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=pin_memory)

    # ========== 13. 优化器 & 损失函数 ==========
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    # ========== 14. 训练循环 ==========
    epoch_num = 10
    best_val_acc = 0.0
    total_loss_train, total_acc_train = [], []
    total_loss_val, total_acc_val = [], []

    for epoch in tqdm(range(1, epoch_num + 1)):
        loss_train, acc_train = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, device,
            total_loss_train, total_acc_train
        )
        loss_val, acc_val = validate(val_loader, model, criterion, epoch, device)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' %
                  (epoch, loss_val, acc_val))
            print('*****************************************************')

    # ========== 15. 简单画训练/验证曲线 ==========
    plt.figure()
    plt.plot(total_acc_val, label='validation accuracy')
    plt.plot(total_loss_val, label='validation loss')
    plt.legend()
    plt.title('Validation curves')
    plt.show()

    plt.figure()
    plt.plot(total_acc_train, label='training accuracy')
    plt.plot(total_loss_train, label='training loss')
    plt.legend()
    plt.title('Training curves')
    plt.show()


# ========= Windows 下必须加这个入口保护 =========
if __name__ == "__main__":
    main()

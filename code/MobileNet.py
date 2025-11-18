# python libraties
import os, cv2, itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ensure results are reproducible
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)


def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


# Values stored to save future run time
norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index]).convert("RGB")
        y = torch.tensor(int(self.df['cell_type_idx'][index]))
        if self.transform:
            X = self.transform(X)
        return X, y


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101 """
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        model_ft = models.resnet50(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        model_ft = models.mobilenet_v2(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 替换最后分类层
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device,
                    total_loss_train, total_acc_train):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
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
        for i, data in tqdm(enumerate(val_loader)):
            images, labels = data
            N = images.size(0)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            acc = prediction.eq(labels.view_as(prediction)).sum().item() / N
            val_acc.update(acc, N)

            val_loss.update(criterion(outputs, labels).item(), N)

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' %
          (epoch, val_loss.avg, val_acc.avg))
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg


def main():
    # 打印一下 GPU
    print("CUDA available:", torch.cuda.is_available())

    print(os.listdir('../kaggle/input/'))

    data_dir = '../kaggle/input/skin-cancer-mnist-ham10000'
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

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
    print(df_original.head())

    # Determine how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    print(df_undup.head())

    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    df_original['duplicates'] = df_original['lesion_id'].apply(get_duplicates)
    print(df_original.head())
    print(df_original['duplicates'].value_counts())

    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    print(df_undup.shape)

    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    print(df_val.shape)
    print(df_val['cell_type_idx'].value_counts())

    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df_original['train_or_val'] = df_original['image_id'].apply(get_val_rows)
    df_train = df_original[df_original['train_or_val'] == 'train']
    print(len(df_train))
    print(len(df_val))
    print(df_train['cell_type'].value_counts())

    # 过采样
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    augmented_frames = [df_train]
    for i, rate in enumerate(data_aug_rate):
        if rate > 1:
            df_i = df_train[df_train['cell_type_idx'] == i]
            augmented_frames.append(pd.concat([df_i] * (rate - 1), ignore_index=True))

    df_train = pd.concat(augmented_frames, ignore_index=True)
    print("After oversampling, df_train cell_type counts:\n", df_train['cell_type'].value_counts())

    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=101)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(len(df_test))
    print(df_test['cell_type'].value_counts())

    # 初始化模型
    model_name = "mobilenet"
    num_classes = 7
    feature_extract = False
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using device: cuda')
        print("GPU name:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("Using device: cpu")

    model = model_ft.to(device)

    # transforms
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

    # Dataset & DataLoader
    training_set = HAM10000(df_train, transform=train_transform)
    validation_set = HAM10000(df_val, transform=val_transform)
    test_set = HAM10000(df_test, transform=test_transform)

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(training_set, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=pin_memory)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    epoch_num = 10
    best_val_acc = 0
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


if __name__ == "__main__":
    # 关键：Windows + DataLoader(num_workers>0) 必须加这一行保护
    main()

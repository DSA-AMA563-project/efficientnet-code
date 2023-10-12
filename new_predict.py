# -*- coding: utf-8 -*-
import os
from multiprocessing import freeze_support

import pandas as pd
import numpy as np

import time
from PIL import Image

import torch

import config

import lightgbm as lgb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一块gpu
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.transforms as transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.utils.data.dataset import Dataset


class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def crop_2(self, img):
        # 以最长的一边为边长，把短的边补为一样长，做成正方形，避免resize时会改变比例
        dowm = img.shape[0]
        up = img.shape[1]
        max1 = max(dowm, up)
        dowm = (max1 - dowm) // 2
        up = (max1 - up) // 2
        dowm_zuo, dowm_you = dowm, dowm
        up_zuo, up_you = up, up
        if (max1 - img.shape[0]) % 2 != 0:
            dowm_zuo = dowm_zuo + 1
        if (max1 - img.shape[1]) % 2 != 0:
            up_zuo = up_zuo + 1
        matrix_pad = np.pad(img, pad_width=((dowm_zuo, dowm_you),  # 向上填充n个维度，向下填充n个维度
                                            (up_zuo, up_you),  # 向左填充n个维度，向右填充n个维度
                                            (0, 0))  # 通道数不填充
                            , mode="constant",  # 填充模式
                            constant_values=(0, 0))
        img = matrix_pad
        return img

    def crop_1(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        # print(img.shape)
        index = np.where(img > 50)  # 找出像素值大于50的所以像素值的坐标
        # print(index)
        x = index[0]
        y = index[1]
        max_x = max(x)
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)
        max_x = max_x + 10
        min_x = min_x - 10
        max_y = max_y + 10
        min_y = min_y - 10
        if max_x > img.shape[0]:
            max_x = img.shape[0]
        if min_x < 0:
            min_x = 0
        if max_y > img.shape[1]:
            max_y = img.shape[1]
        if min_y < 0:
            min_y = 0
        img = img[min_x:max_x, min_y:max_y, :]
        return self.crop_2(img)

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.train_jpg[index]).convert('RGB')

        # Apply cropping here before the transformation to tensor
        img = self.crop_1(self.train_jpg[index])
        img = Image.fromarray(img)  # Convert numpy array back to PIL image

        if self.transform is not None:
            img = self.transform(img)

        label = 0
        if 'CN' in self.train_jpg[index]:
            label = 0
        elif 'AD' in self.train_jpg[index]:
            label = 1
        return img, torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.train_jpg)


class DogeNet(nn.Module):
    def __init__(self):
        super(DogeNet, self).__init__()
        # Load the pretrained EfficientNet-b5
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        # Remove the last fully connected layer
        self.features = self.efficientnet.extract_features
        # Add a global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # Add a fully connected layer for classification
        in_features = self.efficientnet._fc.in_features
        self.fc = nn.Linear(in_features, 2)

    def forward(self, img, feature_extract=False):
        # Extract features
        x = self.features(img)
        # Global average pooling
        x = self.global_avg_pool(x)
        # Flatten the output
        x = x.view(x.size(0), -1)

        # If feature_extract is True, just return the features.
        if feature_extract:
            return x

        # Otherwise, return the class scores.
        out = self.fc(x)
        return out

def test():
    print("Entering test")
    args = config.args

    test_jpg = [args.dataset_test_path + '/{0}.jpg'.format(x) for x in range(1, 100)]
    test_jpg = np.array(test_jpg)

    test_pred = None
    # model_path = 'best_acc_dogenet_b8' + args.v + '.pth'  # 模型名称

    test_loader = torch.utils.data.DataLoader(
        QRDataset(test_jpg,
                  transforms.Compose([
                      # transforms.RandomCrop(128),
                      transforms.RandomRotation(degrees=args.RandomRotation, expand=True),  # 没旋转只有0.85,旋转有0.90
                      transforms.Resize((args.Resize, args.Resize)),
                      transforms.ColorJitter(brightness=args.ColorJitter, contrast=args.ColorJitter,
                                             saturation=args.ColorJitter),  # 加入1
                      # transforms.CenterCrop((450, 450)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True
    )

    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    model = DogeNet().cuda()
    # model.load_state_dict(torch.load(args.save_dir + '/' + model_path))  # 模型文件路径，默认放在args.save_dir下
    # print(args.save_dir)
    # print(model_path)
    # model = nn.DataParallel(model).cuda()
    # 加载LightGBM模型
    bst = lgb.Booster(model_file='model.txt')  # 请替换成您的LightGBM模型文件路径

    test_pred =None

    # 准备测试数据
    # 4. 定义用于存储特征和标签的列表
    test_features_list = []
     #使用模型提取特征
    with torch.no_grad():
            for images, _ in test_loader:  # 注意这里只需要图像，不需要目标标签
                images = images.cuda()
                # 使用模型来提取特征，而不是获得分类得分
                features = model(images, feature_extract=True).squeeze(-1).squeeze(-1)
                test_features_list.append(features.cpu().numpy())

    # 6. 将提取的特征转换为 NumPy 数组
    test_features = np.concatenate(test_features_list, axis=0)
    # 将特征数组转换为DataFrame格式
    df_test = pd.DataFrame(test_features)
    # 使用训练好的LightGBM模型进行预测
    predictions = bst.predict(df_test)

    # 如果预测的是二分类问题，您可以将概率值转换为类别标签（例如，0或1）
    predicted_labels = [1 if pred >= 0.5 else 0 for pred in predictions]

    # 创建一个DataFrame来存储结果
    result_df = pd.DataFrame({'uuid': range(1, len(predicted_labels) + 1),
                              'label': predicted_labels})

    # 将结果保存为CSV文件
    result_df.to_csv('predicted_labels.csv', index=False)
    print("exit")

if __name__ == '__main__':
    freeze_support()
    test()
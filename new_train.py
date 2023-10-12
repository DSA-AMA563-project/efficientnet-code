# -*- coding: utf-8 -*-
import os, sys, glob, argparse

import torch.optim as optim
import numpy as np
from efficientnet_pytorch import EfficientNet
import time
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import torch
import torchvision.models as models
import lightgbm as lgb

import torchvision.transforms as transforms

import torch.nn as nn

from torch.utils.data.dataset import Dataset
import logging
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一块gpu
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#创建日志文件并设置日志记录级别
def write_log(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = log_dir + '/log_dogenet-b8_' + 'v7'+ '.log'
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
    )
    return logging


# 自定义的数据集类，用于加载训练和验证数据
class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

# 将裁剪后的图像填充到正方形形状。
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

#裁剪图像的有用部分，即像素值大于50的部分，并稍微扩大裁剪区域。
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

    # 从文件中加载图像。并根据文件名中的关键字为图像分配标签。
    def __getitem__(self, index):
        start_time = time.time()
        # img = Image.open(self.train_jpg[index]).convert('RGB')
        img = self.crop_1(self.train_jpg[index])

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        label = 0
        if 'CN' in self.train_jpg[index]:
            label = 0
        elif 'AD' in self.train_jpg[index]:
            label = 1
        return img, torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.train_jpg)


# output：模型的输出
# target：真是的目标类别标签
# topk:一个元组，表示要计算的前K个预测准确度
#计算给定输出和目标的准确率。
def accuracy(output, target, topk=(1,)):
    """计算前k个预测的准确率"""
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))  # 修正maxk以匹配类别数和topk请求
        batch_size = target.size(0)

        # 获取前k个预测的索引
        _, pred = output.topk(maxk, 1, True, True)

        # 判断每个预测是否正确
        correct = pred.eq(target.view(-1, 1).expand_as(pred))


        res = []
        for k in topk:
            # 计算前k个预测的准确率
            correct_k = correct[:, :k].float().sum().item()  # 使用item()获取标量值
            res.append(correct_k * 100.0 / batch_size)
        return res


# 用于跟踪和记录不同变量的平均值。用于跟踪和记录训练过程中的各种指标。
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# 用于在训练过程中输出和记录每个批次的进度信息，包括当前批次号、损失值、准确度等指标
class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        logging.info(('\t'.join(entries)))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# 自定义的模型类，使用预训练的 EfficientNet-b5 模型作为基础，
# 并将其输出层替换为适合您的分类任务的输出层（2个类别）。
# class DogeNet(nn.Module):
#     def __init__(self):
#         super(DogeNet, self).__init__()
#         # model = EfficientNet.from_pretrained('efficientnet-b5', weights_path='./model/efficientnet-b5-b6417697.pth')
#         model = EfficientNet.from_pretrained('efficientnet-b5')
#         in_channel = model._fc.in_features
#         model._fc = nn.Linear(in_channel, 2)  # 最后一层全连接层，修改输出层为2个类别
#         self.efficientnet = model

#     def forward(self, img):
#         out = self.efficientnet(img)  # 表示将输入的图像 `img` 通过 EfficientNet 模型进行前向传播，得到模型的输出 `out
#         return out

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


#评估模型在验证数据上的性能。
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            # loss = criterion(output, target)
            loss = CrossEntropyLoss_label_smooth(output, target, num_classes=2)  # 加2 修改输出的类别为2
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))  # 类别改为2
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            top5.update(acc5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.info((' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5)))
        return top1

#使用Test Time Augmentation（TTA）为测试数据集生成预测。
def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

#训练模型的函数。
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    epoch_loss = []
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        # compute output
        output = model(input)
        # loss = criterion(output, target)
        loss = CrossEntropyLoss_label_smooth(output, target, num_classes=2)  # 加2
        # '''warm up module'''
        # if epoch<warm_epoch:
        #     warm_up=min(1.0,warm_up+0.9/warm_iteration)
        #     loss*=warm_up

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))  # 类别数改为2
        losses.update(loss.item(), input.size(0))
        epoch_loss.append(loss.item())
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))  # 更新这里的调用方法

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)
    return np.mean(epoch_loss)

#这是一个实现了标签平滑技巧的交叉熵损失函数。标签平滑是一种正则化技巧，可以防止模型对其预测过于自信。
def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=10, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1))
    targets = targets.data.cpu()
    targets = targets.long();
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1 - epsilon)
    # outputs = outputs.data.cpu()
    log_prob = nn.functional.log_softmax(outputs, dim=1).cpu()
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss



if __name__ == '__main__':
    logging = write_log('v0920_b8')
    print('K={}\tepochs={}\tbatch_size={}\tresume={}\tlr={}'.format(30,10,3,False,0.01))
    logging.info('K={}\tepochs={}\tbatch_size={}\tresume={}\tlr={}'.format(30,10,3,False,0.01))
    k_logger = SummaryWriter('./picture/k/tensorboard/b8_' + 'v7')  # 记录每k折的loss和acc曲线图
    # tensorboard =2.0.0
    train_jpg = np.array(glob.glob('/content/drive/MyDrive/dataset/enhancement/train/*/*.png'))
    skf = KFold(n_splits=10, random_state=233, shuffle=True) #调整k折
    best_acc = 0

    train_features_list = []
    val_features_list = []
    train_labels_list = []
    val_labels_list = []

    for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg, train_jpg)):
        train_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[train_idx],
                      transforms.Compose([
                          # transforms.RandomGrayscale(),
                          transforms.RandomRotation(degrees=180, expand=True),

                          transforms.Resize((224,224)),
                          transforms.RandomAffine(10),
                          transforms.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5),  # 加入1
                          # transforms.ColorJitter(hue=.05, saturation=.05),
                          # transforms.RandomCrop((450, 450)),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),

                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=3, shuffle=True, num_workers=10, pin_memory=True, drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[val_idx],
                      transforms.Compose([
                          # transforms.RandomCrop(128),
                          transforms.RandomRotation(degrees=180, expand=True),

                          transforms.Resize((224,224)),
                          transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                 saturation=0.5),  # 加入1
                          # transforms.Resize((124, 124)),
                          # transforms.RandomCrop((450, 450)),
                          # transforms.RandomCrop((88, 88)),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=3, shuffle=False, num_workers=10, pin_memory=True
        )

        lr = 0.01
        # lr = lr * 0.1
        use_gpu = torch.cuda.is_available()
        print('use_gpu', use_gpu)
        start_epoch = 0
        model = DogeNet().cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        resume=False
        if resume:  # 万一训练中断，可以恢复训练
            checkpoint_path = '/content'+'/v0920_b8' + '/checkpoint' + '/best_new_dogenet_b8' + 'v7' + '.pth.tar'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                start_flod_idx = checkpoint['flod_idx']
                if start_flod_idx > flod_idx:
                    continue
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                print('加载epoch={}成功!,best_acc:{}'.format(checkpoint['epoch'], best_acc))
                logging.info('加载epoch={}成功!,best_acc:{}'.format(checkpoint['epoch'], best_acc))
            else:
                print('无保存模型，重新训练')
                logging.info('无保存模型，重新训练')
        if flod_idx > 0 and not False:  # 从第二折起，迭代前面最好的模型继续训练
            model.load_state_dict(torch.load('v0920_b8' + '/best_acc_dogenet_b8' + 'v7' + '.pth'))
            print('加载最好的模型')
            logging.info('加载最好的模型')
        all_loss = []
        epoch_logger = SummaryWriter(
            './picture/epoch/tensorboard/b8_k=' + str(flod_idx + 1) + '_' + 'v7')  # 记录每个epoch的loss和acc曲线图
        for epoch in range(start_epoch,15):
            print('K/Epoch[{}/{} {}/{}]:'.format(flod_idx, 20, epoch, 30))

            logging.info('K/Epoch[{}/{} {}/{}]:'.format(flod_idx, 20, epoch, 30))
            loss = train(train_loader, model, criterion, optimizer, epoch)
            all_loss.append(loss)
            val_acc = validate(val_loader, model, criterion)
            resume = False
            # if val_acc.avg.item() >= best_acc:
            if val_acc.avg >= best_acc:
                # best_acc = val_acc.avg.item()
                best_acc = val_acc.avg
                torch.save(model.state_dict(), 'v0920_b8' + '/best_acc_dogenet_b8' + 'v7' + '.pth')
            print('best_acc is :{}, lr:{}'.format(best_acc, optimizer.param_groups[0]["lr"]))
            logging.info('best_acc is :{}, lr:{}'.format(best_acc, optimizer.param_groups[0]["lr"]))
            # 保存最新模型
            checkpoint_path = 'v0920_b8' + '/checkpoint'
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'best_acc': best_acc, 'flod_idx': flod_idx, 'scheduler': scheduler.state_dict()}
            torch.save(state, checkpoint_path + '/best_new_dogenet_b8' + 'v7' + '.pth.tar')
            scheduler.step()
            # with epoch_logger.as_default():  # 将acc写入TensorBoard
            # epoch_logger.add_scalar('epoch_loss', loss, step=(epoch + 1))
            # epoch_logger.add_scalar('val_acc', val_acc.avg.item(), step=(epoch + 1))
            epoch_logger.add_scalar('epoch_loss', loss)
            epoch_logger.add_scalar('val_acc', val_acc.avg)
            torch.cuda.empty_cache()
            # 2. 提取特征
            model.eval()

            with torch.no_grad():
                for images, targets in train_loader:
                    images = images.cuda()
                    # 这里使用模型来提取特征，而不是获得分类得分
                    try:
                      features = model(images, feature_extract=True).squeeze(-1).squeeze(-1)
                    except Exception as e:
                      print(f"An error occurred: {str(e)}")

                    train_features_list.append(features.cpu().numpy())
                    train_labels_list.append(targets.numpy())

                for images, targets in val_loader:
                    images = images.cuda()
                    # 这里同样使用模型来提取特征，而不是获得分类得分
                    features = model(images, feature_extract=True).squeeze(-1).squeeze(-1)
                    val_features_list.append(features.cpu().numpy())
                    val_labels_list.append(targets.numpy())


        k_logger.add_scalar('K_loss', np.mean(all_loss))
        k_logger.add_scalar('best_acc', best_acc, )

        # 将特征与标签从列表转换为numpy数组
        train_features = np.concatenate(train_features_list, axis=0)
        val_features = np.concatenate(val_features_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        val_labels = np.concatenate(val_labels_list, axis=0)

        # 将数据转换为DataFrame格式
        df = pd.DataFrame(train_features)
        df['Label'] = train_labels

        # 打印前10行
        print(df.head(10))

        # 保存提取的特征为.npy文件
        np.save('train_features_fold_{}.npy'.format(flod_idx), train_features)
        np.save('val_features_fold_{}.npy'.format(flod_idx), val_features)
        np.save('train_labels_fold_{}.npy'.format(flod_idx), train_labels)
        np.save('val_labels_fold_{}.npy'.format(flod_idx), val_labels)


        # 3. 使用LightGBM进行训练
        train_data = lgb.Dataset(train_features, label=train_labels)
        val_data = lgb.Dataset(val_features, label=val_labels, reference=train_data)
        # 二分类
        param = {'objective': 'binary', 'metric': 'binary_logloss'}
        # num_round = 100
        # bst = lgb.train(param, train_data, num_boost_round=num_round, valid_sets=[val_data], early_stopping_rounds=10)
        # 创建一个早期停止的回调函数
        early_stopping = lgb.early_stopping(stopping_rounds=10, first_metric_only=False, verbose=True)
        num_round=500
        bst = lgb.train(
            param,
            train_data,
            num_boost_round=num_round,
            valid_sets=[val_data],
            callbacks=[early_stopping]


            )


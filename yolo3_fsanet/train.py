import os
import numpy as np
import time
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.config import Config
from torch.utils.data import DataLoader
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss,Generator
from nets.yolo3 import YoloBody
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,writer):
    total_loss = 0
    val_loss = 0
    start_time = time.time()

    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.cuda.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.cuda.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
                idx = iteration + epoch * epoch_size
                writer.add_scalar("Loss_x", loss_item[1], idx)
                writer.add_scalar("Loss_y", loss_item[2], idx)
                writer.add_scalar("Loss_w", loss_item[3], idx)
                writer.add_scalar("Loss_h", loss_item[4], idx)
                writer.add_scalar("Loss_conf", loss_item[5], idx)
                writer.add_scalar("Loss_cls", loss_item[6], idx)
                writer.add_scalar("Loss_yaw", loss_item[7], idx)
                writer.add_scalar("Loss_pitch", loss_item[8], idx)
                writer.add_scalar("Loss_roll", loss_item[9], idx)

            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.cuda.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.cuda.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    logs_train = "/media/2tb/Hoang/multitask/logs/training"
    torch.save(model.state_dict(), logs_train + '/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))



if __name__ == "__main__":

    annotation_path = 'BIWI_train.txt'
    model = YoloBody(Config)

    log_dir = "/media/2tb/Hoang/multitask/runs/training"
    writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
    Cuda = True
    #-------------------------------#
    #   Dataloader
    #-------------------------------#
    Use_Data_Loader = True

    #-------------------------------------------#
    #   load pre-trained model
    #-------------------------------------------#
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("/media/2tb/Hoang/multitask/logs/training/Epoch22-Total_Loss63.1342-Val_Loss157.6234.pth", map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    # loss
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"],[-1,2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))

    # 0.1val，0.9train
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
        # print(lines[:10])
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val


    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        # 最开始使用1e-3的学习率可以收敛的更快
        lr = 1e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 20
        optimizer = optim.Adam(net.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
            val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                             (Config["img_h"], Config["img_w"])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                             (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   freeze backbone of model
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda, writer)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 8
        Freeze_Epoch = 20
        Unfreeze_Epoch = 100
        optimizer = optim.Adam(net.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.95)
        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
            val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                             (Config["img_h"], Config["img_w"])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                             (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   unfrerze backbone
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda, writer)
            lr_scheduler.step()
    writer.close()

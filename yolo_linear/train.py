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
from nets.yolo_training import YOLOLoss
from nets.yolo3 import YoloBody
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

min_val_loss = 100000000


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_ont_epoch(net, yolo_losses, optimizer, epoch, epoch_size, 
                    epoch_size_val, gen, genval, Epoch, cuda, writer=None, mode=""):
    total_loss = 0
    val_loss = 0
    start_time = time.time()

    net.train()
    with tqdm(total=epoch_size, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            num_pos_all = 0
            for i in range(3):
                all_loss, num_pos = yolo_losses[i](outputs[i], targets)
                num_pos_all += num_pos 

                idx = iteration + epoch * epoch_size
                if epoch < 25 or mode == "freeze":
                    losses.append(all_loss["bbox attr"])
                    if writer != None:
                        writer.add_scalar(f"{mode}_backbone/Loss_x", all_loss["x"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_y", all_loss["y"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_w", all_loss["w"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_h", all_loss["h"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_conf", all_loss["confidence"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_cls", all_loss["class"], idx)
                else:
                    losses.append(all_loss["bbox attr"] + all_loss["poses"])
                    if writer != None:
                        writer.add_scalar(f"{mode}_backbone/Loss_x", all_loss["x"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_y", all_loss["y"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_w", all_loss["w"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_h", all_loss["h"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_conf", all_loss["confidence"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_cls", all_loss["class"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_yaw", all_loss["yaw"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_pitch", all_loss["pitch"], idx)
                        writer.add_scalar(f"{mode}_backbone/Loss_roll", all_loss["roll"], idx) 
                               

            loss = sum(losses) / num_pos_all
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{"total_loss": total_loss / (iteration + 1),
                                "lr"        : get_lr(optimizer),
                                "step/s"    : waste_time
                                }
            )
            pbar.update(1)

            start_time = time.time()

    net.eval()
    print("Start Validation")
    with tqdm(total=epoch_size_val, desc=f"Epoch {epoch + 1}/{Epoch}", postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    all_loss, num_pos = yolo_losses[i](outputs[i], targets_val)
                    if epoch < 25 or mode == "freeze":
                        losses.append(all_loss["bbox attr"])
                    else:
                        losses.append(all_loss["bbox attr"] + all_loss["poses"])
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{"total_loss": val_loss / (iteration + 1)})
            pbar.update(1)

    print("Finish Validation")
    print("Epoch:" + str(epoch + 1) + "/" + str(Epoch))
    print("Total Loss: %.4f || Val Loss: %.4f "
        % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    )

    logs_train = "/content/drive/MyDrive/yolo_linear/Multitask_head_pose/yolo_linear/logs"
    weight_name = f"/{mode}Backbone_Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth"% ((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    torch.save(model.state_dict(), logs_train + weight_name)
    torch.save(optimizer.state_dict(), logs_train + f"/{mode}Backbone_Epoch%d_optimizer.pth" % (epoch + 1))
    print(f"Saved {weight_name} weigth successully !!!")
    print("#" + '-'*80 + "#" + '\n' + "#" + '-'*80 + "#")



if __name__ == "__main__":

    annotation_path = "/content/data/CMU_data_origin/CMU_trainval.txt"
    model = YoloBody(Config)
    logTensorBoard_dir = "/content/drive/MyDrive/yolo_linear/Multitask_head_pose/yolo_linear/runs"
    writer = SummaryWriter(log_dir=logTensorBoard_dir)
    Cuda = True
    normalize = False

    # -------------------------------------------#
    #   load pre-trained model
    # -------------------------------------------#
    pre_trained = "/content/drive/MyDrive/yolo_linear/Multitask_head_pose/yolo_linear/logs/unfreezeBackbone_Epoch86-Total_Loss1661.9517-Val_Loss1659.5705.pth"
    if len(pre_trained) != 0:
        print("Loading weights into state dict...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pre_trained, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if np.shape(model_dict[k]) == np.shape(v)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Finished!")

    net = model.train()


    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    # loss
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda, normalize)
        )

    # 0.1val，0.9train
    use_data = 0.1
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
        # print(lines[:10])
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = int(len(lines)) - num_val

    train_lines = lines[:num_train]
    val_lines = lines[:num_val]

    miniTrain_lines = np.random.choice(train_lines, int(num_train * use_data))
    miniVal_lines = np.random.choice(val_lines, int(num_val * use_data))

    # ------------------------------------------------------#
    #   freezing backbone
    #   Init_Epoch = 0
    #   Freeze_Epoch = 30
    #   Epoch = 100
    # ------------------------------------------------------#

    Init_Epoch = 0
    Freeze_Epoch = 50
    optimizer_training = "/content/drive/MyDrive/yolo_linear/Multitask_head_pose/yolo_linear/logs/unfreezeBackbone_Epoch86_optimizer.pth"
    if Init_Epoch < Freeze_Epoch and False:
        f_optimizer = optim.Adam(net.parameters(), 1e-3)
        if len(optimizer_training) != 0:
            print("Loading optimizer into state dict...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            optimizer_dict = f_optimizer.state_dict()
            training_dict = torch.load(optimizer_training, map_location=device)
            training_dict = {k: v  for k, v in training_dict.items()
                if np.shape(optimizer_dict[k]) == np.shape(v)}
            optimizer_dict.update(training_dict)
            f_optimizer.load_state_dict(optimizer_dict)
            print("Finished!")

        Batch_size = 64
        lr_scheduler = optim.lr_scheduler.StepLR(f_optimizer, step_size=1, gamma=0.92)
        train_dataset = YoloDataset(miniTrain_lines, (Config["img_h"], Config["img_w"]), is_train=True)
        val_dataset = YoloDataset(miniVal_lines, (Config["img_h"], Config["img_w"]), is_train=False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4,
            pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,
            pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = len(miniTrain_lines) // Batch_size
        epoch_size_val = len(miniVal_lines) // Batch_size
        # ------------------------------------#
        #   freeze backbone of model
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_ont_epoch(net, yolo_losses, f_optimizer, epoch, epoch_size, \
                epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda, writer, mode="freeze")
            lr_scheduler.step()

    Init_Epoch = 84
    Unfreeze_Epoch = 100
    if True:
        Batch_size = 16
        lr = 1e-4
        optimizer = optim.Adam(net.parameters(), lr)
        if Init_Epoch >= Freeze_Epoch:
            if len(optimizer_training) != 0:
                print("Loading optimizer into state dict...")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                optimizer_dict = optimizer.state_dict()
                training_dict = torch.load(optimizer_training, map_location=device)
                training_dict = {k: v for k, v in training_dict.items()
                    if np.shape(optimizer_dict[k]) == np.shape(v)}
                optimizer_dict.update(training_dict)
                optimizer.load_state_dict(optimizer_dict)
                print("Finished!")

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        train_dataset = YoloDataset(miniTrain_lines, (Config["img_h"], Config["img_w"]), is_train=True)
        val_dataset = YoloDataset(miniVal_lines, (Config["img_h"], Config["img_w"]), is_train=False)
        gen = DataLoader( train_dataset, batch_size=Batch_size, num_workers=4,
            pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,
            pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = len(miniTrain_lines) // Batch_size
        epoch_size_val = len(miniVal_lines) // Batch_size
        # ------------------------------------#
        #   unfrerze backbone
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Init_Epoch, Unfreeze_Epoch):
            fit_ont_epoch(net, yolo_losses, optimizer, epoch, epoch_size, \
                epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda, writer, mode="unfreeze")
            lr_scheduler.step()
    writer.close()

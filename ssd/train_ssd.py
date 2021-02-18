import argparse
import os
import logging
import sys
import itertools
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.autograd import Variable

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import Custom_MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import (
    create_mobilenetv3_large_ssd_lite,
    create_mobilenetv3_small_ssd_lite,
)
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.cmu_dataset import CMUdataset, cmu_dataset_collate
from vision.datasets.collation import object_detection_collate
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import Custom_TrainAugmentation, Custom_TestTransform

parser = argparse.ArgumentParser(description="Single Shot MultiBox Detector Training With Pytorch")

parser.add_argument("--dataset", help="Dataset directory path")
parser.add_argument("--balance_data", action="store_true", help="Balance training data by down-sampling more frequent labels.")


parser.add_argument("--net", default="mb1-ssd", help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.",)
parser.add_argument("--freeze_base_net", action="store_true", help="Freeze base net layers.")
parser.add_argument("--freeze_net", action="store_true", help="Freeze all the layers except the prediction head.")

parser.add_argument("--mb2_width_mult", default=1.0, type=float, help="Width Multiplifier for MobilenetV2")

# Params for SGD
parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay for SGD")
parser.add_argument("--gamma", default=0.1, type=float, help="Gamma update for SGD")
parser.add_argument("--base_net_lr", default=None, type=float, help="initial learning rate for base net.")
parser.add_argument("--extra_layers_lr", default=None, type=float, help="initial learning rate for the layers not in base net and prediction heads.")

# Params for loading pretrained basenet or checkpoints.
parser.add_argument("--base_net", help="Pretrained base model")
parser.add_argument("--pretrained_ssd", help="Pre-trained base model")
parser.add_argument("--resume", default=None, type=str, help="Checkpoint state_dict file to resume training from")

# Scheduler
parser.add_argument("--scheduler", default="multi-step", type=str, help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument("--milestones", default="80,100", type=str, help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument("--t_max", default=120, type=float, help="T_max value for Cosine Annealing Scheduler.")

# Train params
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
parser.add_argument("--num_epochs", default=120, type=int, help="the number epochs")
parser.add_argument("--num_workers", default=4, type=int, help="Number of workers used in dataloading")
parser.add_argument("--validation_epochs", default=5, type=int, help="the number epochs")
parser.add_argument("--debug_steps", default=100, type=int, help="Set the debug log output frequency.")
parser.add_argument("--use_cuda", default=True, type=str2bool, help="Use CUDA to train model")

parser.add_argument("--checkpoint_folder", default="models/", help="Directory for saving checkpoint models")
parser.add_argument("--train_ratio", default=0.8, type=float, help="Split data to training and validating")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    running_pose_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels, poses = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        poses = poses.to(device)

        optimizer.zero_grad()
        confidence, locations, pred_poses = net(images)
        regression_loss, classification_loss, pose_loss = criterion(
            confidence, locations, pred_poses, labels, boxes, poses
        )  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss + pose_loss
        # print(f"[INFO]train_ssd:  classification_loss: {classification_loss}, boxes_loss: {regression_loss}, pose_loss: {pose_loss}")
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        running_pose_loss += pose_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            avg_pose_loss = running_pose_loss / debug_steps

            logging.info(
                f"Epoch: {epoch}, Step: {i}, "
                + f"Average Loss: {avg_loss:.4f}, "
                + f"Average Regression Loss {avg_reg_loss:.4f}, "
                + f"Average Classification Loss: {avg_clf_loss:.4f}, "
                + f"Average Pose Loss: {avg_pose_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            running_pose_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    running_pose_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels, poses = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        poses = poses.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations, pred_poses = net(images)
            regression_loss, classification_loss, pose_loss = criterion(
                confidence, locations, pred_poses, labels, boxes, poses
            )
            loss = regression_loss + classification_loss + pose_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        running_pose_loss += pose_loss.item()
    return (
        running_loss / num,
        running_regression_loss / num,
        running_classification_loss / num,
        running_pose_loss / num,
    )


if __name__ == "__main__":
    timer = Timer()

    logging.info(args)
    if args.net == "vgg16-ssd":
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == "mb1-ssd":
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == "mb1-ssd-lite":
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == "sq-ssd-lite":
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == "mb2-ssd-lite":
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == "mb3-large-ssd-lite":
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        config = mobilenetv1_ssdvalidation_epochs
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = Custom_TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = Custom_MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    test_transform = Custom_TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    dataset_path = args.dataset
    annotation_file = os.path.join(dataset_path, "preprocess_data/CMU_annotate.txt")
    with open(annotation_file, "r") as f:
        lines = f.readlines()
    num_trains = int(len(lines) * args.train_ratio)
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    train_lines = lines[:num_trains]
    val_lines = lines[num_trains:]
    mini_train = num_trains / 100

    train_dataset = CMUdataset(train_lines, train_transform, target_transform)
    val_dataset = CMUdataset(val_lines, test_transform, target_transform)
    # train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
    logging.info("Prepare Validation datasets.")
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,num_workers=args.num_workers, shuffle=True)

    logging.info("Build network.")
    num_class = len(train_dataset.class_names)
    net = create_net(num_class)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = (args.extra_layers_lr if args.extra_layers_lr is not None else args.lr)

    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters(),
            net.regression_headers.parameters(),
            net.classification_headers.parameters(),
            net.regression_poses.parameters()
        )
        params = [
            {
                "params": itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ),
                "lr": extra_layers_lr,
            },
            {
                "params": itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters(),
                    net.regression_poses.parameters()
                )
            },
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters(),
            net.regression_poses.parameters()
        )
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {"params": net.base_net.parameters(), "lr": base_net_lr},
            {
                "params": itertools.chain(
                    net.source_layer_add_ons.parameters(),
                    net.extras.parameters()
                ),
                "lr": extra_layers_lr,
            },
            {
                "params": itertools.chain(
                    net.regression_headers.parameters(),
                    net.classification_headers.parameters(),
                    net.regression_poses.parameters()
                )
            },
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, " + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == "multi-step":
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == "cosine":
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        np.random.shuffle(train_lines)
        x = np.random.rand()/2
        mini_train_lines = train_lines[int(num_trains*x): int(num_trains*x+mini_train)]
        train_dataset = CMUdataset(mini_train_lines, train_transform, target_transform)
        train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)

            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}")

            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")

            net.save(model_path)
            logging.info(f"Saved model {model_path}")

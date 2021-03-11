import cv2
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image
from utils.utils import bbox_iou, jaccard, clip_by_tensor


def MSELoss(pred, target):
    return (pred - target) ** 2


def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def WrappedLoss(pred, target):
    """
    loss for (yaw, pitch, toll)
    """
    loss_1 = (pred - target) ** 2
    loss_2 = (2 - torch.abs(pred - target)) ** 2
    loss = torch.min(loss_1, loss_2)
    return loss


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda, normalize):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.feature_length = [img_size[0] // 32, img_size[0] // 16, img_size[0] // 8]
        self.num_poses = 3
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 0.5
        self.lambda_yaw = 4/3
        self.lambda_pitch = 4/3
        self.lambda_roll = 4/3
        self.cuda = cuda
        self.normalize = normalize

    def forward(self, input, targets=None):
        #--------------------------------------------------------------#
        #   input shape:    bs, 3*(5+num_classes+num_poses), 13, 13
        #                   bs, 3*(5+num_classes+num_poses), 26, 26
        #                   bs, 3*(5+num_classes+num_poses), 52, 52
        #--------------------------------------------------------------#

        # batch size
        bs = input.size(0)
        # grid size
        in_h = input.size(2)
        in_w = input.size(3)

        #--------------------------------------------------------------------------------------------------------#
        # Calculate each feature point corresponds to how many pixels on the original picture
        # If the feature layer is 13x13, one feature point corresponds to 32 pixels on the original image
        # If the feature layer is 26x26, one feature point corresponds to 16 pixels on the original image
        # If the feature layer is 52x52, a feature point corresponds to 18 pixels on the original image
        # stride_h = stride_w = 32, 16, 8
        #--------------------------------------------------------------------------------------------------------#
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        # Calculate anchors follow stride
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        #----------------------------------------------------------------------------------------#
        # bs, 3*(5+num_classes+num_poses), 13, 13 -> bs, 3, 13, 13, (5+num_classes+num_poses)
        # bs, 3*(5+num_classes+num_poses), 26, 26 -> bs, 3, 13, 13, (5+num_classes+num_poses)
        # bs, 3*(5+num_classes+num_poses), 52, 52 -> bs, 3, 13, 13, (5+num_classes+num_poses)
        #----------------------------------------------------------------------------------------#
        prediction = input.view(bs, int(self.num_anchors / 3), 
            self.bbox_attrs + self.num_poses, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Adjustment parameters for the center position of the prior box
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        # The width and height adjustment parameters of the prior box
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        # confidence
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        # class
        pred_cls = torch.sigmoid(prediction[..., 5 : 5 + self.num_classes])
        # poses
        pred_yaw = torch.tanh(prediction[..., 6])
        pred_pitch = torch.tanh(prediction[..., 7])
        pred_roll = torch.tanh(prediction[..., 8])

        #-----------------------------------------------------------------------------------#
        #   Find which prior boxes contain objects
        #   Calculate the intersection ratio using the real box and the prior box
        #   mask        batch_size, 3, in_h, in_w   objective feature points
        #   noobj_mask  batch_size, 3, in_h, in_w   no objective feature point
        #   tx          batch_size, 3, in_h, in_w   Center x offset
        #   ty          batch_size, 3, in_h, in_w   Center y offset
        #   tw          batch_size, 3, in_h, in_w   The true value of the width adjustment
        #   th          batch_size, 3, in_h, in_w   The true value of the height adjustment
        #   tconf       batch_size, 3, in_h, in_w   True value of confidence
        #   tcls        batch_size, 3, in_h, in_w, num_classes  true value of class
        #   yaw        batch_size, 3, in_h, in_w, num_classes  true value of yaw
        #   pitch        batch_size, 3, in_h, in_w, num_classes  true value of pitch
        #   roll        batch_size, 3, in_h, in_w, num_classes  true value of roll
        #------------------------------------------------------------------------------------#
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, yaw, pitch, roll, \
            box_loss_scale_x, box_loss_scale_y = self.get_target(
                targets, scaled_anchors, in_w, in_h, self.ignore_threshold)
        """        
        Decode the prediction result and judge the degree of 
            overlap between the prediction result and the true value
        """
        noobj_mask = self.get_ignore(
            prediction, targets, scaled_anchors, in_w, in_h, noobj_mask
        )
        if self.cuda:
            box_loss_scale_x = (box_loss_scale_x).cuda()
            box_loss_scale_y = (box_loss_scale_y).cuda()
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            yaw, pitch, roll = yaw.cuda(), pitch.cuda(), roll.cuda()

        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y

        #  losses.
        loss_x = torch.sum(BCELoss(x, tx) * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) * box_loss_scale * mask)
        loss_w = torch.sum(MSELoss(w, tw) * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) * 0.5 * box_loss_scale * mask)

        loss_conf = torch.sum(BCELoss(conf, mask) * mask) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask)
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]))

        loss_yaw = torch.sum(WrappedLoss(pred_yaw, yaw) * mask)
        loss_pitch = torch.sum(MSELoss(pred_pitch, pitch) * mask)
        loss_roll = torch.sum(MSELoss(pred_roll, roll) * mask)

        # print(f'[INFO] loss_x: {loss_x.device}')
        # print(f'[INFO] loss_y: {loss_y.device}')
        # print(f'[INFO] loss_w: {loss_w.device}')
        # print(f'[INFO] loss_h: {loss_h.device}')
        # print(f'[INFO] conf: {conf[0].device}')
        # print(f'[INFO] mask: {mask[0].device}')
        # print(f'[INFO] loss_conf: {loss_conf.device}')
        # print(f'[INFO] box_loss_scale_x: {box_loss_scale_x.device}')

        # print(f'[INFO] box_loss_scale_y: {box_loss_scale_y.device}')
        # print(f'[INFO] box_loss_scale: {box_loss_scale.device}')
        # print(f'[INFO] loss_yaw: {loss_yaw.device}')
        # print(f'[INFO] loss_pitch: {loss_pitch.device}')
        # print(f'[INFO] loss_roll: {loss_roll.device}')

        # loss = ((loss_x + loss_y) * self.lambda_xy + (loss_w + loss_h) * self.lambda_wh \
        #     + loss_conf * self.lambda_conf + loss_cls * self.lambda_cls) * self.lambda_bbox_attrs \
        #     + (self.lambda_yaw * loss_yaw + self.lambda_pitch * loss_pitch \
        #     + self.lambda_roll * loss_roll ) * self.lambda_pose

        if self.normalize:
            num_pos = torch.sum(mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        else:
            num_pos = bs/3
        losses = {
            "x"         : loss_x.item(),
            "y"         : loss_y.item(),
            "w"         : loss_w.item(),
            "h"         : loss_h.item(),
            "confidence": loss_conf.item(),
            "class"     : loss_cls.item(),
            "yaw"       : loss_yaw.item(),
            "pitch"     : loss_pitch.item(),
            "roll"      : loss_roll.item(),
            "bbox attr" : self.lambda_xy * (loss_x + loss_y) + self.lambda_wh * (loss_w + loss_h) \
                                            + self.lambda_conf * loss_conf + self.lambda_cls * loss_cls,
            "poses"     : self.lambda_yaw * loss_yaw + self.lambda_pitch * loss_pitch + self.lambda_roll * loss_roll
        }
        return losses, num_pos

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        # print(f'[INFO] target: {target[0].shape}')
        # batch size of target
        bs = len(target)

        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]

        mask = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)

        tx      = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        ty      = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tw      = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        th      = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tconf   = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tcls    = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, self.num_classes, requires_grad=False)
        yaw     = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        pitch   = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        roll    = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        for b in range(bs):
            if len(target[b]) == 0:
                continue
            # rescale
            gxs = target[b][:, 0:1] * in_w
            gys = target[b][:, 1:2] * in_h

            gws = target[b][:, 2:3] * in_w
            ghs = target[b][:, 3:4] * in_h

            gis = torch.floor(gxs)
            gjs = torch.floor(gys)

            gt_box = torch.FloatTensor(
                torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1)
            )

            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1)
            )

            anch_ious = jaccard(gt_box, anchor_shapes)

            # Find the best matching anchor box
            best_ns = torch.argmax(anch_ious, dim=-1)
            for i, best_n in enumerate(best_ns):
                if best_n not in anchor_index:
                    continue
                # Masks
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]
                # Masks
                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index

                    # have object (1) or None (0)
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1

                    tx[b, best_n, gj, gi] = gx - gi.float()
                    ty[b, best_n, gj, gi] = gy - gj.float()

                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n + subtract_index][0])
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n + subtract_index][1])
                    # Ratio used to obtain xywh
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]

                    # confidence grid
                    tconf[b, best_n, gj, gi] = 1

                    # class
                    tcls[b, best_n, gj, gi, int(target[b][i, 4])] = 1

                    # pose
                    yaw[b, best_n, gj, gi] = target[b][i, 5]
                    pitch[b, best_n, gj, gi] = target[b][i, 6]
                    roll[b, best_n, gj, gi] = target[b][i, 7]
                else:
                    print("Step {0} out of bound".format(b))
                    print("gj: {0}, height: {1} | gi: {2}, width: {3}".format(gj, in_h, gi, in_w))
                    continue

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, \
                yaw, pitch, roll, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, box_prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        # print(scaled_anchors)

        x = torch.sigmoid(box_prediction[..., 0])
        y = torch.sigmoid(box_prediction[..., 1])

        w = box_prediction[..., 2]  # Width
        h = box_prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1) \
            .repeat(int(bs * self.num_anchors / 3), 1, 1).view(x.shape).type(FloatTensor)
        
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t() \
            .repeat(int(bs * self.num_anchors / 3), 1, 1).view(y.shape).type(FloatTensor)
        

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(box_prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh], -1)).type(FloatTensor)

                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_ious_max > self.ignore_threshold] = 0
                # print(torch.max(anch_ious))
        return noobj_mask

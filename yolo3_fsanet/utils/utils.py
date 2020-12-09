from __future__ import division
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont
from math import cos, sin


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.num_poses = 3
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        # input shape: bs*(3*(5+num_class) + 3*num_poses)*13*13
        box_input = input[:, :(self.num_anchors*self.bbox_attrs), :, :]
        pose_input = input[:, (self.num_anchors*self.bbox_attrs):, :, :]

        batch_size = box_input.size(0)
        input_height = box_input.size(2)
        input_width = box_input.size(3)

        # stride
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # normalize anchors
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                          for anchor_width, anchor_height in self.anchors]

        # reshape prediction
        # bs, 3*6, 13, 13 -> bs, 3 ,13, 13, 6
        box_prediction = box_input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        # # bs, 3*3, 13, 13 -> bs, 3 ,13, 13, 3
        pose_prediction = pose_input.view(batch_size, self.num_anchors,
                                self.num_poses, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        # adust center width height
        x = torch.sigmoid(box_prediction[..., 0])
        y = torch.sigmoid(box_prediction[..., 1])
        w = box_prediction[..., 2]  # Width
        h = box_prediction[..., 3]  # Height

        # confidence
        conf = torch.sigmoid(box_prediction[..., 4])
        # class
        pred_cls = torch.sigmoid(box_prediction[..., 5:5+self.num_classes])  # Cls pred.

        # pose
        yaw = torch.torch.cuda.FloatTensor(pose_prediction[..., 0])
        pitch = torch.torch.cuda.FloatTensor(pose_prediction[..., 1])
        roll = torch.torch.cuda.FloatTensor(pose_prediction[..., 2])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # create grid, priority box, top left coordinate
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # width and height of anchor box
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(
            1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(
            1, 1, input_height * input_width).view(h.shape)

        # convert to center width height
        pred_boxes = FloatTensor(box_prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # rescale boxes
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes), yaw.view(batch_size, -1, 1), pitch.view(batch_size, -1, 1), roll.view(batch_size, -1, 1)), -1)
        return output.data


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(
        ((top+bottom)/2, (left+right)/2), axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top, right-left), axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # print(f"[INFO] prediction: {prediction[:, 7]}")
    # convert center width height to top left bottom right
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # conf and 3 angles yaw, pitch, roll
        #print(f'[INFO] prediction: {prediction.shape}')
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        #print(f'[INFO] image_pred[:, 5:5 + num_classes]: {image_pred[:, 5:5 + num_classes].shape}')
        #print(f"[INFO] class_conf: {class_conf.shape}")
        # select boxes with threshold "conf_thres"
        conf_mask = (image_pred[:, 4]*class_conf[:, 0] >= conf_thres).squeeze()
        #print(f"[INFO] conf_mask: {conf_mask.shape}")
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # convert shape to (x1, y1, x2, y2, obj_conf, class_conf)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float(), image_pred[:,6:]), 1)

        unique_labels = detections[:, 6].cpu().unique()
        # print(f"[INFO] unique_labels: {unique_labels.shape}")

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, 6] == c]
            # print(f"[INFO] detections: {detections.shape}")
            # print(f"[INFO] detections_class: {detections_class.shape}")
            keep = nms(
                detections[:, :4],
                detections[:, 4]*detections_class[:, 6],
                nms_thres
            )
            max_detections = detections[keep]
            # print(f"[INFO] max_detections: {max_detections}")
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                 * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                 * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img

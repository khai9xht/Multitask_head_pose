#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo3 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from utils.config import Config
from utils.utils import non_max_suppression, bbox_iou, DecodeBox, letterbox_image, yolo_correct_boxes, draw_axis

#--------------------------------------------#
#   Use trained_model to predict image
#--------------------------------------------#


class YOLO(object):
    _defaults = {
        "model_path": 'logs/Epoch96-Total_Loss317.6722-Val_Loss1151.2637.pth',
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "iou": 0.3,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   define YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.num_classes = 1
        self.config = Config
        self.generate()
    #---------------------------------------------------#
    #   generate all infor about model and data
    #---------------------------------------------------#

    def generate(self):
        self.config["yolo"]["classes"] = self.num_classes
        self.net = YoloBody(self.config)

        # load state of model
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(
                self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"],  (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # set inconsistent color for frame
        hsv_tuples = [(x / self.num_classes, 1., 1.)
                      for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   detect object in image
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(
            image, (self.model_image_size[0], self.model_image_size[1])))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)

        images = np.asarray(images)
        images = torch.from_numpy(images)
        if self.cuda:
            images = images.cuda()

        with torch.no_grad():
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)
            # print(f'[INFO] batch_detections: {batch_detections}')
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return []
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,6],np.int32)
        top_angle = batch_detections[top_index, 7:]
        top_bboxes = np.array(batch_detections[top_index, :4])

        # print(f'[INFO] top_index: {top_index}')
        # print(f'[INFO] top_conf: {top_conf}')
        # print(f'[INFO] top_label: {top_label}')
        # print(f'[INFO] top_angle: {top_angle}')
        # print(f'[INFO] top_bboxes: {top_bboxes}')

        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
            top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array(
            [self.model_image_size[0], self.model_image_size[1]]), image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=15)

        thickness = (np.shape(image)[0] + np.shape(image)
                     [1]) // self.model_image_size[0]

        predictions = []
        for i, score in enumerate(top_conf):
            infor = {}
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            yaw, pitch, roll = top_angle[i]

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(
                bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(
                right + 0.5).astype('int32'))
            infor["box"] = [left, top, right, bottom]
            infor["angle"] = [yaw, pitch, roll]
            predictions.append(infor)

            # draw box and angle in image
            # draw = ImageDraw.Draw(image)

            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline="red")
            # draw.text([left + i*10, top + i*10], str(score),
            #           fill=(255, 0, 0), font=font)
            # del draw
            # image_numpy = np.array(image)
            # print(f'[PREDICT] box: {[top, left, bottom, right]}')
            # print(f'[PREDICT] yaw = {yaw}, pitch = {pitch}, roll = {roll}')
            # img = draw_axis(image_numpy, yaw, pitch, roll, (left+right)//2, (top + bottom)//2)
            # image = Image.fromarray(img)
            # image.save('test.jpg')
            # print('save successfully !!!')

        return predictions

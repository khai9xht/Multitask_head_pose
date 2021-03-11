from random import shuffle
import numpy as np
import math
from PIL import Image
from torch.utils.data.dataset import Dataset
import albumentations as A
import cv2
import os

class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size, is_train):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.is_train = is_train

        self.train_transforms = A.Compose([
            A.RandomCrop(width=800, height=800, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=7, p=0.5),
            A.RandomContrast(limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ], bbox_params=A.BboxParams(format='coco', min_visibility=0.4))

        self.test_transform = A.Compose([
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ], bbox_params=A.BboxParams(format='coco', min_visibility=0.4))

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=0.1, hue=0.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.strip().split("\t", 1)
        # print(f'[INFO] line: {line}')
        image_path = line[0].replace("/media/2tb/Hoang/multitask", "/content")
        image_name = os.path.basename(image_path)
        # print(f'[INFO] line[0]: {line[0]}')
        image = Image.open(image_path)
        iw, ih = image.size
        h, w = input_shape
        line1 = line[1].strip().split("\t")
        # print(f'[INFO] line1: [{line1}]')
        box = np.array([np.array(list(map(float, box.split(" ")))) for box in line1], dtype=np.float32)
        box[:, [2,3]] = box[:, [2,3]] - box[:, [0,1]]
        image = np.array(image, dtype=np.uint8)

        if not random:
            transformed = self.test_transform(image=image, bboxes=box)
            image, box = transformed['image'], transformed['bboxes']
            image = Image.fromarray((image * 255).astype(np.uint8))
            if len(box) > 0:
                box = np.array(box, dtype=np.float32)
                box[:, [2,3]] = box[:, [2,3]] + box[:, [0,1]]
            box = np.array(box, dtype=np.float32)

            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # preprocess coordination target
            box_data = np.zeros((len(box), 7))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # keep valid bounding box
                box_data = np.zeros((len(box), 7))
                box_data[:len(box)] = box
            classes = np.zeros((len(box_data), 1))
            box_data = np.concatenate((box_data[:, :4], classes, box_data[:, 4:]), axis=1)
            return image_data, box_data

        
        transformed = self.train_transforms(image=image, bboxes=box)
        image, box = transformed['image'], transformed['bboxes']
        # imga = image.copy()
        # if len(box) != 0:
        #     for b in box:
        #         x, y, wd, ht = b[:4]
        #         cv2.rectangle(imga, (int(x), int(y), int(wd), int(ht)), (0, 255, 0), 3)
        #     cv2.imwrite("/content/drive/MyDrive/yolo_linear/Multitask_head_pose/yolo_linear/"+image_name, imga)
        image = Image.fromarray((image * 255).astype(np.uint8))
        if len(box) > 0:
            box = np.array(box, dtype=np.float32)
            box[:, [2,3]] = box[:, [2,3]] + box[:, [0,1]]
        box = np.array(box, dtype=np.float32)
        # Resize image
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.9, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # Place picture
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB",(w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        image_data = np.array(image, dtype=np.float32)

        # preprocess coordination target
        box_data = np.zeros((len(box), 7))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # keep valid bounding box
            box_data = np.zeros((len(box), 7))
            box_data[:len(box)] = box
        classes = np.zeros((len(box_data), 1))
        box_data = np.concatenate((box_data[:, :4], classes, box_data[:, 4:]), axis=1)
        return image_data, box_data

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        if self.is_train:
            img, y = self.get_random_data(lines[index], self.image_size[0:2], random=True)
        else:
            img, y = self.get_random_data(lines[index], self.image_size[0:2], random=False)

        boxes = np.array(y[:, :4], dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] / self.image_size[1]
        boxes[:, 1] = boxes[:, 1] / self.image_size[0]
        boxes[:, 2] = boxes[:, 2] / self.image_size[1]
        boxes[:, 3] = boxes[:, 3] / self.image_size[0]

        boxes = np.maximum(np.minimum(boxes, 1), 0)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        y = np.concatenate([boxes, y[:, 4:]], axis=-1)
        # convert to [-1 , 1]
        y[:, 5] = y[:, 5] / 180.0
        y[:, 6:] = y[:, 6:] / 90.0

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)
        # print(f"tmp_targets: {(tmp_targets[:, :4]<0).any()}")
        # print(f'[INFO] tmp_inp: {tmp_inp.shape}')
        # print(f'[INFO] tmp_targets: {tmp_targets.shape}')
        return tmp_inp, tmp_targets


# DataLoader
def yolo_dataset_collate(batch):
    images = []
    bboxes_ = []
    for img, box in batch:
        images.append(img)
        bboxes_.append(box)
    images = np.array(images)
    return images, bboxes_

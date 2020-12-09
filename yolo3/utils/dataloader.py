from random import shuffle
import numpy as np
import math
from PIL import Image
from torch.utils.data.dataset import Dataset
import cv2


class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        """Random preprocessing for real-time data enhancement"""
        line = annotation_line.strip().split('\t', 1)
        # print(f'[INFO] line: {line}')
        image_path = line[0]
        image = Image.open(image_path)
        iw, ih = image.size
        h, w = input_shape
        line = line[1].strip().split('\t')
        # print(f'[INFO] line1: [{line1}]')
        box = np.array([np.array(list(map(float, box.split(' ')))) for box in line])
        # print(f"[INFO] box: {box}")
        # resize image larger than input shape
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # resize with image smaller than input shape
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip transforms
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # color transforms
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        image = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
        image[..., 0] += hue*360
        image[..., 0][image[..., 0] > 1] -= 1
        image[..., 0][image[..., 0] < 0] += 1
        image[..., 1] *= sat
        image[..., 2] *= val
        image[image[:, :, 0] > 360, 0] = 360
        image[:, :, 1:][image[:, :, 1:] > 1] = 1
        image[image < 0] = 0
        image_data = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)*255

        # preprocess coordination target
        box_data = np.zeros((len(box), 7))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # keep valid box
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).all():
            return image_data, box_data
        else:
            return image_data, []

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        index = index % self.train_batches
        img, y = self.get_random_data(self.train_lines[index], self.image_size[0:2])
        if len(y) != 0:
            # normalize coordinate to [0, 1]
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            # print(f'[INFO] boxes: {boxes.shape}')
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2

            #convert 'C to [-1 , 1]
            y[:, 4:] = y[:, 4:] / 90
            # print(f'[INFO] y[:, 4:] : {y[:, 4:]}')
            y = np.concatenate([boxes, y[:, 4:]], axis=-1)
            # print(f'[INFO] y: {y.shape}')

        img = np.array(img, dtype=np.float32)

        img = np.transpose(img / 255.0, (2, 0, 1))
        y = np.array(y, dtype=np.float32)
        # print(f'[INFO] tmp_inp: {tmp_inp.shape}')
        # print(f'[INFO] tmp_targets: {tmp_targets.shape}')
        return img, y


# DataLoader
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


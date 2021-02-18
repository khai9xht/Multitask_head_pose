from random import shuffle
import numpy as np
from torch.utils.data.dataset import Dataset
import cv2


class CMUdataset(Dataset):
    def __init__(self, train_lines, transform=None, target_transform=None):
        super(CMUdataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.class_names = ("BACKGROUND", "face")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.train_batches

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        index = index % self.train_batches

        image_id, annotation = self.get_annotation(index)
        while len(annotation) <= 0 or image_id==None:
            index = (index+234) % self.train_batches
            image_id, annotation = self.get_annotation(index)
        image = self.get_image(image_id)
        boxes, labels, poses = annotation

        if self.transform:
            image, boxes, labels, poses = self.transform(image, boxes, labels, poses)
        # print(f"[INFO]cmu_dataset1: img: {img.shape}\t boxes: {boxes}\t labels: {labels}\t poses: {poses}")
        if self.target_transform:
            boxes, labels, poses = self.target_transform(boxes, labels, poses)
        # print(f"[INFO]3 images: {img.shape}")
        # print(f"[INFO]cmu_dataset2: img: {img.shape}\t boxes: {boxes}\t labels: {labels}\t poses: {poses}")
        return image, boxes, labels, poses

    def get_annotation(self, index):
        line = self.train_lines[index]
        # print(f"[INFO] line: {line}")
        line = line.strip().split('\t')
        image_id = line[0]
        # print(f"[INFO] image_id: {image_id}")
        annotation = line[1:]
        if len(annotation) != 0:
            return image_id, self._get_annotation(annotation)
        else:
            return None, []

    def _get_annotation(self, annotation):
        boxes_atr = np.array([np.array(list(map(float, box.split(" ")))) for box in annotation])
        boxes = boxes_atr[:, :4]
        poses = boxes_atr[:, 4:]
        labels = np.ones((boxes_atr.shape[0]), dtype=np.long)
        return (np.array(boxes, dtype=np.float32), labels,
        np.array(poses, dtype=np.float32))

    def get_image(self, image_id):
        # print(f"[INFO] image_id: {image_id}")
        image = cv2.imread(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


# DataLoader
def cmu_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    poses = []
    for img, box, label, pose in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
        poses.append(pose)
    images = np.array(images)
    return images, bboxes, labels, poses

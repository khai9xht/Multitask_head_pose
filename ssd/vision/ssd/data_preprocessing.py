from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # # ConvertFromInts(),
            # # PhotometricDistort(),
            # # Expand(self.mean),
            # # RandomSampleCrop(),
            # # # RandomMirror(),
            # # ToPercentCoords(),
            # Resize(self.size),
            # # SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)

class Custom_TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Custom_Compose([
            Custom_ConvertFromInts(),
            Custom_PhotometricDistort(),
            Custom_Expand(self.mean),
            # RandomSampleCrop(),
            # # RandomMirror(),
            Custom_ToPercentCoords(),
            Custom_Resize(self.size),
            Custom_SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None, poses=None: (img / std, boxes, labels, poses),
            Custom_ToTensor(),
        ])

    def __call__(self, img, boxes, labels, poses):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels, poses)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            # ToPercentCoords(),
            # Resize(size),
            # SubtractMeans(mean),
            lambda img, boxes=None, labels=None, poses=None: (img / std, boxes, labels, poses),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

class Custom_TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Custom_Compose([
            Custom_ToPercentCoords(),
            # Resize(size),
            # SubtractMeans(mean),
            lambda img, boxes=None, labels=None, poses=None: (img / std, boxes, labels, poses),
            Custom_ToTensor(),
        ])

    def __call__(self, image, boxes, labels, poses):
        return self.transform(image, boxes, labels, poses)

class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            # SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image

class Custom_PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Custom_Compose([
            Custom_Resize(size),
            # SubtractMeans(mean),
            lambda img, boxes=None, labels=None, poses=None: (img / std, boxes, labels, poses),
            Custom_ToTensor()
        ])

    def __call__(self, image):
        image, _, _, _ = self.transform(image)
        return image

from yolo import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm

def readfile(path):
    with open(path, 'r') as f:
        test_data = f.readlines()

    annotation = []
    for data in test_data:
        anno = {}
        line = data.split(' ', 1)
        anno["image path"] = line[0].replace('hpdb/', '../../data/BIWI')
        anno["infor"] = []
        infors = line[1].replace('\n', '')
        infors = infors.split('\t')
        for infor in infors:
            dict_box = {}
            box_infor = [float(x) for x in infor.split(' ')]
            dict_box["box"] = box_infor[:4] #top left rigth bottom
            dict_box["angle"] = box_infor[4:] #yaw pitch roll
            anno["infor"].append(dict_box)
        annotation.append(anno)
    return annotation

def calculate_iou(box_a, box_b):
    inter_l = max(box_a[0], box_b[0])
    inter_t = max(box_a[1], box_b[1])
    inter_r = min(box_a[2], box_b[2])
    inter_b = min(box_a[3], box_b[3])

    interArea = max(0, inter_r - inter_l + 1) * max(0, inter_b - inter_t + 1)

    box_a_Area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_Area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = interArea / float(box_a_Area + box_b_Area - interArea)
    return iou

def evaluate(yolo, ground_truth):
    """
    params:
        yolo: (class YOLO in file yolo.py)
        ground_truth:
            structrue list [dict{
                "image path": String
                "infor": list[dict{
                    "box": list
                    "angle": list
                    }
                ]
            }]
    return:
        box: iou
        angle: MAE
    """
    ious = []
    yaw_MAEs = []
    pitch_MAEs = []
    roll_MAEs = []
    for infor in tqdm(ground_truth):
        image_path = infor["image path"]
        # print(f"[INFO] image path: {image_path}")
        image = Image.open(image_path)

        y_true = infor["infor"]
        y_pred = yolo.detect_image(image)
        # print(f"[INFO] y_true: {y_true}")
        # print(f"[INFO] y_pred: {y_pred}")
        if y_pred == None:
            continue
        if len(y_true) != len(y_pred):
            continue

        for yt, yp in zip(y_true, y_pred):
            iou = calculate_iou(yt["box"], yp["box"])
            yaw_mae = abs(yt["angle"][0] - yp["angle"][0])
            pitch_mae = abs(yt["angle"][1] - yp["angle"][1])
            roll_mae = abs(yt["angle"][2] - yp["angle"][2])

            ious.append(iou)
            yaw_MAEs.append(yaw_mae)
            pitch_MAEs.append(pitch_mae)
            roll_MAEs.append(roll_mae)


    mean_iou = np.mean(np.array(ious))
    mean_yaw_mae = np.mean(np.array(yaw_MAEs))
    mean_pitch_mae = np.mean(np.array(pitch_MAEs))
    mean_roll_mae = np.mean(np.array(roll_MAEs))

    return mean_iou, mean_yaw_mae, mean_pitch_mae, mean_roll_mae

if __name__ == "__main__":
    test_path = 'BIWI_test.txt'
    ground_truth = readfile(test_path)
    print(f"[INFO] ground truth: {len(ground_truth)}")

    yolo = YOLO()
    mean_iou, mean_yaw_mae, mean_pitch_mae, mean_roll_mae = evaluate(yolo, ground_truth)

    print("Loss:\n\t Box iou: {}\n\t Mean yaw MAE: {}\n\t Mean pitch MAE: {}\n\t Mean roll MAE: {}".format(mean_iou, mean_yaw_mae, mean_pitch_mae, mean_roll_mae))

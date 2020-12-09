#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
yolo = YOLO()

test_path = 'BIWI_test.txt'
with open(test_path, 'r') as f:
    test_data = f.readlines()
print(f'[INFO] test_data: {len(test_data)}')


while True:
    index = input("select image:")
    index = int(index)
    # print(f"[INFO] data: {test_data[index]}")
    line = test_data[index].split(' ', 1)
    image_path = line[0].replace('hpdb/', '../../data/BIWI')
    line1 = line[1].replace('\n', '')
    # print(f'[INFO] line1: {line1}')
    box_infors = [float(x) for x in line1.split(' ')]
    # print(f"[INFO] box_infors: {box_infors}, shape: {len(box_infors)}")
    box = box_infors[:4]
    yaw, pitch, roll = box_infors[4:]
    print(f"[GROUND TRUTH] box: {box}")
    print(f"[GROUND TRUTH] yaw: {yaw}, pitch: {pitch}, roll: {roll}")
    try:
        image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
        continue
    else:
        predictions = yolo.detect_image(image)
        print(f"[PREDICT] prediction: {predictions}")
        print("run completely !!!")


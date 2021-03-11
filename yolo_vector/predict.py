from yolo import YOLO
from PIL import Image
import numpy as np
yolo = YOLO()


while True:
    image_path = input("select image:")
    try:
        image = Image.open(image_path)
        print(image.size)
    except:
        print('Open Error! Try again!')
        continue
    else:
        predictions = yolo.detect_image(image)
        print(f"[PREDICT] prediction: {predictions}")
        print("run completely !!!")


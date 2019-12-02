import cv2
import pytesseract
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np

data = {}
image = Image.open(r"D:\MyProject\Machine Learning\data\OCRTest.jpg")
content = pytesseract.image_to_data(image, lang="chi_sim", output_type="dict")  # 解析图片
for i in range(len(content["text"])):
    if 0 < len(content["text"][i]):
        if content["text"][i] == "姓名":
            (x, y, w, h) = (
                content["left"][i],
                content["top"][i],
                content["width"][i],
                content["height"][i],
            )
            data[content["text"][i]] = [
                content["left"][i],
                content["top"][i],
                content["width"][i],
                content["height"][i],
            ]
            print(x, y, w, h)
            im = cv2.imread(r"D:\MyProject\Machine Learning\data\OCRTest.jpg")
            cv2.rectangle(im, (x-20, y-20), (x + w + 500, y + h + 50), (0, 255, 0), 2)
            cv2.imwrite("./test.jpg", im)

from kfbreader_linux import kfbReader as kr
import cv2
import numpy as np
import json
import os


def readname(filePath):
    filenames = os.listdir(filePath)
    for name in filenames:
        filenames[filenames.index(name)] = name[:-4]
    return filenames


def get_roi(label):
    with open(label, "r") as f:
        js = json.load(f)
    rois = []
    roi = {}
    for dic in js:
        if dic["class"] == "roi":
            roi = dic
            roi["poses"] = []
            rois.append(roi)
        else:
            pass
    for dic in js:
        if dic["class"] == "roi":
            pass
        else:
            for roi1 in rois:
                if (
                    roi1["x"] <= dic["x"]
                    and roi1["y"] <= dic["y"]
                    and dic["x"] + dic["w"] <= roi1["x"] + roi1["w"]
                    and dic["y"] + dic["h"] <= roi1["y"] + roi1["h"]
                ):
                    roi1["poses"].append(dic)
    return rois


filePath = "pos_1/"
scale = 20
names = readname(filePath)

for name in names:
    file1 = filePath + name + ".kfb"
    label1 = "labels/" + name + ".json"
    reader = kr.reader()
    kr.reader.ReadInfo(reader, file1, scale, True)

    rois = get_roi(label1)
    for i, roi1 in enumerate(rois):
        roi = reader.ReadRoi(roi1["x"], roi1["y"], roi1["w"], roi1["h"], scale)
        for pos in roi1["poses"]:
            rx = pos["x"] - roi1["x"]
            ry = pos["y"] - roi1["y"]
            cv2.rectangle(roi, (rx, ry), (rx + pos["w"], ry + pos["h"]), (0, 255, 0), 4)
        save_name = "PosRoi/" + name + "roi" + str(i) + ".jpg"
        cv2.imwrite(save_name, roi)
        print("save roi img:" + save_name)

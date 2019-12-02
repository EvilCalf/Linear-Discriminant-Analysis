import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import sys
import json
import cv2
from multiprocessing import Process, Manager
import multiprocessing
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import kfbReader
from build_network import build_network
from data_loader import CervicalDataset, WsiDataset, collater
from augmentation import Normalizer
from util import BBoxTransform, ClipBoxes
import cfg
from lib.nms.gpu_nms import gpu_nms


def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations. Accept dets as tensor" ""
    dets = dets.cpu().numpy()
    return gpu_nms(dets, thresh)


def calc_split_num(image_shape):
    strides = cfg.patch_size
    # strides = [cfg.patch_size[0]//2, cfg.patch_size[1]//2]
    x_num = (image_shape[0] - cfg.patch_size[0]) // strides[0] + 2
    y_num = (image_shape[1] - cfg.patch_size[1]) // strides[1] + 2
    return strides, x_num, y_num


def predict(sample_paths, args):
    model, start_epoch = build_network(snapshot=args.snapshot, backend="retinanet")
    model.eval()
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    print(
        "Begin to predict mask: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    )
    for sample_path in sample_paths:
        filename = sample_path.split("/")[-1].split(".")[0]

        read = kfbReader.reader()
        read.ReadInfo(sample_path, 20, False)
        width = read.getWidth()
        height = read.getHeight()
        image_shape = (width, height)

        strides, x_num, y_num = calc_split_num((width, height))

        model.eval()
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        transformed_all = []
        classification_all = []
        for i in range(x_num // 2):
            for j in range(y_num // 2):
                x = (
                    strides[0] * i
                    if i < x_num - 1
                    else image_shape[0] - cfg.patch_size[0]
                )
                y = (
                    strides[1] * j
                    if j < y_num - 1
                    else image_shape[1] - cfg.patch_size[1]
                )

                img = read.ReadRoi(
                    x, y, cfg.patch_size[0], cfg.patch_size[1], scale=20
                ).copy()
                img = img.transpose((2, 0, 1))
                img = img[np.newaxis, :, :, :]
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).float()
                with torch.no_grad():
                    classification, regression, anchors = model(img.cuda())
                transformed_anchors = regressBoxes(anchors, regression)
                transformed_anchors = clipBoxes(transformed_anchors)

                scores = classification
                scores_over_thresh = (scores > 0.05)[0, :, 0]

                if scores_over_thresh.sum() == 0:
                    continue

                classification = classification[0, scores_over_thresh, :]
                transformed_anchors = transformed_anchors[0, scores_over_thresh, :]
                transformed_anchors[:, 0] = transformed_anchors[:, 0] + x
                transformed_anchors[:, 1] = transformed_anchors[:, 1] + y
                transformed_anchors[:, 2] = transformed_anchors[:, 2] + x
                transformed_anchors[:, 3] = transformed_anchors[:, 3] + y
                scores = scores[0, scores_over_thresh, :]
                transformed_all.append(torch.cat([transformed_anchors, scores], dim=1))
                classification_all.append(classification)

        transformed_all = torch.cat(transformed_all, dim=0)
        classification_all = torch.cat(classification_all, dim=0)
        anchors_num_idx = nms(transformed_all, 0.5)
        nms_scores = classification_all[anchors_num_idx, :]
        nms_transformed = transformed_all[anchors_num_idx, :]

        scores = nms_scores.detach().cpu().numpy()
        transformed = nms_transformed.detach().cpu().numpy()
        pos_all = []
        for i in range(scores.shape[0]):
            x = int(transformed[i, 0])
            y = int(transformed[i, 1])
            w = max(int(transformed[i, 2] - transformed[i, 0]), 1)
            h = max(int(transformed[i, 3] - transformed[i, 1]), 1)
            p = float(scores[i, 0])
            pos = {"x": x, "y": y, "w": w, "h": h, "p": p}
            pos_all.append(pos)

        with open(os.path.join(cfg.result_path, filename + ".json"), "w") as f:
            json.dump(pos_all, f)

        print(
            "Finish predict mask: ",
            filename,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict viable tumor mask")
    parser.add_argument("--snapshot", default=None, type=str, help="snapshot")
    args = parser.parse_args()

    sample_paths = glob(os.path.join(cfg.test_path, "test_*", "*.kfb"))
    predict(sample_paths, args)


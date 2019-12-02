import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from build_network import build_network
from data_loader import CervicalDataset, collater
from augmentation import Normalizer
from util import BBoxTransform, ClipBoxes
import cfg
from lib.nms.gpu_nms import gpu_nms


def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor" ""
    # return pth_nms(dets, thresh)
    dets = dets.cpu().numpy()
    return gpu_nms(dets, thresh)


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
    )


def transform_anchors(classification, regression, anchors, regressBoxes, clipBoxes):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors)

    scores = classification
    scores_over_thresh = (scores > 0.01)[0, :, 0]

    if scores_over_thresh.sum() == 0:
        # no boxes to NMS, just return
        return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

    classification = classification[:, scores_over_thresh, :]
    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
    scores = scores[:, scores_over_thresh, :]

    anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

    nms_scores = classification[0, anchors_nms_idx, :]

    return [nms_scores, transformed_anchors[0, anchors_nms_idx, :]]


def predict(sample_path, args):
    model, start_epoch = build_network(snapshot=args.snapshot, backend="retinanet")
    model.eval()
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)

    test_data = CervicalDataset(
        sample_path, cfg.patch_size, transform=transforms.Compose([Normalizer()])
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        collate_fn=collater,
        num_workers=0,
    )

    model.eval()
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            st = time.time()
            annotations = data["label"].cuda()
            classification, regression, anchors = model(data["img"].cuda())

            scores, transformed_anchors = transform_anchors(
                classification, regression, anchors, regressBoxes, clipBoxes
            )
            print("Elapsed time: {}".format(time.time() - st))
            scores = scores.detach().cpu().numpy()
            transformed_anchors = transformed_anchors.detach().cpu().numpy()

            idxs = np.where(scores > 0.5)
            img = np.array(255 * data["img"][0, :, :, :]).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            img_anno = img.copy()

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                draw_caption(img, (x1, y1, x2, y2), str(scores[idxs[0][j]]))

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            for j in range(annotations.shape[1]):
                bbox = annotations[0, j, :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                draw_caption(img_anno, (x1, y1, x2, y2), "pos")

                cv2.rectangle(
                    img_anno, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2
                )

            merge_img = np.hstack([img, img_anno])
            cv2.imwrite(
                os.path.join(cfg.result_path, "result" + str(idx) + ".jpg"), merge_img
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict viable tumor mask")
    parser.add_argument("--snapshot", default=None, type=str, help="snapshot")
    args = parser.parse_args()

    path = cfg.visual_sample_path
    sample_path = [os.path.join(path, x) for x in os.listdir(path) if ".npz" in x]
    predict(sample_path, args)


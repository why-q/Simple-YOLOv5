import math
import os
import random

# from torch.utils.data import DataLoader
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "..")
from utils.general import xywhn2xyxy, xyxy2xywh

# from general import xyxy2xywh, xywhn2xyxy


class CustomDataset(Dataset):
    def __init__(
        self,
        img_path: list,
        label_path: list,
        img_size=640,
        augment=False,
        hyp=None,
        radam_perspect=False,
    ):
        """
        :param img_path: all image path in a list
        :param label_path: all image label in a list
        :param img_size: net input size
        :param augment: image augment
        :param hyp: hyp params for image augment
        :param radam_perspect: use random_perspective func or not
        """
        self.img_path = img_path
        self.label_path = label_path
        assert (
            len(self.img_path) == len(self.label_path) or len(self.img_path) > 0
        ), "image count:{} != label cnt:{}".format(
            len(self.img_path), len(self.label_path)
        )
        self.len = len(self.img_path)
        self.img_formats = [
            "bmp",
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
        ]  # acceptable image suffixes
        for img in self.img_path:
            img_fmt = img.split(".")[-1]
            assert (
                img_fmt in self.img_formats
            ), "{} img format is not acceptable".format(img)  # check image format
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.radam_perspect = radam_perspect

    def load_labels(self, index):
        # loads labels of 1 image from dataset, returns labels(np) [[] []...]
        path = self.label_path[index]

        labels = []
        with open(path) as f:
            labels_str = f.readlines()
        for lstr in labels_str:
            l = lstr.strip().split(" ")  # noqa: E741
            labels.append([float(i) for i in l])
        return np.float32(labels)

    def load_image(self, index):
        # loads 1 image from dataset, returns img(np), original hw, resized hw
        path = self.img_path[index]
        # print("path: ", path)

        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def __getitem__(self, index):
        # return No.index data(tensor)
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        # Letterbox
        shape = (
            self.img_size,
            self.img_size,
        )  # final letterboxed shape, use net input img shape
        img, ratio, pad = letterbox(
            img, shape, auto=False, scaleup=self.augment
        )  # img, ratio, (dw, dh)

        # Load labels
        labels = self.load_labels(index).copy()
        if labels.shape:  # normalized xywh to pixel xyxy format
            # print("labels.shape: ", labels.shape)
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
            )

        hyp = self.hyp
        if self.augment:
            assert hyp is not None, "hyp is None"  # check hyp
            # Augment imagespace
            if (
                self.radam_perspect
            ):  # default: not use random_perspective fucn, radam_perspect is False
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

            # Augment colorspace
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(
                labels[:, 1:5]
            )  # convert pixel xyxy to pixel xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # img_origin = np.copy(img)  # for Visualize
            # Augment flip up-down
            if random.random() < hyp["flipud"]:  # default: not use
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
            # Augment flip left-right
            if (
                random.random() < hyp["fliplr"]
            ):  # default: fliplr is 0.5, 50% probability to fliplr
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
            # # for Visualize
            # import matplotlib.pyplot as plt
            # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
            # ax[0].imshow(img_origin[:, :, ::-1])  # base
            # ax[1].imshow(img[:, :, ::-1])  # warped

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = (
            img[:, :, ::-1].transpose(2, 0, 1) / 255.0
        )  # BGR to RGB, to 3x416x416(3x640x640), norm
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img).float(), labels_out

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):  # noqa: E741
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_perspective(
    img,
    targets=(),
    segments=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    # scale is zoom gain in cv2.getRotationMatrix2D

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # img_origin = np.copy(img)  # for Visualize
    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img_origin[:, :, ::-1])  # base
    # ax[1].imshow(img[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
            n, 8
        )  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(
            box1=targets[:, 1:5].T * s,
            box2=new.T,
            area_thr=0.01 if use_segments else 0.10,
        )
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(
    box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16
):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + eps) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def get_data_path(data: dict):
    # train
    train_img_path = data["train"]
    train_label_path = train_img_path.replace("images", "labels")
    train_img_path = [
        os.path.join(train_img_path, f) for f in os.listdir(train_img_path)
    ]
    train_img_path.sort()
    train_label_path = [
        os.path.join(train_label_path, f) for f in os.listdir(train_label_path)
    ]
    train_label_path.sort()
    # val
    val_img_path = data["val"]
    val_label_path = val_img_path.replace("images", "labels")
    val_img_path = [os.path.join(val_img_path, f) for f in os.listdir(val_img_path)]
    val_img_path.sort()
    val_label_path = [
        os.path.join(val_label_path, f) for f in os.listdir(val_label_path)
    ]
    val_label_path.sort()
    return train_img_path, train_label_path, val_img_path, val_label_path


if __name__ == "__main__":
    # img
    train_sample_dataset_path = "../data/train_sample_dataset"
    train_img_path = os.path.join(train_sample_dataset_path, "images", "train")
    train_img_path = [
        os.path.join(train_img_path, f) for f in os.listdir(train_img_path)
    ]
    train_img_path.sort()
    # label
    train_label_path = os.path.join(train_sample_dataset_path, "labels", "train")
    train_label_path = [
        os.path.join(train_label_path, f) for f in os.listdir(train_label_path)
    ]
    train_label_path.sort()
    # Hyperparameters
    import yaml

    hyp_path = "../data/hyp.aerial_infra.yaml"
    with open(hyp_path) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    # CustomDataset
    dataset = CustomDataset(
        train_img_path, train_label_path, augment=True, hyp=hyp, radam_perspect=False
    )
    for i in range(len(dataset)):
        img, labels = dataset.__getitem__(i)
        print("img shape:{}, labels shape:{}".format(img.shape, labels.shape))

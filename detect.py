import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from models.common import Conv
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


def detect(opt):
    source, weights, imgsz, _data = opt.source, opt.weights, opt.img_size, opt.data

    # Directories
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = device.type != "cpu"  # half precision only supported on CUDA
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    cname = data_dict["names"]

    # Load model
    ckpt = torch.load(weights, map_location=device)  # load
    # model = ckpt["model"].float().eval()  # official model yolov5s.pt
    model = ckpt.float().eval()  # self model
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if half:
        model.half()  # to FP16

    # Load image
    img = cv2.imread(source)
    assert img is not None, "Image Not Found " + source
    im0 = img.copy()
    h0, w0 = img.shape[:2]  # orig hw
    r = imgsz / max(h0, w0)  # resize image to img_size
    if r != 1:
        img = cv2.resize(
            img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR
        )
    img, ratio, pad = letterbox(img, (imgsz, imgsz), auto=False)  # img, ratio, (dw, dh)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                cn = cname[c]  # class name
                label = "{}, {:0.2f}".format(cn, conf.item())
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(
                    im0, c1, c2, (0, 255, 255), thickness=1, lineType=cv2.LINE_AA
                )
                cv2.putText(
                    im0,
                    label,
                    (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            # Show and Save image
            if opt.save:
                cv2.imwrite("result/result.jpg", im0)
            cv2.imshow("result", im0)
            cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/yolo_voc.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov5s_pretrain.pth", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="data/test_images/motorbike.jpg", help="source"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--save", action="store_true", help="save inference image")
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt)

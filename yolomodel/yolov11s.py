import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Basic Conv Block
# ------------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ------------------------------
# Bottleneck Block
# ------------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2 // 2, 3, 1)
        self.cv2 = Conv(c2 // 2, c2, 3, 1)

    def forward(self, x):
        return self.cv2(self.cv1(x)) + x

# ------------------------------
# C3k Block (Bottlenecks + Concat)
# ------------------------------
class C3k(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(hidden * 2, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# ------------------------------
# C3k2 Block (C3k as sub-block)
# ------------------------------
class C3k2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv2 = Conv(c1 + c1 // 2, c2, 1, 1)
        self.m = nn.ModuleList([C3k(c1, c1 // 2)])

    def forward(self, x):
        x1 = self.cv1(x)
        for m in self.m:
            x1 = m(x1)
        return self.cv2(torch.cat([x1, x], dim=1))

# ------------------------------
# SPPF (Spatial Pyramid Pooling - Fast)
# ------------------------------
class SPPF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c1 // 2, 1, 1)
        self.cv2 = Conv(c1 * 2, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

# ------------------------------
# Detect Head (Multi-scale and Full)
# ------------------------------
class Detect(nn.Module):
    def __init__(self, num_classes=80, anchors=((10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)), ch=(128, 256, 512)):
        super().__init__()
        self.nc = num_classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(ch)  # number of layers
        self.na = len(anchors) // self.nl  # number of anchors per layer
        self.stride = [8, 16, 32]  # strides per layer
        self.grid = [torch.zeros(1)] * self.nl

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)

        self.m = nn.ModuleList([nn.Conv2d(c, self.no * self.na, 1) for c in ch])

    def forward(self, x):
        z = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = self.m[i](x[i])
            x[i] = x[i].view(bs, self.na, self.no, ny, nx)
            x[i] = x[i].permute(0, 1, 3, 4, 2).contiguous()
            z.append(x[i].view(bs, -1, self.no))
        return torch.cat(z, 1)

# ------------------------------
# YOLOv11s Architecture (multi-scale output)
# ------------------------------
class YOLOv11s(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.layer1 = nn.Sequential(  # 320x320
            Conv(3, 32, 3, 2),
            Conv(32, 64, 3, 2),
            C3k2(64, 128),
        )
        self.layer2 = nn.Sequential(  # 160x160
            Conv(128, 128, 3, 2),
            C3k2(128, 256),
        )
        self.layer3 = nn.Sequential(  # 80x80
            Conv(256, 256, 3, 2),
            C3k2(256, 256),
        )
        self.layer4 = nn.Sequential(  # 40x40
            Conv(256, 512, 3, 2),
            C3k2(512, 512),
            SPPF(512, 512)
        )

        self.detect = Detect(num_classes=num_classes, ch=(128, 256, 512))

    def forward(self, x):
        x1 = self.layer1(x)  # small
        x2 = self.layer2(x1)  # medium
        x3 = self.layer3(x2)  # large
        x4 = self.layer4(x3)  # SPPF output
        return self.detect([x1, x2, x4])

import torch
import torchvision

def postprocess(pred, conf_thres=0.25, iou_thres=0.45, num_classes=1, image_size=640):
    """
    pred: Tensor of shape [B, N, 6] → (x, y, w, h, obj_conf, class_conf)
    returns: List of [n, 6] → (x1, y1, x2, y2, conf, class)
    """
    pred = torch.sigmoid(pred)  # sigmoid: normalize to 0~1
    xywh = pred[..., 0:4]
    conf = pred[..., 4:5]
    cls = pred[..., 5:]

    # final confidence = objectness × class_conf
    score = conf * cls  # shape: [B, N, num_classes]

    # xywh to xyxy
    box = xywh.clone()
    box[..., 0] = xywh[..., 0] - xywh[..., 2] / 2  # x1
    box[..., 1] = xywh[..., 1] - xywh[..., 3] / 2  # y1
    box[..., 2] = xywh[..., 0] + xywh[..., 2] / 2  # x2
    box[..., 3] = xywh[..., 1] + xywh[..., 3] / 2  # y2

    # scale back to image size (optional, if training is normalized)
    box = box * image_size

    results = []
    for i in range(pred.shape[0]):  # for each image
        boxes_i = box[i]
        scores_i, class_ids = torch.max(score[i], dim=1)

        # confidence threshold
        mask = scores_i > conf_thres
        boxes_i = boxes_i[mask]
        scores_i = scores_i[mask]
        class_ids = class_ids[mask]

        # NMS
        keep = torchvision.ops.nms(boxes_i, scores_i, iou_thres)
        final_boxes = torch.cat([
            boxes_i[keep],
            scores_i[keep].unsqueeze(1),
            class_ids[keep].unsqueeze(1).float()
        ], dim=1)  # [n, 6]

        results.append(final_boxes)

    return results  # list of [n, 6] per image



if __name__ == "__main__":
    model = YOLOv11s(num_classes=1)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)  # [1, 97200, 6]
        results = postprocess(y, conf_thres=0.3, iou_thres=0.5, num_classes=1)

    for det in results:
        print(det)  # [x1, y1, x2, y2, score, class_id]

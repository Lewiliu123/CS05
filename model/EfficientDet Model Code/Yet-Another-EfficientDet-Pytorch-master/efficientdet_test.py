import os
import time
import argparse
import torch
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, standard_to_bgr, get_index_label, plot_one_box

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weights", type=str, required=True, help="Path to weight file")
parser.add_argument("-i", "--input", type=str, default="F:/COMP5703_Project/datasets/moana/val/", help="Input image folder")
parser.add_argument("-o", "--output", type=str, default="F:/COMP5703_Project/test_output/", help="Output folder")
parser.add_argument("-c", "--compound_coef", type=int, default=2, help="EfficientDet coefficient")
parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold")
parser.add_argument("--iou_threshold", type=float, default=0.2, help="IoU threshold")
parser.add_argument("--cuda", type=bool, default=True, help="Use GPU")
parser.add_argument("--float16", type=bool, default=False, help="Use float16")
args = parser.parse_args()

# Config
compound_coef = args.compound_coef
use_cuda = args.cuda
use_float16 = args.float16
threshold = args.threshold
iou_threshold = args.iou_threshold

weights_path = args.weights
val_image_dir = args.input
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

anchor_ratios = [(1.0, 1.0), (1.0, 0.5), (0.5, 1.0)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
obj_list = ['object']
color_list = [(255, 0, 0)]

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef]

# Load model
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()
if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Load images
image_paths = sorted(glob(os.path.join(val_image_dir, '*.png')))

# Inference loop
for img_path in tqdm(image_paths, desc="Inference"):
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(f).cuda() for f in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(f) for f in framed_imgs], 0)
    x = x.to(torch.float16 if use_float16 else torch.float32).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        preds = postprocess(x, anchors, regression, classification,
                            regressBoxes, clipBoxes, threshold, iou_threshold)
        preds = invert_affine(framed_metas, preds)

    # Draw boxes and save
    for i in range(len(preds)):
        img = ori_imgs[i].copy()
        rois = preds[i].get('rois', [])
        class_ids = preds[i].get('class_ids', [])
        scores = preds[i].get('scores', [])

        det_num = min(len(rois), len(class_ids), len(scores))

        for j in range(det_num):
            x1, y1, x2, y2 = rois[j].astype(np.int32)
            obj = obj_list[class_ids[j]]
            score = float(scores[j])
            color = color_list[0]
            plot_one_box(img, [x1, y1, x2, y2], label=obj, score=score, color=color)

        save_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)

print(f"\nDone. {len(image_paths)} images saved to: {output_dir}")
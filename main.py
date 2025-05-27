import os
import json

# 配置路径
json_dir = r"C:\Users\PC1\Desktop\2D\X-band"
image_dir = r"C:\Users\PC1\Desktop\2D\X-band"
output_img_dir = os.path.join(json_dir, "images", "train")
output_lbl_dir = os.path.join(json_dir, "labels", "train")

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

img_w, img_h = 2048, 2048
converted_files = []

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(json_dir, filename)
        image_path = os.path.join(image_dir, base_name + ".png")
        label_path = os.path.join(output_lbl_dir, base_name + ".txt")

        if not os.path.exists(image_path):
            continue

        dst_image_path = os.path.join(output_img_dir, base_name + ".png")
        if not os.path.exists(dst_image_path):
            os.system(f'copy "{image_path}" "{dst_image_path}"')

        with open(json_path, "r") as f:
            data = json.load(f)

        with open(label_path, "w") as out_f:
            for ann in data.get("annotations", []):
                x = ann["xmin"]
                y = ann["ymin"]
                w = ann["width"]
                h = ann["height"]
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                ww = w / img_w
                hh = h / img_h
                out_f.write(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        converted_files.append(base_name)

import os
import shutil

base_dir = r"C:\Users\PC1\Desktop\2D\X-band"
train_img = os.path.join(base_dir, "images", "train")
train_lbl = os.path.join(base_dir, "labels", "train")
val_img = os.path.join(base_dir, "images", "val")
val_lbl = os.path.join(base_dir, "labels", "val")

os.makedirs(train_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

image_files = sorted([f for f in os.listdir(train_img) if f.endswith(".png")])

split_idx = int(len(image_files) * 0.8)
val_samples = image_files[split_idx:]

for img_file in val_samples:
    base = os.path.splitext(img_file)[0]
    label_file = base + ".txt"

    shutil.move(os.path.join(train_img, img_file), os.path.join(val_img, img_file))

    if os.path.exists(os.path.join(train_lbl, label_file)):
        shutil.move(os.path.join(train_lbl, label_file), os.path.join(val_lbl, label_file))

print(f"timesplit {len(val_samples)} to val")



from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

sensor_dir = r"C:\Users\PC1\Downloads\X_band_radar\X_band_radar"
output_dir = r"C:\Users\PC1\Desktop\2D\X-band\images\test_2"
os.makedirs(output_dir, exist_ok=True)

all_png_files = [f for f in os.listdir(sensor_dir) if f.endswith(".png")]
selected_files = all_png_files

converted = []

for filename in tqdm(selected_files):
    img_path = os.path.join(sensor_dir, filename)
    img = Image.open(img_path).convert("RGB")
    np_img = np.array(img)

    hsv_img = np.array(img.convert("HSV"))
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    mask = ((h >= 20) & (h <= 40)) & (s > 60) & (v > 100)

    output_img = np.zeros_like(np_img)
    output_img[mask] = np_img[mask]

    output_path = os.path.join(output_dir, filename)
    Image.fromarray(output_img).save(output_path)
    converted.append(filename)

print(f" {len(converted)} image")

import os
import shutil
from sklearn.model_selection import TimeSeriesSplit

base_dir = r"C:\Users\PC1\Desktop\2D\X-band"
image_dirs = [os.path.join(base_dir, "images", "train"), os.path.join(base_dir, "images", "val")]
label_dirs = [os.path.join(base_dir, "labels", "train"), os.path.join(base_dir, "labels", "val")]
kfold_base_dir = os.path.join(base_dir, "kfold_time_split")

image_files = []
for image_dir in image_dirs:
    if os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            if f.endswith(".png") and "_" in f:
                try:
                    ts = int(f.split("_")[1].split(".")[0])
                    image_files.append((ts, f, image_dir))
                except:
                    continue

image_files.sort(key=lambda x: x[0])
file_tuples = [(f, d) for (_, f, d) in image_files]

tscv = TimeSeriesSplit(n_splits=5)
folds = list(tscv.split(file_tuples))

if os.path.exists(kfold_base_dir):
    shutil.rmtree(kfold_base_dir)
os.makedirs(kfold_base_dir, exist_ok=True)

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    for subset, idxs in [('train', train_idx), ('val', val_idx)]:
        img_out = os.path.join(kfold_base_dir, f"fold_{fold_idx}", "images", subset)
        lbl_out = os.path.join(kfold_base_dir, f"fold_{fold_idx}", "labels", subset)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for i in idxs:
            filename, img_src_dir = file_tuples[i]
            base = os.path.splitext(filename)[0]

            shutil.copy(os.path.join(img_src_dir, filename), os.path.join(img_out, filename))

            for label_dir in label_dirs:
                label_path = os.path.join(label_dir, base + ".txt")
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(lbl_out, base + ".txt"))
                    break

print("k-fold split done")


import os

base_dir = r"C:\Users\PC1\Desktop\2D\X-band"
kfold_dir = os.path.join(base_dir, "kfold_time_split")
yaml_output_dir = os.path.join(base_dir, "kfold_split_yaml")
os.makedirs(yaml_output_dir, exist_ok=True)

nc = 1
names = ['ship']

for fold_idx in range(5):
    train_path = os.path.join(kfold_dir, f"fold_{fold_idx}", "images", "train").replace("\\", "/")
    val_path = os.path.join(kfold_dir, f"fold_{fold_idx}", "images", "val").replace("\\", "/")
    yaml_path = os.path.join(yaml_output_dir, f"fold_{fold_idx}.yaml")

    with open(yaml_path, "w") as f:
        f.write(f"train: {train_path}\n")
        f.write(f"val: {val_path}\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

print("5 YAML")

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

image_dirs = [
    r"C:\Users\PC1\Desktop\2D\X-band\images\train",
    r"C:\Users\PC1\Desktop\2D\X-band\images\val"
]

image_files = []
for image_dir in image_dirs:
    if os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            if f.endswith(".png") and "_" in f:
                try:
                    ts = int(f.split("_")[1].split(".")[0])
                    image_files.append((ts, f))
                except:
                    continue

image_files.sort(key=lambda x: x[0])
timestamps = [ts for ts, _ in image_files]
n_samples = len(timestamps)

tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(timestamps))

plt.figure(figsize=(12, 6))

for k, (train_idx, val_idx) in enumerate(splits):
    plt.scatter(train_idx, [k] * len(train_idx), color='blue', label='Train' if k == 0 else "")
    plt.scatter(val_idx, [k] * len(val_idx), color='orange', label='Val' if k == 0 else "")

plt.xlabel("Image Index (sorted by timestamp)")
plt.ylabel("Fold Index")
plt.title("TimeSeriesSplit (K-Fold) Visualization on Image Sequence")
plt.xticks([0, 1, 2, 3, 4])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

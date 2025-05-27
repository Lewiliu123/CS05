import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def visualize_radar_labels(image_id, root="C:/Users/PC1/Downloads/water"):
    # === 路径设置 ===
    image_path = os.path.join(root, "image", f"{image_id}.jpg")
    radar_path = os.path.join(root, "radar", f"{image_id}.csv")
    calib_path = os.path.join(root, "calib", f"{image_id}.txt")

    # === 图像加载 ===
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Image not found: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    # === 雷达数据加载 ===
    df = pd.read_csv(radar_path)
    if not {'x', 'y', 'z', 'label'}.issubset(df.columns):
        print(f"❌ Missing required columns in radar CSV: {radar_path}")
        return

    # === 提取点并过滤 label == -1 ===
    points_xyz = df[["x", "y", "z"]].values
    labels = df["label"].values.astype(int)

    if len(points_xyz) == 0:
        print("⚠️ No valid labeled points to display.")
        return

    # === 相机参数 ===
    with open(calib_path, "r") as f:
        lines = f.readlines()
    extrinsic = np.array([float(x) for x in lines[0].split(":")[1].strip().split()]).reshape(4, 4)
    intrinsic = np.array([float(x) for x in lines[1].split(":")[1].strip().split()]).reshape(3, 4)

    # === 投影到图像坐标 ===
    N = points_xyz.shape[0]
    points_homo = np.hstack((points_xyz, np.ones((N, 1))))  # (N, 4)
    cam_points = (extrinsic @ points_homo.T).T              # (N, 4)
    img_coords = (intrinsic @ cam_points[:, :4].T).T        # (N, 3)

    img_coords[:, 0] /= img_coords[:, 2]
    img_coords[:, 1] /= img_coords[:, 2]
    u = img_coords[:, 0].astype(int)
    v = img_coords[:, 1].astype(int)

    # === 过滤图像边界内的点 ===
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_img]
    v = v[in_img]
    labels = labels[in_img]

    if len(u) == 0:
        print("⚠️ All projected points are outside image bounds.")
        return

    # === 固定颜色映射（label 0~6）===
    label_names = ['pier', 'buoy', 'sailor', 'ship', 'boat', 'vessel', 'kayak']
    fixed_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    cmap = ListedColormap(fixed_colors)

    # === 可视化 ===
    plt.figure(figsize=(12, 7))
    plt.imshow(image)
    sc = plt.scatter(u, v, c=labels, cmap=cmap, s=8, alpha=0.9, vmin=0, vmax=6)

    # === 图例 ===
    legend_patches = [mpatches.Patch(color=fixed_colors[i], label=f"{i} - {label_names[i]}") for i in range(7)]
    plt.legend(handles=legend_patches, title="Label (ID - Name)", loc='upper right', fontsize=10, title_fontsize=11)

    plt.title(f"Radar Points (Colored by Label) on Image: {image_id}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# === 使用示例 ===
visualize_radar_labels("00193")

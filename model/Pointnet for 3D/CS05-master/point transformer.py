import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def convert_split(split_name, npy_dir, out_root):
    out_dir = os.path.join(out_root, split_name)
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]

    for fname in tqdm(files, desc=f"Converting {split_name}"):
        fpath = os.path.join(npy_dir, fname)
        data = np.load(fpath)
        xyz = data[:, :3]         # (N, 3)
        feats = data[:, :6]       # (N, 6)
        labels = data[:, 6]       # (N,)
        labels[labels == -1] = 7  # background 为第 8 类

        torch_data = Data(
            pos=torch.tensor(xyz, dtype=torch.float32),
            x=torch.tensor(feats, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.long)
        )

        out_path = os.path.join(out_dir, fname.replace(".npy", ".pth"))
        torch.save(torch_data, out_path)

# ==== 输入输出路径 ====
root_npy = "C:/Users/PC1/Downloads/water/radar_npy_-1"
output_pth_root = "C:/Users/PC1/Downloads/water/transformed_dataset"

# ==== 执行转换 ====
convert_split("train", os.path.join(root_npy, "train"), output_pth_root)
convert_split("val",   os.path.join(root_npy, "val"),   output_pth_root)
convert_split("test",  os.path.join(root_npy, "test"),  output_pth_root)



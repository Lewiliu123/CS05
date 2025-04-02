import os
import pandas as pd
import numpy as np

# CSV 原始数据目录
csv_dir = "C:/Users/PC1/Downloads/water/radar"

# 输出 .npy 保存目录
npy_dir = os.path.join("C:/Users/PC1/Downloads/water", "radar_npy_-1")
os.makedirs(npy_dir, exist_ok=True)

# 要保留的字段
fields = ['x', 'y', 'z', 'power', 'u', 'v', 'label']

for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_dir, filename)
        try:
            df = pd.read_csv(csv_path)

            # 确保字段存在
            if not all(f in df.columns for f in fields):
                print(f"❌ 缺少字段，跳过：{filename}")
                continue

            # 取需要的列
            data = df[fields].to_numpy()

            # 保存为 .npy
            base_name = os.path.splitext(filename)[0]
            np.save(os.path.join(npy_dir, base_name + ".npy"), data)
            print(f"✅ 已保存 {base_name}.npy")
        except Exception as e:
            print(f"⚠️ 处理出错：{filename} - {e}")

print("🎉 所有 CSV 文件已转换完成！")







import os
import shutil

# 原始 npy 数据文件夹
source_dir = "C:/Users/PC1/Downloads/water/radar_npy_-1"

# txt 文件路径（里面是类似 ./images/39484.jpg 这样的路径）
train_list_file = "C:/Users/PC1/Downloads/water/train.txt"
val_list_file = "C:/Users/PC1/Downloads/water/val.txt"
test_list_file = "C:/Users/PC1/Downloads/water/test.txt"

# 输出文件夹
train_dir = os.path.join(source_dir, "train")
val_dir = os.path.join(source_dir, "val")
test_dir = os.path.join(source_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 从 txt 中读取文件名（只保留数字或编号部分，变成 .npy）
def read_npy_list(txt_path):
    file_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                basename = os.path.splitext(os.path.basename(line))[0]
                npy_name = f"{basename}.npy"
                file_list.append(npy_name)
    return file_list

train_files = read_npy_list(train_list_file)
val_files = read_npy_list(val_list_file)
test_files = read_npy_list(test_list_file)

# 拷贝函数
def move_files(files, target_dir):
    for file in files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, file)
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"⚠️ 文件不存在：{file}")

# 执行移动
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print(f"✅ 数据划分完成：\nTrain: {len(train_files)}\nVal: {len(val_files)}\nTest: {len(test_files)}")


import pandas as pd
import numpy as np

file_path = "C:/Users/PC1/Downloads/water/radar_npy_-1/train/39484.npy"

data = np.load(file_path)

# 人工指定列名
columns = ['x', 'y', 'z', 'power', 'u', 'v', 'label']

df = pd.DataFrame(data, columns=columns)

print("📐 数据形状:", df.shape)
print("📋 前几行数据：")
print(df.head())

import os
import numpy as np
from collections import Counter
from tqdm import tqdm

# Path to your dataset directory (including train/val/test)
base_dir = "C:/Users/PC1/Downloads/water/radar_npy_-1"

# Collect all .npy files (including in subdirectories)
npy_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".npy"):
            npy_files.append(os.path.join(root, f))

# Count label occurrences (including -1)
label_counter = Counter()

print(f"Processing {len(npy_files)} files...")

for file in tqdm(npy_files, desc="Counting labels"):
    data = np.load(file)
    labels = data[:, 6].astype(int)  # Column 7 is label
    label_counter.update(labels)

# Print results
print("\nLabel Distribution (including -1):")
for label, count in sorted(label_counter.items()):
    label_name = "Background (-1)" if label == -1 else f"Class {label}"
    print(f"{label_name}: {count} points")

print(f"\nTotal {len(label_counter)} unique labels (including -1)")


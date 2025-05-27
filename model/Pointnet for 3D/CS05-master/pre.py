import os
import numpy as np
from tqdm import tqdm

input_dir = "C:/Users/PC1/Downloads/water/radar_npy"  # 原始路径
output_dir = "C:/Users/PC1/Downloads/water/radar_npy_filtered"  # 输出路径
os.makedirs(output_dir, exist_ok=True)

for filename in tqdm(os.listdir(input_dir), desc="Filtering -1 labels"):
    if not filename.endswith(".npy"):
        continue

    file_path = os.path.join(input_dir, filename)
    data = np.load(file_path)  # (N, 7)

    # 过滤掉 label == -1 的点
    labels = data[:, 6]
    mask = labels >= 0
    filtered_data = data[mask]

    # 如果全部被过滤，可以跳过 or 继续保存空的（可选）
    if filtered_data.shape[0] == 0:
        print(f"⚠️ All points removed in {filename}, skipping.")
        continue

    save_path = os.path.join(output_dir, filename)
    np.save(save_path, filtered_data)


import numpy as np

# 修改为你要检查的 .npy 文件路径
npy_path = "C:/Users/PC1/Downloads/water/radar_npy_-1/test/53571.npy"

# 加载数据
data = np.load(npy_path)

print(f"📦 文件包含点数: {data.shape[0]}")
print(f"📐 数据维度: {data.shape}")
print(f"🔢 每列含义: [x, y, z, power, u, v, label]\n")

# 打印前几个点
print("📄 前 5 个点示例：")
for i in range(min(5, len(data))):
    x, y, z, power, u, v, label = data[i]
    print(f"点 {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}, power={power:.1f}, u={u:.1f}, v={v:.1f}, label={label}")

# 检查 u, v 是否在合理范围内
u = data[:, 4]
v = data[:, 5]

print("\n🧪 u, v 范围检查：")
print(f"u 最小值: {u.min()}, 最大值: {u.max()}")
print(f"v 最小值: {v.min()}, 最大值: {v.max()}")

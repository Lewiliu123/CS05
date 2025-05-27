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

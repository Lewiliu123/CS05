import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob


class RadarPointCloudDataset(Dataset):
    def __init__(self, file_list, num_points=1024):
        self.files = file_list
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # (N, 7)
        points = data[:, :6].astype(np.float32)
        labels = data[:, 6].astype(np.int64)

        labels[labels == -1] = 7

        if len(points) >= self.num_points:
            idxs = np.random.choice(len(points), self.num_points, replace=False)
        else:
            pad = np.random.choice(len(points), self.num_points - len(points), replace=True)
            idxs = np.concatenate([np.arange(len(points)), pad])

        points = points[idxs]
        labels = labels[idxs]

        return torch.tensor(points), torch.tensor(labels)


import torch
import torch.nn as nn
import torch.nn.functional as F



class PointNet(nn.Module):
    def __init__(self, input_dim=3, num_classes=50):
        super(PointNet, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.feature_mlp = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.seg_mlp = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):  # (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N)
        x1 = self.input_mlp(x)  # (B, 64, N)
        x2 = self.feature_mlp(x1)  # (B, 1024, N)
        x_global = torch.max(x2, 2, keepdim=True)[0]  # (B, 1024, 1)
        x_global = x_global.repeat(1, 1, x.size(2))  # (B, 1024, N)
        x_concat = torch.cat([x1, x_global], 1)  # (B, 64+1024=1088, N)
        out = self.seg_mlp(x_concat)  # (B, num_classes, N)
        return out.transpose(1, 2)  # (B, N, num_classes)



def train_pointnet(data_dir, num_points=1024, batch_size=200, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".npy")]
    val_files   = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".npy")]
    test_files  = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".npy")]

    train_ds = RadarPointCloudDataset(train_files, num_points)
    val_ds = RadarPointCloudDataset(val_files, num_points)
    test_ds = RadarPointCloudDataset(test_files, num_points)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = PointNet(input_dim=6, num_classes=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    counts = torch.tensor([
        946559,  # Class 0
        221502,  # Class 1
        36962,  # Class 2
        2764193,  # Class 3
        399903,  # Class 4
        888027,  # Class 5
        2372,  # Class 6
        13348513  # Background (-1), now class 7
    ], dtype=torch.float32)

    # Compute inverse frequency weights and normalize
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum()

    # Define criterion using weights
    class_weights = weights.to(device)  # Make sure 'device' is defined (cuda or cpu)
    criterion = nn.CrossEntropyLoss()

    from tqdm import tqdm
    train_acc_list, val_acc_list, loss_list, val_loss_list = [], [], [],[]

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_points = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}")
        for pts, lbls in pbar:
            pts, lbls = pts.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(pts)
            loss = criterion(out.permute(0, 2, 1), lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (lbls != -1).sum().item()  # 只统计有效点的 loss
            pred = out.argmax(dim=-1)
            valid_mask = lbls != -1
            total_correct += (pred[valid_mask] == lbls[valid_mask]).sum().item()
            total_points += valid_mask.sum().item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Train Acc": f"{(total_correct / total_points):.4f}"
            })

        train_acc = total_correct / total_points
        train_acc_list.append(train_acc)

        avg_loss = total_loss / total_points
        loss_list.append(avg_loss)

        print(f"Epoch {epoch + 1:02d} | Avg Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        model.eval()
        val_loss, val_correct, val_points = 0, 0, 0
        with torch.no_grad():
            for pts, lbls in val_loader:
                pts, lbls = pts.to(device), lbls.to(device)
                out = model(pts)
                loss = criterion(out.permute(0, 2, 1), lbls)

                val_loss += loss.item() * lbls.numel()  # or use (lbls != -1).sum() if you want
                pred = out.argmax(dim=-1)
                val_correct += (pred == lbls).sum().item()
                val_points += lbls.numel()

        val_acc = val_correct / val_points
        avg_val_loss = val_loss / val_points

        val_acc_list.append(val_acc)
        val_loss_list.append(avg_val_loss)

        print(f"Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")

    model.eval()
    test_correct, test_points = 0, 0
    from tqdm import tqdm

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for pts, lbls in pbar:
            pts, lbls = pts.to(device), lbls.to(device)
            out = model(pts)
            pred = out.argmax(dim=-1)
            test_correct += (pred == lbls).sum().item()
            test_points += lbls.numel()
            acc = test_correct / test_points
            pbar.set_postfix({"Running Acc": f"{acc:.4f}"})

    test_acc = test_correct / test_points
    print(f"Test Accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), "100_without_class_weights.pth")
    print("save pointnet_trained.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, label="Train Accuracy", marker='o')
    plt.plot(val_acc_list, label="Val Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("100_without_class_weights.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label="Train Loss", marker='o', color='red')
    plt.plot(val_loss_list, label="Val Loss", marker='o', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("100_without_class_weights.png", dpi=150)
    plt.show()


# ========== Run ==========
data_dir = "C:/Users/PC1/Downloads/water/radar_npy_-1"
#train_pointnet(data_dir)

print("\nEvaluating on the full test set (with loss)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load
model = PointNet(input_dim=6, num_classes=8).to(device)
model.load_state_dict(torch.load("100_without_class_weights.pth"))

model.eval()

# test data
test_dir = os.path.join(data_dir, "test")
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".npy")]
test_ds = RadarPointCloudDataset(test_files, num_points=1024)
test_loader = DataLoader(test_ds, batch_size=30, shuffle=False)

# loss
criterion = nn.CrossEntropyLoss()

test_correct, test_points, total_loss = 0, 0, 0
from tqdm import tqdm

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Testing")
    for pts, lbls in pbar:
        pts, lbls = pts.to(device), lbls.to(device)
        out = model(pts)
        loss = criterion(out.permute(0, 2, 1), lbls)

        pred = out.argmax(dim=-1)
        test_correct += (pred == lbls).sum().item()
        test_points += lbls.numel()
        total_loss += loss.item() * lbls.numel()

        acc = test_correct / test_points
        pbar.set_postfix({"Running Acc": f"{acc:.4f}", "Loss": f"{loss.item():.4f}"})

test_acc = test_correct / test_points
avg_test_loss = total_loss / test_points

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss: {avg_test_loss:.4f}")

from sklearn.metrics import classification_report
import numpy as np

# ====== precision / recall / f1 ======
all_preds = []
all_labels = []

with torch.no_grad():
    for pts, lbls in test_loader:
        pts, lbls = pts.to(device), lbls.to(device)
        out = model(pts)
        pred = out.argmax(dim=-1)

        all_preds.append(pred.cpu().numpy())
        all_labels.append(lbls.cpu().numpy())

all_preds = np.concatenate([p.flatten() for p in all_preds])
all_labels = np.concatenate([l.flatten() for l in all_labels])

all_labels[all_labels == -1] = 7
all_preds = np.clip(all_preds, 0, 7)

label_names = ['pier', 'buoy', 'sailor', 'ship', 'boat', 'vessel', 'kayak', 'other']

print("\nClassification Report:")
report_str = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
print(report_str)



# ========== Evaluate & Visualize ==========
import os
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== calib =====
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    ext_values = list(map(float, lines[0].split(":")[1].strip().split()))
    ext_matrix = np.array(ext_values).reshape(4, 4)

    int_values = list(map(float, lines[1].split(":")[1].strip().split()))
    K = np.array(int_values).reshape(3, 4)[:, :3]  # 去掉最后一列 0

    return ext_matrix, K

# ===== visualization =====
def visualize_prediction_vs_groundtruth(points, labels, preds, image_path, ext, K):

    import matplotlib.patches as mpatches

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    label_names = ['pier', 'buoy', 'sailor', 'ship', 'boat', 'vessel', 'kayak', 'other']
    custom_colors = {
        0: (0, 0, 255),        # blue
        1: (255, 165, 0),      # orange
        2: (0, 128, 0),        # green
        3: (255, 0, 0),        # red
        4: (128, 0, 128),      # purple
        5: (139, 69, 19),      # brown
        6: (255, 192, 203),    # pink
        7: (128, 128, 128),    # gray for other
    }

    u = points[:, 4].astype(int)
    v = points[:, 5].astype(int)

    # Ground truth
    gt_image = image.copy()
    for i in range(len(u)):
        cls = int(labels[i])
        if 0 <= u[i] < W and 0 <= v[i] < H:
            color = custom_colors.get(cls, (255, 255, 255))  # fallback white
            cv2.circle(gt_image, (u[i], v[i]), radius=6, color=color, thickness=-1)

    # Prediction
    pred_image = image.copy()
    for i in range(len(u)):
        cls = int(preds[i])
        if 0 <= u[i] < W and 0 <= v[i] < H:
            color = custom_colors.get(cls, (255, 255, 255))
            cv2.circle(pred_image, (u[i], v[i]), radius=6, color=color, thickness=-1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(gt_image)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred_image)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    handles = []
    for i in range(len(label_names)):
        color = np.array(custom_colors[i]) / 255
        handles.append(mpatches.Patch(color=color, label=f"{i} - {label_names[i]}"))

    fig.legend(handles=handles, loc="upper right", fontsize=12, title="Label (ID - Name)")
    plt.tight_layout()
    plt.show()

print("\n Visualizing Prediction Example...")
sample_name = "00072"

npy_path   = os.path.join("C:/Users/PC1/Downloads/water/radar_npy_-1/test", f"{sample_name}.npy")
img_path   = os.path.join("C:/Users/PC1/Downloads/water/image", f"{sample_name}.jpg")
calib_path = os.path.join("C:/Users/PC1/Downloads/water/calib", f"{sample_name}.txt")

ext, K = load_calibration(calib_path)

data = np.load(npy_path)
points_all = data[:, :6].astype(np.float32)
labels_all = data[:, 6].astype(np.int64)
labels_all[labels_all == -1] = 7

if len(points_all) >= 1024:
    idx = np.random.choice(len(points_all), 1024, replace=False)
else:
    pad = np.random.choice(len(points_all), 1024 - len(points_all), replace=True)
    idx = np.concatenate([np.arange(len(points_all)), pad])

points = points_all[idx]
labels = labels_all[idx]

model = PointNet(input_dim=6, num_classes=8).to(device)
model.load_state_dict(torch.load("100_without_class_weights.pth"))

model.eval()

input_tensor = torch.tensor(points).unsqueeze(0).to(device)
with torch.no_grad():
    preds = model(input_tensor).argmax(dim=-1).squeeze().cpu().numpy()
    preds = np.clip(preds, 0, 7)

visualize_prediction_vs_groundtruth(points, labels, preds, img_path, ext, K)

# Input (B, N, C)
#    │
#    ├─ input_mlp (per-point encoding) → (B, 64, N) → x1
#    │
#    ├─ feature_mlp (extract global features) → (B, 1024, N)
#    ├─ max pooling → x_global: (B, 1024, 1)
#    ├─ repeat across all points → (B, 1024, N)
#    │
#    ├─ concat(x1, x_global) → (B, 1088, N)
#    ├─ seg_mlp (per-point classification) → (B, num_classes, N)
#    └─ transpose → (B, N, num_classes)

import itertools

lr0_list = [0.001, 0.003, 0.005]
momentum_list = [0.85, 0.9, 0.937]
weight_decay_list = [0.0001, 0.0005]
hsv_h_list = [0.01, 0.015]
scale_list = [0.4, 0.5]
batch_list = [16]

dataset = r"C:/Users/PC1/Desktop/2D/X-band/dataset.yaml"
imgsz = 1024
epochs = 30

with open("run_hyperparam_trials.bat", "w") as f:
    for i, (lr0, momentum, weight_decay, hsv_h, scale, batch) in enumerate(itertools.product(
        lr0_list, momentum_list, weight_decay_list, hsv_h_list, scale_list, batch_list
    )):
        cmd = (
            f'yolo detect train model=yolov8n.pt data="{dataset}" imgsz={imgsz} epochs={epochs} '
            f'project=autotune name=trial_{i} '
            f'lr0={lr0} momentum={momentum} weight_decay={weight_decay} hsv_h={hsv_h} scale={scale} batch={batch}'
        )
        f.write(cmd + '\n')

import itertools

lr0_list = [0.001, 0.005, 0.01, 0.1]
weight_decay_list = [0.0005, 0.001]
imgsz_options = [2048, 1024]
batch_list = [3]

dataset = r"C:/Users/PC1/Desktop/2D/X-band/dataset.yaml"
epochs = 200

with open("run_hyperparam_trials_imgsize_100_adam.bat", "w") as f:
    for i, (lr0, weight_decay, imgsz, batch) in enumerate(itertools.product(
        lr0_list, weight_decay_list, imgsz_options, batch_list
    )):
        cmd = (
            f'yolo detect train model=yolo11n.pt data="{dataset}" imgsz={imgsz} epochs={epochs} '
            f'project=imagesize_100_adam name=trial_{i} '
            f'lr0={lr0} weight_decay={weight_decay} batch={batch} optimizer=adam'
        )
        f.write(cmd + '\n')

print("run_hyperparam_trials_imgsize_100_adam.bat")

import itertools

lr0_list = [0.0001, 0.001, 0.01, 0.1]
weight_decay_list = [0.0005, 0.01]
imgsz_options = [2048, 1024]
batch_list = [3]
epochs = 30

yaml_dir = r"C:/Users/PC1/Desktop/2D/X-band/kfold_split_yaml"

output_bat = "run_kfold_hyper_adam_30epoch.bat"

with open(output_bat, "w") as f:
    combo_id = 0
    for fold in range(5):
        for lr0, weight_decay, imgsz, batch in itertools.product(
            lr0_list, weight_decay_list, imgsz_options, batch_list
        ):
            name = f"fold{fold}_adam_lr{lr0}_wd{weight_decay}_sz{imgsz}_bs{batch}"
            cmd = (
                f'yolo detect train model=yolo11n.pt data={yaml_dir}/fold_{fold}.yaml '
                f'imgsz={imgsz} epochs={epochs} optimizer=adam '
                f'lr0={lr0} weight_decay={weight_decay} batch={batch} '
                f'project=kfold_hyper_adam name={name}'
            )
            f.write(cmd + '\n')
            combo_id += 1

print("5 folds")



import os
import pandas as pd

base_path = r"C:\Users\PC1\AppData\Roaming\JetBrains\PyCharm2024.3\light-edit\autotune"
bat_path = r"C:\Users\PC1\AppData\Roaming\JetBrains\PyCharm2024.3\light-edit\run_hyperparam_trials.bat"

all_results = []
for folder in os.listdir(base_path):
    exp_path = os.path.join(base_path, folder)
    csv_path = os.path.join(exp_path, 'results.csv')
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        if 'metrics/mAP50-95(B)' in df.columns:
            best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
            all_results.append({
                'experiment': folder,
                'mAP50(B)': best_row['metrics/mAP50(B)'],
                'mAP50-95(B)': best_row['metrics/mAP50-95(B)'],
                'epoch': best_row['epoch']
            })

df_all = pd.DataFrame(all_results)
top10 = df_all.sort_values(by='mAP50-95(B)', ascending=False).head(50)

print("Top 10 Experiments:")
print(top10)

with open(bat_path, 'r') as f:
    lines = f.readlines()

print("\nMatched commands from .bat:")

for exp_name in top10['experiment']:
    match = [line.strip() for line in lines if f'name={exp_name}' in line]
    if match:
        print(f"{exp_name}: {match[0]}")
    else:
        print(f"{exp_name}: ⚠ Not found in .bat")


import os
import pandas as pd
import matplotlib.pyplot as plt

base_path = r"C:\Users\PC1\AppData\Roaming\JetBrains\PyCharm2024.3\light-edit\imagesize_100_adam"

target_trials = ['trial_19', 'trial_51', 'trial_3', 'trial_63', 'trial_7']

plt.figure(figsize=(12, 6))

for trial in target_trials:
    csv_path = os.path.join(base_path, trial, "results.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        if 'metrics/mAP50-95(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label=trial)

plt.title("Training mAP50-95(B) Curve per Epoch")
plt.xlabel("Epoch")
plt.ylabel("mAP50-95(B)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

base_path = r"C:\Users\PC1\AppData\Roaming\JetBrains\PyCharm2024.3\light-edit\kfold_hyper_adam"
all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

results = []

for folder in all_folders:
    result_csv = os.path.join(base_path, folder, "results.csv")
    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)
        if 'metrics/mAP50-95(B)' in df.columns:
            best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
            results.append({
                "experiment": folder,
                "epoch": int(best_row["epoch"]),
                "mAP50": best_row["metrics/mAP50(B)"],
                "mAP50-95": best_row["metrics/mAP50-95(B)"],
                "precision": best_row["metrics/precision(B)"],
                "recall": best_row["metrics/recall(B)"],
                "val_loss": best_row["val/box_loss"] + best_row["val/cls_loss"] + best_row["val/dfl_loss"]
            })

df_all = pd.DataFrame(results)
df_all = df_all.sort_values(by="mAP50-95", ascending=False)

print("Top 10 Experiments by mAP50-95(B):")
print(df_all.head(10)[["experiment", "epoch", "mAP50", "mAP50-95", "precision", "recall", "val_loss"]])

import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = r"C:\Users\PC1\AppData\Roaming\JetBrains\PyCharm2024.3\light-edit\kfold_hyper_adam"

selected_experiments = [
    "fold1_adam_lr0.001_wd0.0005_sz2048_bs3",
    "fold1_adam_lr0.001_wd0.01_sz2048_bs3",
    "fold3_adam_lr0.01_wd0.01_sz1024_bs3",
    "fold2_adam_lr0.001_wd0.01_sz2048_bs3",
]

plt.figure(figsize=(12, 6))

for exp in selected_experiments:
    csv_path = os.path.join(base_dir, exp, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label=exp)

plt.xlabel("Epoch")
plt.ylabel("mAP50-95(B)")
plt.title("mAP50-95(B) Curve per Epoch for Selected Experiments")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


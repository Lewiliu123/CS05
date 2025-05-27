import os
import pandas as pd

base_dir = r"C:\Users\PC1\AppData\Roaming\JetBrains\PyCharm2024.3\light-edit\kfold_hyper_adam"

results = {}

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    result_csv = os.path.join(folder_path, "results.csv")
    if os.path.isfile(result_csv):
        df = pd.read_csv(result_csv)
        if 'metrics/mAP50-95(B)' in df.columns:
            best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
            config = folder.split("_", 1)[1]
            fold = folder.split("_")[0]
            if config not in results:
                results[config] = []
            results[config].append({
                'fold': fold,
                'mAP50-95': best_row['metrics/mAP50-95(B)'],
                'mAP50': best_row['metrics/mAP50(B)'],
                'precision': best_row['metrics/precision(B)'],
                'recall': best_row['metrics/recall(B)']
            })

summary = []
for config, records in results.items():
    if len(records) == 5:
        avg_map = sum(r['mAP50-95'] for r in records) / 5
        avg_map50 = sum(r['mAP50'] for r in records) / 5
        avg_prec = sum(r['precision'] for r in records) / 5
        avg_rec = sum(r['recall'] for r in records) / 5
        summary.append({
            'config': config,
            'avg_mAP50-95': avg_map,
            'avg_mAP50': avg_map50,
            'avg_precision': avg_prec,
            'avg_recall': avg_rec
        })

summary_df = pd.DataFrame(summary)
top10 = summary_df.sort_values(by='avg_mAP50-95', ascending=False).head(10)

print(top10)


import matplotlib.pyplot as plt

top_configs = top10['config'].tolist()[:5]

plt.figure(figsize=(12, 6))
for config in top_configs:
    records = results[config]
    folds = [int(''.join(filter(str.isdigit, r['fold']))) for r in records]
    maps = [r['mAP50-95'] for r in records]
    plt.plot(folds, maps, marker='o', label=config)

plt.xlabel("Fold Index")
plt.ylabel("mAP50-95")
plt.title("Top 5 Configs - mAP50-95 over K-Fold")
plt.xticks([0, 1, 2, 3, 4])
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


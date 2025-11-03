import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data.IRDataSet import IRDataSet
from data.QM9Set import QM9IRDataSet

import matplotlib.pyplot as plt
import seaborn as sns

# 确保绘图风格统一
sns.set(style="whitegrid")


dataset = IRDataSet('data/common_rows_with_gap.csv')

X_valid = np.array([dataset[i]['ir'].tolist() for i in range(len(dataset))])
y_valid = np.array([dataset[i]['homo_lumo_gap'] for i in range(len(dataset))])



qm9set = QM9IRDataSet("/home/yanggk/Data/SpecGLU/QM9/SpecBert/gaussian_summary.csv")


X = np.array([qm9set[i]['ir'].tolist() for i in range(len(qm9set))])
y = np.array([qm9set[i]['homo_lumo_gap'] for i in range(len(qm9set))])
# 数据划分

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 模型集合
models = {
    "ExtraTrees": ExtraTreesRegressor(n_estimators=800, max_depth=100, random_state=42, n_jobs=24),
    }


# 训练与评估
for name, model in models.items():
    print(f"\n=== {name} Training ===")
    model.fit(X_train, y_train)
    print(f"\n=== {name} Finish Training ===")
    print(f"\n=== {name} Testing ===")
    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 Score:", r2_score(y_test, preds))
        # ====== 添加测试集绘图 ======
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 理想拟合线
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    # plt.text()
    plt.title(f"{name} Prediction on Test Set")
    plt.tight_layout()
    plt.savefig(f"figs/{name}_scatter_plot_test.png", dpi=300)
    plt.cla()
    plt.close()
    with open(f'data/{name}_model_test.txt', 'w') as f:
        f.write('True, Predicted\n')
        for true, pred in zip(y_test, preds):
            f.write(f"{true}, {pred}\n")
    # plt.show()

    print(f"\n=== {name} Validing ===")
    preds = model.predict(X_valid)
    print("MAE:", mean_absolute_error(y_valid, preds))
    print("R2 Score:", r2_score(y_valid, preds))
        # ====== 添加验证集绘图 ======
    plt.figure(figsize=(6, 6))
    plt.scatter(y_valid, preds, alpha=0.6, edgecolor='k')
    plt.plot([y_valid.min(), y_valid.max()], [y_test.min(), y_test.max()], 'r--')  # 理想拟合线
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    # plt.text()
    plt.title(f"{name} Prediction on Test Set")
    plt.tight_layout()
    plt.savefig(f"figs/{name}_scatter_plot_valid.png", dpi=300)
    plt.cla()
    with open(f'data/{name}_model_trans.txt', 'w') as f:
        f.write('True, Predicted\n')
        for true, pred in zip(y_test, preds):
            f.write(f"{true}, {pred}\n")

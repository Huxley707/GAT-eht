import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')


# 复用你的标签处理器以保证对齐
class PhotocatalysisLabelProcessor:
    TASK_COLUMNS = {
        'band_gap_pred': 'band_gap_pred',
        'distortion_pred': 'distortion_pred',
        'ionization_energy': 'ionization_energy',
    }

    def __init__(self, task_names=None):
        self.task_names = task_names or list(self.TASK_COLUMNS.keys())
        self.column_names = [self.TASK_COLUMNS[t] for t in self.task_names if t in self.TASK_COLUMNS]
        self.scalers = {}
        self.stats = {}
        self.valid_mask = None

    def process_labels(self, df):
        processed_labels = {}
        n_samples = len(df)
        all_valid_masks = []

        for task_name, col_name in zip(self.task_names, self.column_names):
            if col_name not in df.columns:
                continue

            y = df[col_name].values.copy()
            valid_mask = ~np.isnan(y) & ~np.isinf(y)
            all_valid_masks.append(valid_mask)

            # 异常值处理
            q1, q3 = np.percentile(y[valid_mask], 25), np.percentile(y[valid_mask], 75)
            iqr = q3 - q1
            y_clean = np.clip(y[valid_mask], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            y[valid_mask] = y_clean

            # RobustScaler 归一化
            scaler = RobustScaler()
            y_normalized = y.copy()
            scaler.fit(y[valid_mask].reshape(-1, 1))
            y_normalized[valid_mask] = scaler.transform(y[valid_mask].reshape(-1, 1)).flatten()

            self.scalers[task_name] = scaler
            processed_labels[task_name] = {
                'raw': y,
                'normalized': y_normalized,
                'valid_mask': valid_mask
            }

        if all_valid_masks:
            self.valid_mask = np.all(all_valid_masks, axis=0)

        return processed_labels

    def inverse_transform(self, normalized_predictions, task_name):
        if task_name not in self.scalers:
            return normalized_predictions
        scaler = self.scalers[task_name]
        return scaler.inverse_transform(normalized_predictions.reshape(-1, 1)).flatten()


def run_rf_experiment():
    print("=" * 70)
    print("光催化性能预测 - 随机森林 (RF) Baseline 训练流水线")
    print("=" * 70)

    # 1. 设置路径
    features_path = os.path.join("features", 'photocatalysis_features.csv')
    targets_path = os.path.join("features", 'photocatalysis_targets.csv')

    run_name = f"rf_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join("runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(features_path) or not os.path.exists(targets_path):
        print("❌ 错误: 请确保 features/ 目录下存在 photocatalysis_features.csv 与 photocatalysis_targets.csv")
        return

    # 2. 读取特征与标签
    features_df = pd.read_csv(features_path)
    targets_df = pd.read_csv(targets_path)

    task_names = ['band_gap_pred', 'distortion_pred', 'ionization_energy']
    label_processor = PhotocatalysisLabelProcessor(task_names)
    processed_labels = label_processor.process_labels(targets_df)

    # 对齐全局有效样本
    valid_mask = label_processor.valid_mask
    print(f"✅ 全局有效样本数: {valid_mask.sum()}/{len(targets_df)}")

    # 准备特征矩阵 X (排除非数值/文件名列)
    drop_cols = ['filename', 'mp_id', 'formula']
    X_cols = [c for c in features_df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(features_df[c])]

    # 填补特征矩阵缺失值 (如存在)
    X = features_df.loc[valid_mask, X_cols].fillna(0).values

    # 准备目标矩阵 Y (归一化尺度，与 GAT 训练目标一致)
    Y_norm_list = [processed_labels[t]['normalized'][valid_mask] for t in task_names]
    Y_norm = np.column_stack(Y_norm_list)

    Y_raw_list = [processed_labels[t]['raw'][valid_mask] for t in task_names]
    Y_raw = np.column_stack(Y_raw_list)

    # 3. 5-Fold 交叉验证评估
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {t: {'mae_norm': [], 'mae_raw': [], 'r2': []} for t in task_names}
    feature_importances = []

    print("\n开始 5-Fold 交叉验证...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train_norm, Y_test_norm = Y_norm[train_idx], Y_norm[test_idx]
        Y_test_raw = Y_raw[test_idx]

        # 多输出随机森林模型
        rf_base = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(rf_base)
        model.fit(X_train, Y_train_norm)

        # 预测 (归一化尺度)
        preds_norm = model.predict(X_test)

        # 记录各任务评估结果
        for i, task_name in enumerate(task_names):
            p_norm = preds_norm[:, i]
            t_norm = Y_test_norm[:, i]

            # 反归一化到原始尺度
            p_raw = label_processor.inverse_transform(p_norm, task_name)
            t_raw = Y_test_raw[:, i]

            mae_n = mean_absolute_error(t_norm, p_norm)
            mae_r = mean_absolute_error(t_raw, p_raw)
            r2 = r2_score(t_norm, p_norm)

            metrics[task_name]['mae_norm'].append(mae_n)
            metrics[task_name]['mae_raw'].append(mae_r)
            metrics[task_name]['r2'].append(r2)

        # 收集特征重要性 (平均 3 个任务的 RF 特征重要性)
        imp = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
        feature_importances.append(imp)

    # 4. 打印汇总评估结果
    print("\n================ FINAL RF RESULTS (5-Fold Mean) ================")
    summary_data = []
    for task_name in task_names:
        m = metrics[task_name]
        mean_mae_norm = np.mean(m['mae_norm'])
        mean_mae_raw = np.mean(m['mae_raw'])
        mean_r2 = np.mean(m['r2'])

        print(f"📊 任务: {task_name}")
        print(f"   MAE (归一化): {mean_mae_norm:.4f}")
        print(f"   MAE (原始):   {mean_mae_raw:.4f}")
        print(f"   R² Score:     {mean_r2:.4f}\n")

        summary_data.append({
            'Task': task_name,
            'MAE_Norm': mean_mae_norm,
            'MAE_Raw': mean_mae_raw,
            'R2_Score': mean_r2
        })

    # 保存指标摘要到 CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "rf_metrics_summary.csv"), index=False)

    # 5. 绘制 Top-15 重要特征图
    mean_importances = np.mean(feature_importances, axis=0)
    top_idx = np.argsort(mean_importances)[-15:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(15), mean_importances[top_idx], align='center', color='skyblue')
    plt.yticks(range(15), [X_cols[i] for i in top_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances (Random Forest Baseline)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rf_feature_importance.png"))
    plt.close()

    print(f"✅ 随机森林对比实验完成！结果已保存至: {output_dir}/")


if __name__ == "__main__":
    run_rf_experiment()
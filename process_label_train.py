"""
光催化性能预测完整特征工程流水线
适用于GAT网络训练，支持多任务学习和可解释性输出
修正版本 - 与特征工程输出完全对应（仅包含准确计算的特征）
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import skew, kurtosis
from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.analysis.ewald import EwaldSummation
import warnings
import os
import glob
import traceback
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ==================== 数据封装类 ====================

class PhotocatalysisData(Data):
    """光催化图数据封装类"""

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 global_features=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, **kwargs)
        self.global_features = global_features

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'global_features':
            return None  # 全局特征不进行拼接
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'global_features':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


# ==================== GAT模型定义 ====================

class PhotocatalysisGAT(nn.Module):
    def __init__(self, node_in_dim=12, edge_in_dim=8, global_in_dim=32,
                 hidden_dim=64, num_heads=4, output_dim=3):
        """
        参数:
            node_in_dim: 节点特征维度 (来自特征工程: 12)
            edge_in_dim: 边特征维度 (来自特征工程: 8，移除了键级估计)
            global_in_dim: 全局特征维度 (来自特征工程: 动态确定)
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            output_dim: 输出维度 (对应3个准确计算的目标特征)
        """
        super().__init__()

        # 保存维度信息
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.output_dim = output_dim

        # 节点编码
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # GATv2层
        self.gat1 = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_in_dim,
            dropout=0.1
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_in_dim,
            dropout=0.1
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.gat3 = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_in_dim,
            dropout=0.1

        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.attentions = {}

        # 全局特征处理
        self.global_encoder = nn.Sequential(
            nn.Linear(global_in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )

        # 图特征投影
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid()
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 处理全局特征
        if hasattr(data, 'global_features') and data.global_features is not None:
            global_features = data.global_features
            # 确保全局特征是2维 [batch_size, global_dim]
            if global_features.dim() == 3:
                global_features = global_features.squeeze(1)
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)
        else:
            # 如果没有全局特征，创建默认特征
            batch_size = batch.max().item() + 1
            global_features = torch.zeros(batch_size, 32, device=x.device)

        # 节点编码
        x = self.node_encoder(x)
        x = self.node_norm(x)
        x = F.relu(x)

        # GAT层1
        x1 = self.gat1(x, edge_index, edge_attr)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        # GAT层2（带残差）
        x2 = self.gat2(x1, edge_index, edge_attr)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)

        # GAT层3（带残差）
        x3 = self.gat3(x2, edge_index, edge_attr)
        x3 = self.norm3(x3)
        x3 = F.relu(x3)

        # 图级池化
        x_final=x1+x2+x3
        graph_features = global_mean_pool(x_final, batch)  # [batch_size, hidden_dim]

        # 处理全局特征
        global_encoded = self.global_encoder(global_features)  # [batch_size, hidden_dim//2]

        # 图特征投影
        graph_proj = self.graph_proj(graph_features)  # [batch_size, hidden_dim//2]

        # 门控融合
        fusion_input = torch.cat([graph_proj, global_encoded], dim=1)  # [batch_size, hidden_dim]
        gate = self.fusion_gate(fusion_input)  # [batch_size, hidden_dim//2]

        # 加权融合
        fused = gate * graph_proj + (1 - gate) * global_encoded  # [batch_size, hidden_dim//2]

        # 输出预测
        out = self.output_layer(fused)  # [batch_size, output_dim]

        return out


# ==================== 物理约束类 ====================

class PhysicalConstraints(nn.Module):
    """
    将物理约束编码到损失函数中
    提高模型的可解释性和物理合理性
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # 标准电极电位 (eV vs vacuum)
        self.register_buffer('H2_H2O_potential', torch.tensor(-4.44))
        self.register_buffer('O2_H2O_potential', torch.tensor(-5.67))

    def to(self, device):
        """移动张量到指定设备"""
        self.device = device
        return super().to(device)

    def band_position_constraint(self, cbm_pred, vbm_pred):
        """
        导带底必须高于价带顶
        L_constraint = max(0, vbm - cbm)²
        """
        violation = torch.relu(vbm_pred - cbm_pred)
        return torch.mean(violation ** 2)

    def band_gap_consistency(self, band_gap_pred, cbm_pred, vbm_pred):
        """
        带隙 = 导带底 - 价带顶
        L_consistency = (band_gap - (cbm - vbm))²
        """
        return torch.mean((band_gap_pred - (cbm_pred - vbm_pred)) ** 2)

    def redox_capability_constraint(self, cbm_pred, vbm_pred):
        """
        氧化还原能力约束
        导带应足够负以还原H⁺，价带应足够正以氧化H₂O
        """
        # 产氢能力: 导带 < H⁺/H₂电位
        h2_violation = torch.relu(cbm_pred - self.H2_H2O_potential)

        # 产氧能力: 价带 > O₂/H₂O电位
        o2_violation = torch.relu(self.O2_H2O_potential - vbm_pred)

        return torch.mean(h2_violation ** 2 + o2_violation ** 2)


# ==================== 标签处理器 ====================

class PhotocatalysisLabelProcessor:
    """
    光催化性能标签处理器
    处理多任务学习中的标签
    与特征工程中的3个准确计算目标特征完全对应
    """

    # 任务名称到列的映射 - 只保留准确计算的特征
    TASK_COLUMNS = {
        # 核心性能指标 - 对应targets.csv中的列名
        'band_gap_pred': 'band_gap_pred',  # 带隙 (MEGNet预测)
        'distortion_pred': 'distortion_pred',  # 结构畸变 (q6参数)
        'ionization_energy': 'ionization_energy',  # 电离能 (准确)
    }

    # 任务权重（基于物理意义）
    TASK_WEIGHTS = {
        'band_gap_pred': 1.0,  # 带隙 - 最关键
        'distortion_pred': 0.7,  # 结构畸变
        'ionization_energy': 0.75,  # 电离能
    }

    def __init__(self, task_names=None):
        """
        初始化标签处理器
        task_names: 要预测的任务列表，例如 ['band_gap_pred', 'distortion_pred']
                    如果为None，则使用所有3个任务
        """
        if task_names is None:
            # 默认使用所有3个任务
            self.task_names = list(self.TASK_COLUMNS.keys())
        else:
            self.task_names = task_names

        self.column_names = [self.TASK_COLUMNS[t] for t in self.task_names if t in self.TASK_COLUMNS]
        self.scalers = {}
        self.stats = {}
        self.valid_mask = None

        # 记录原始列名到任务名的映射
        self.column_to_task = {v: k for k, v in self.TASK_COLUMNS.items() if v in self.column_names}

    def process_labels(self, df):
        """
        处理标签数据

        参数:
            df: 包含标签列的DataFrame (从targets.csv加载)

        返回:
            processed: 处理后的标签信息
        """
        print("\n" + "=" * 60)
        print("处理光催化性能标签")
        print("=" * 60)
        print(f"目标任务: {self.task_names}")
        print(f"对应列名: {self.column_names}")

        processed_labels = {}
        n_samples = len(df)

        # 检查所有需要的列是否存在
        missing_cols = [col for col in self.column_names if col not in df.columns]
        if missing_cols:
            print(f"⚠️ 警告: 以下列不存在: {missing_cols}")
            # 从任务列表中移除缺失的列
            for col in missing_cols:
                task_name = self.column_to_task.get(col)
                if task_name in self.task_names:
                    self.task_names.remove(task_name)
            self.column_names = [self.TASK_COLUMNS[t] for t in self.task_names]
            print(f"更新后的任务: {self.task_names}")

        if not self.column_names:
            raise ValueError("没有有效的目标列可处理")

        # 创建全局有效掩码的累积
        all_valid_masks = []

        for task_name, col_name in zip(self.task_names, self.column_names):
            if col_name not in df.columns:
                print(f"⚠️ 跳过任务 '{task_name}': 列 '{col_name}' 不存在")
                continue

            y = df[col_name].values.copy()
            valid_mask = ~np.isnan(y) & ~np.isinf(y)
            all_valid_masks.append(valid_mask)

            print(f"\n📊 任务: {task_name}")
            print(f"  对应列: {col_name}")
            print(f"  有效样本数: {valid_mask.sum()}/{n_samples} ({valid_mask.sum() / n_samples * 100:.1f}%)")

            if valid_mask.sum() == 0:
                print(f"  ⚠️ 无有效样本，跳过")
                continue

            # 1. 异常值处理
            y, outlier_info = self._handle_outliers(y, valid_mask)
            if outlier_info['n_outliers'] > 0:
                print(f"  处理异常值: {outlier_info['n_outliers']} 个")

            # 2. 检查分布
            skewness = self._check_skewness(y[valid_mask])
            print(f"  分布偏度: {skewness:.3f}")

            # 3. 归一化
            y_normalized, scaler = self._normalize_labels(y, valid_mask)

            # 保存统计信息
            self.scalers[task_name] = scaler
            self.stats[task_name] = {
                'min': float(y[valid_mask].min()),
                'max': float(y[valid_mask].max()),
                'mean': float(y[valid_mask].mean()),
                'std': float(y[valid_mask].std()),
                'skewness': float(skewness),
                'valid_count': int(valid_mask.sum())
            }

            processed_labels[task_name] = {
                'raw': y,
                'normalized': y_normalized,
                'valid_mask': valid_mask,
                'column': col_name
            }

            print(f"  原始范围: [{y[valid_mask].min():.4f}, {y[valid_mask].max():.4f}]")
            print(f"  归一化范围: [{y_normalized[valid_mask].min():.4f}, {y_normalized[valid_mask].max():.4f}]")

        # 创建全局有效掩码（所有选定任务都有值的样本）
        if all_valid_masks:
            self.valid_mask = np.all(all_valid_masks, axis=0)
            print(f"\n✅ 全局有效样本: {self.valid_mask.sum()}/{n_samples}")

        return processed_labels

    def _handle_outliers(self, y, valid_mask):
        """处理异常值（IQR方法）"""
        y_clean = y.copy()

        q1 = np.percentile(y[valid_mask], 25)
        q3 = np.percentile(y[valid_mask], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 识别异常值
        outliers = (y[valid_mask] < lower_bound) | (y[valid_mask] > upper_bound)
        outlier_indices = np.where(valid_mask)[0][outliers]

        # 截断异常值
        y_clean[valid_mask] = np.clip(y[valid_mask], lower_bound, upper_bound)

        return y_clean, {'n_outliers': len(outlier_indices), 'indices': outlier_indices}

    def _check_skewness(self, y):
        """检查偏度"""
        return float(skew(y))

    def _normalize_labels(self, y, valid_mask):
        """归一化标签（使用RobustScaler对异常值更鲁棒）"""
        scaler = RobustScaler()
        y_normalized = y.copy()

        scaler.fit(y[valid_mask].reshape(-1, 1))
        y_normalized[valid_mask] = scaler.transform(y[valid_mask].reshape(-1, 1)).flatten()

        return y_normalized, scaler

    def inverse_transform(self, normalized_predictions, task_name):
        """将归一化的预测值转换回原始尺度"""
        if task_name not in self.scalers:
            return normalized_predictions

        scaler = self.scalers[task_name]
        return scaler.inverse_transform(normalized_predictions.reshape(-1, 1)).flatten()

    def get_task_weights(self):
        """获取任务权重"""
        weights = []
        for task in self.task_names:
            if task in self.TASK_WEIGHTS:
                weights.append(self.TASK_WEIGHTS[task])
            else:
                weights.append(0.5)
        return weights

    def get_valid_samples(self, graph_data_list, indices):
        """获取有效样本的图数据"""
        valid_data = []
        for idx in indices:
            if self.valid_mask is not None and idx < len(self.valid_mask) and self.valid_mask[idx]:
                if idx < len(graph_data_list):
                    valid_data.append(graph_data_list[idx])
        return valid_data


# ==================== 可解释性输出 ====================

class PhotocatalysisInterpreter:
    """
    光催化性能预测的可解释性输出
    将模型输出转换为材料科学建议
    更新以匹配3个准确计算的目标特征
    """

    def __init__(self, label_processor):
        self.label_processor = label_processor
        self.stats = label_processor.stats if hasattr(label_processor, 'stats') else {}

        # 参考值 - 更新以匹配3个准确计算的目标特征
        self.reference_values = {
            'band_gap_pred': {'ideal': [1.8, 2.5], 'unit': 'eV', 'name': '带隙'},
            'distortion_pred': {'ideal': [0.1, 0.3], 'unit': '', 'name': '结构畸变'},
            'ionization_energy': {'ideal': [5, 10], 'unit': 'eV', 'name': '电离能'}
        }

    def generate_report(self, predictions, structure_info=None):
        """
        生成可解释的光催化性能报告

        参数:
            predictions: dict {task_name: predicted_value}
            structure_info: dict 包含结构信息（可选）

        返回:
            report: 格式化的报告字符串
        """
        # 转换回原始尺度
        raw_predictions = {}
        for task_name, pred_norm in predictions.items():
            if task_name in self.label_processor.scalers:
                try:
                    raw = self.label_processor.inverse_transform(
                        np.array([pred_norm]), task_name
                    )[0]
                    raw_predictions[task_name] = raw
                except:
                    raw_predictions[task_name] = pred_norm
            else:
                raw_predictions[task_name] = pred_norm

        report = []
        report.append("=" * 70)
        report.append("光催化性能预测分析报告")
        report.append("=" * 70)

        # 材料信息
        if structure_info:
            report.append(f"\n📌 材料: {structure_info.get('formula', 'Unknown')}")
            report.append(f"   空间群: {structure_info.get('spacegroup', 'Unknown')}")
            report.append(f"   晶系: {structure_info.get('crystal_system', 'Unknown')}")

        # 电子结构特征
        report.append("\n【电子结构特征】")
        self._add_metric(report, raw_predictions, 'band_gap_pred', '带隙')
        self._add_metric(report, raw_predictions, 'ionization_energy', '电离能')

        # 结构特征
        report.append("\n【结构特征】")
        self._add_metric(report, raw_predictions, 'distortion_pred', '结构畸变')

        # 性能限制因素分析
        report.append("\n【性能限制因素】")
        self._analyze_limitations(report, raw_predictions)

        # 改进建议
        report.append("\n【材料优化建议】")
        self._generate_suggestions(report, raw_predictions, structure_info)

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def _add_metric(self, report, predictions, key, name):
        """添加指标到报告"""
        if key in predictions:
            value = predictions[key]
            ref_info = self.reference_values.get(key, {})
            unit = ref_info.get('unit', '')

            # 评估等级
            grade = self._evaluate_metric(key, value)

            report.append(f"  {name}: {value:.3f} {unit}  [{grade}]")

    def _evaluate_metric(self, key, value):
        """评估指标等级"""
        ref = self.reference_values.get(key)
        if not ref or 'ideal' not in ref:
            return "——"

        ideal_min, ideal_max = ref['ideal']

        if ideal_min <= value <= ideal_max:
            return "✓✓ 优秀"
        elif value < ideal_min * 0.7:
            return "⬇ 偏低"
        elif value > ideal_max * 1.3:
            return "⬆ 偏高"
        elif value < ideal_min:
            return "○ 偏低"
        elif value > ideal_max:
            return "○ 偏高"
        else:
            return "✓ 良好"

    def _analyze_limitations(self, report, predictions):
        """分析性能限制因素"""
        limitations = []

        # 带隙分析
        bg = predictions.get('band_gap_pred')
        if bg:
            if bg > 3.0:
                limitations.append("• 带隙过大 (>3.0 eV)，可见光吸收效率低")
            elif bg < 1.5:
                limitations.append("• 带隙过小 (<1.5 eV)，光生载流子易复合")

        # 结构畸变分析
        dist = predictions.get('distortion_pred')
        if dist:
            if dist > 0.5:
                limitations.append("• 结构畸变较大，可能影响载流子迁移率")
            elif dist < 0.05:
                limitations.append("• 结构过于规整，可能缺乏活性位点")

        # 电离能分析
        ie = predictions.get('ionization_energy')
        if ie:
            if ie > 12:
                limitations.append("• 电离能过高，光生电子注入效率可能较低")
            elif ie < 5:
                limitations.append("• 电离能过低，材料可能不稳定")

        if not limitations:
            limitations.append("• 未检测到明显性能限制因素")

        for limitation in limitations:
            report.append(f"  {limitation}")

    def _generate_suggestions(self, report, predictions, structure_info):
        """生成改进建议"""
        suggestions = []

        # 基于带隙的掺杂建议
        bg = predictions.get('band_gap_pred')
        if bg:
            if bg > 3.0:
                suggestions.append("• 考虑掺杂N、S等阴离子减小带隙")
            elif bg < 1.5:
                suggestions.append("• 考虑掺杂金属离子引入中间能级，抑制复合")

        # 基于结构畸变的建议
        dist = predictions.get('distortion_pred')
        if dist and dist > 0.5:
            suggestions.append("• 考虑通过元素掺杂稳定晶体结构")
        elif dist and dist < 0.05:
            suggestions.append("• 考虑引入缺陷工程增加结构活性")

        # 基于电离能的建议
        ie = predictions.get('ionization_energy')
        if ie and ie > 12:
            suggestions.append("• 考虑表面修饰降低电离能，改善界面电荷注入")

        if not suggestions:
            suggestions.append("• 当前性能良好，可考虑进一步提高比表面积和结晶度")

        for suggestion in suggestions:
            report.append(f"  {suggestion}")


# ==================== 训练器 ====================

class PhotocatalysisTrainer:
    """
    光催化GAT模型训练器
    支持多任务学习和物理约束
    """

    def __init__(self, model, label_processor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.label_processor = label_processor
        self.device = device
        self.physical_constraints = PhysicalConstraints(device=device)
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    def train(self, train_loader, val_loader, epochs=200, lr=0.001, weight_decay=1e-5,
              constraint_weight=0.1, patience=20):
        """
        训练模型
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )

        task_weights = self.label_processor.get_task_weights()
        task_weights = torch.tensor(task_weights, device=self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print("\n" + "=" * 60)
        print("开始训练光催化GAT模型")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"任务数: {len(self.label_processor.task_names)}")
        print(f"任务权重: {task_weights.cpu().numpy()}")
        print(f"物理约束权重: {constraint_weight}")
        print(f"初始学习率: {lr}")

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_batches = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # 前向传播
                predictions = self.model(batch)

                # 计算损失
                loss = self._compute_loss(predictions, batch.y, task_weights)

                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 检测到 NaN/Inf 损失，跳过此batch")
                    continue

                # 添加物理约束损失
                if constraint_weight > 0:
                    constraint_loss = self._compute_constraint_loss(predictions, batch)
                    if not (torch.isnan(constraint_loss) or torch.isinf(constraint_loss)):
                        loss += constraint_weight * constraint_loss

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            if train_batches > 0:
                avg_train_loss = train_loss / train_batches
            else:
                avg_train_loss = float('inf')

            self.history['train_loss'].append(avg_train_loss)

            # 验证阶段
            val_loss, val_mae = self.evaluate(val_loader, task_weights)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)

            scheduler.step(val_loss)

            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val MAE: {val_mae:.4f} | "
                      f"LR: {current_lr:.6f}")

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停于 epoch {epoch + 1}")
                    break

        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ 最佳验证损失: {best_val_loss:.4f}")

        return self.history

    def _compute_loss(self, predictions, targets, task_weights):
        """计算多任务损失"""
        loss = 0
        batch_size = predictions.shape[0]
        num_tasks = predictions.shape[1]

        # 处理 targets 形状
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 1:
                # 尝试重塑为 [batch_size, num_tasks]
                if len(targets) == batch_size * num_tasks:
                    targets_reshaped = targets.view(batch_size, num_tasks)
                else:
                    targets_reshaped = targets.view(batch_size, -1)
            elif targets.dim() == 2:
                targets_reshaped = targets
            else:
                targets_reshaped = targets.squeeze()
                if targets_reshaped.dim() == 1:
                    targets_reshaped = targets_reshaped.unsqueeze(1)
        else:
            targets_reshaped = targets

        # 确保 targets_reshaped 是2维
        if targets_reshaped.dim() == 1:
            targets_reshaped = targets_reshaped.unsqueeze(1)

        valid_tasks = 0
        for i in range(min(num_tasks, targets_reshaped.shape[1])):
            task_pred = predictions[:, i]
            task_target = targets_reshaped[:, i]

            # 只计算有效样本的损失
            valid_mask = ~torch.isnan(task_target)
            if valid_mask.sum() > 0:
                # 使用 Huber Loss，对异常值更鲁棒
                task_loss = F.huber_loss(
                    task_pred[valid_mask],
                    task_target[valid_mask],
                    delta=1.0,
                    reduction='mean'
                )

                # 应用任务权重
                if isinstance(task_weights, (list, torch.Tensor)):
                    weight = task_weights[i] if i < len(task_weights) else 1.0
                    weighted_loss = weight * task_loss
                else:
                    weighted_loss = task_loss

                loss += weighted_loss
                valid_tasks += 1

        # 返回平均损失
        if valid_tasks > 0:
            return loss / valid_tasks
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _compute_constraint_loss(self, predictions, batch):
        """计算物理约束损失 - 简化版"""
        constraint_loss = 0
        task_names = self.label_processor.task_names

        # 创建任务索引映射
        task_indices = {name: i for i, name in enumerate(task_names)}

        # 1. 带隙与电离能的相关性约束（物理合理的带隙通常与电离能相关）
        if ('band_gap_pred' in task_indices and 'ionization_energy' in task_indices):
            bg_idx = task_indices['band_gap_pred']
            ie_idx = task_indices['ionization_energy']

            # 提取预测值
            bg_pred = predictions[:, bg_idx]
            ie_pred = predictions[:, ie_idx]

            # 物理约束：带隙通常与电离能正相关（简化模型）
            # 这里使用一个非常宽松的约束
            bg_norm = (bg_pred - bg_pred.mean()) / (bg_pred.std() + 1e-8)
            ie_norm = (ie_pred - ie_pred.mean()) / (ie_pred.std() + 1e-8)

            # 惩罚负相关（如果相关系数太负）
            correlation = (bg_norm * ie_norm).mean()
            constraint_loss += 0.005 * torch.relu(-correlation - 0.5)

        # 2. 带隙与结构畸变的关系（通常结构畸变影响带隙）
        if ('band_gap_pred' in task_indices and 'distortion_pred' in task_indices):
            bg_idx = task_indices['band_gap_pred']
            dist_idx = task_indices['distortion_pred']

            bg_pred = predictions[:, bg_idx]
            dist_pred = predictions[:, dist_idx]

            # 结构畸变过大通常导致带隙变化（正负都可能，但不应极端）
            # 这里惩罚极端畸变与极端带隙的组合
            extreme_dist = torch.abs(dist_pred) > 0.8
            extreme_bg = (bg_pred > 3.0) | (bg_pred < 1.0)
            unreasonable = extreme_dist & extreme_bg
            constraint_loss += 0.01 * torch.mean(unreasonable.float() * torch.abs(bg_pred - 2.0))

        return constraint_loss

    def evaluate(self, loader, task_weights=None):
        """评估模型"""
        if task_weights is None:
            task_weights = self.label_processor.get_task_weights()
            task_weights = torch.tensor(task_weights, device=self.device)

        self.model.eval()
        total_loss = 0
        total_mae = 0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                predictions = self.model(batch)

                batch_size = predictions.shape[0]
                num_tasks = predictions.shape[1]

                # 重塑 targets
                try:
                    if batch.y.dim() == 1 and len(batch.y) == batch_size * num_tasks:
                        targets_reshaped = batch.y.view(batch_size, num_tasks)
                    elif batch.y.dim() == 1:
                        targets_reshaped = batch.y.view(batch_size, -1)
                    else:
                        targets_reshaped = batch.y
                except:
                    targets_reshaped = batch.y

                if targets_reshaped.dim() == 1:
                    targets_reshaped = targets_reshaped.unsqueeze(1)

                # 计算损失
                batch_loss = 0
                batch_mae = 0
                valid_tasks = 0

                for i in range(min(num_tasks, targets_reshaped.shape[1])):
                    task_pred = predictions[:, i]
                    task_target = targets_reshaped[:, i]

                    valid_mask = ~torch.isnan(task_target)
                    if valid_mask.sum() > 0:
                        # 使用 Huber Loss
                        task_loss = F.huber_loss(
                            task_pred[valid_mask],
                            task_target[valid_mask],
                            delta=1.0
                        )

                        weight = task_weights[i] if i < len(task_weights) else 1.0
                        batch_loss += weight * task_loss

                        task_mae = F.l1_loss(task_pred[valid_mask], task_target[valid_mask])
                        batch_mae += task_mae
                        valid_tasks += 1

                if valid_tasks > 0:
                    total_loss += batch_loss.item() / valid_tasks
                    total_mae += batch_mae.item() / valid_tasks
                    n_batches += 1

        if n_batches > 0:
            avg_loss = total_loss / n_batches
            avg_mae = total_mae / n_batches
        else:
            avg_loss = float('inf')
            avg_mae = float('inf')

        return avg_loss, avg_mae

    def predict(self, loader):
        """预测"""
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                predictions = self.model(batch)
                all_predictions.append(predictions.cpu())

        return torch.cat(all_predictions, dim=0)

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'task_names': self.label_processor.task_names
        }, path)
        print(f"✅ 模型已保存到: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f"✅ 模型已从 {path} 加载")


# ==================== 主流水线 ====================

class PhotocatalysisPipeline:
    """
    光催化性能预测完整流水线
    整合数据处理、模型训练、可解释性输出
    与特征工程的准确特征输出完全对应
    所有输出保存到 runs/ 目录
    """

    def __init__(self, task_names=None, megnet_model_path=None, device=None, run_name=None):
        """
        初始化流水线

        参数:
            task_names: 要预测的任务列表，默认为None（使用所有3个任务）
            megnet_model_path: MEGNet模型路径（可选）
            device: 计算设备 ('cuda' 或 'cpu')
            run_name: 运行名称，用于创建 runs/ 下的子目录
        """
        # 默认使用3个准确计算的任务
        default_tasks = [
            'band_gap_pred',
            'distortion_pred',
            'ionization_energy'
        ]
        self.task_names = task_names or default_tasks
        self.megnet_model_path = megnet_model_path
        self.model = None
        self.label_processor = None
        self.interpreter = None
        self.trainer = None
        self.graph_data_list = []
        self.feature_df = None
        self.targets_df = None

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 特征维度（与特征工程一致 - 已移除经验公式）
        self.node_dim = 12
        self.edge_dim = 8  # 移除了键级估计
        self.global_dim = None  # 将在加载数据时确定

        # 设置运行目录
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("runs", self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 运行目录: {self.output_dir}")

    def load_data(self, graph_data_path, targets_path=None):
        """
        加载图数据和目标值

        参数:
            graph_data_path: 图数据文件路径 (.pt)
            targets_path: 目标值文件路径 (.csv) - 可选，如果图数据中已有y值可不提供
        """
        print("\n" + "=" * 60)
        print("加载数据")
        print("=" * 60)

        # 1. 加载图数据
        if not os.path.exists(graph_data_path):
            # 尝试从features目录加载
            alt_path = os.path.join("features", os.path.basename(graph_data_path))
            if os.path.exists(alt_path):
                graph_data_path = alt_path
            else:
                raise FileNotFoundError(f"图数据文件不存在: {graph_data_path}")

        self.graph_data_list = torch.load(graph_data_path)
        print(f"✅ 已加载图数据: {len(self.graph_data_list)} 个")

        # 预处理图数据
        for i, data in enumerate(self.graph_data_list):
            # 确保节点特征是float类型
            if hasattr(data, 'x') and data.x is not None:
                data.x = data.x.float()

            # 确保边特征是float类型
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr.float()

            # 处理全局特征
            if hasattr(data, 'global_features') and data.global_features is not None:
                if data.global_features.dim() == 3:
                    data.global_features = data.global_features.squeeze(1)
                if data.global_features.dim() == 1:
                    data.global_features = data.global_features.unsqueeze(0)
                data.global_features = data.global_features.float()

        # 确定全局特征维度
        if self.graph_data_list and len(self.graph_data_list) > 0:
            sample_data = self.graph_data_list[0]
            if hasattr(sample_data, 'global_features') and sample_data.global_features is not None:
                if sample_data.global_features.dim() > 1:
                    self.global_dim = sample_data.global_features.shape[-1]
                else:
                    self.global_dim = 1
            else:
                self.global_dim = 32  # 默认值
            print(f"   全局特征维度: {self.global_dim}")

        # 2. 加载目标值
        if targets_path and os.path.exists(targets_path):
            self.targets_df = pd.read_csv(targets_path)
            print(f"✅ 已加载目标值: {self.targets_df.shape}")
            print(f"   样本数: {len(self.targets_df)}")
            print(f"   目标列: {[c for c in self.targets_df.columns if c != 'filename']}")

            # 将目标值添加到图数据中
            self._add_targets_to_graph_data()
        else:
            # 检查图数据中是否已有y值
            has_y = all(hasattr(data, 'y') and data.y is not None for data in self.graph_data_list[:10])
            if has_y:
                print("✅ 图数据中已包含y值")
            else:
                print("⚠️ 未找到目标值，将使用随机值（仅用于测试）")
                self._create_dummy_targets()

        return self.graph_data_list, self.targets_df

    def _add_targets_to_graph_data(self):
        """将目标值添加到图数据中"""
        if self.targets_df is None or 'filename' not in self.targets_df.columns:
            print("⚠️ 无法添加目标值：缺少文件名信息")
            return

        # 创建文件名到索引的映射
        filename_to_idx = {}
        for i, data in enumerate(self.graph_data_list):
            if hasattr(data, 'filename'):
                filename_to_idx[data.filename] = i

        matched_count = 0
        for _, row in self.targets_df.iterrows():
            filename = row['filename']
            if filename in filename_to_idx:
                idx = filename_to_idx[filename]
                # 创建3维目标向量
                target_vector = []
                for task in self.task_names:
                    if task in row and pd.notna(row[task]):
                        target_vector.append(float(row[task]))
                    else:
                        target_vector.append(float('nan'))

                self.graph_data_list[idx].y = torch.tensor(target_vector, dtype=torch.float)
                matched_count += 1

        print(f"✅ 已为 {matched_count} 个图数据添加目标值")

    def _create_dummy_targets(self):
        """创建模拟目标值（仅用于测试）"""
        for i, data in enumerate(self.graph_data_list):
            # 创建随机目标值
            dummy_y = torch.randn(len(self.task_names))
            data.y = dummy_y
        print("⚠️ 已创建模拟目标值用于测试")

    def prepare_labels(self):
        """准备标签"""
        if self.targets_df is None:
            # 如果图数据中已有y值，直接使用
            if all(hasattr(data, 'y') and data.y is not None for data in self.graph_data_list[:10]):
                print("✅ 使用图数据中已有的y值")
                return None

        # 创建标签处理器
        self.label_processor = PhotocatalysisLabelProcessor(self.task_names)

        if self.targets_df is not None:
            processed_labels = self.label_processor.process_labels(self.targets_df)
            print(f"\n✅ 标签处理完成")
            return processed_labels
        else:
            print("⚠️ 无目标值数据，跳过标签处理")
            return None

    def prepare_data_loaders(self, batch_size=32, val_split=0.15, test_split=0.15, random_seed=42):
        """准备数据加载器"""
        # 筛选有效样本（有y值的样本）
        valid_data = []
        for data in self.graph_data_list:
            if hasattr(data, 'y') and data.y is not None:
                # 检查是否有有效值（非全NaN）
                if not torch.isnan(data.y).all():
                    valid_data.append(data)

        if not valid_data:
            print("\n⚠️ 没有有效样本，无法创建数据加载器")
            return None, None, None

        print(f"\n有效样本数: {len(valid_data)}/{len(self.graph_data_list)}")

        # 划分数据集
        n_total = len(valid_data)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val

        indices = list(range(n_total))
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test + n_val]
        train_indices = indices[n_test + n_val:]

        train_data = [valid_data[i] for i in train_indices]
        val_data = [valid_data[i] for i in val_indices]
        test_data = [valid_data[i] for i in test_indices]

        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size) if test_data else None

        print(f"\n📊 数据集划分:")
        print(f"   训练集: {len(train_data)} 样本")
        print(f"   验证集: {len(val_data)} 样本")
        print(f"   测试集: {len(test_data)} 样本")

        return train_loader, val_loader, test_loader

    def build_model(self, hidden_dim=64, num_heads=4):
        """构建GAT模型"""
        if not self.graph_data_list:
            raise ValueError("没有图数据，请先加载数据")

        # 从第一个图数据获取维度
        sample_data = self.graph_data_list[0]

        # 自动获取输入维度
        node_dim = sample_data.x.size(1) if hasattr(sample_data, 'x') and sample_data.x is not None else self.node_dim
        edge_dim = sample_data.edge_attr.size(1) if hasattr(sample_data,
                                                            'edge_attr') and sample_data.edge_attr is not None else self.edge_dim

        # 获取全局特征维度
        if hasattr(sample_data, 'global_features') and sample_data.global_features is not None:
            if sample_data.global_features.dim() > 1:
                global_dim = sample_data.global_features.size(-1)
            else:
                global_dim = 1
        else:
            global_dim = self.global_dim or 32

        print(f"\n✅ 模型输入维度确认:")
        print(f"   节点特征维度: {node_dim}")
        print(f"   边特征维度: {edge_dim}")
        print(f"   全局特征维度: {global_dim}")
        print(f"   输出维度: {len(self.task_names)}")

        self.model = PhotocatalysisGAT(
            node_in_dim=node_dim,
            edge_in_dim=edge_dim,
            global_in_dim=global_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            output_dim=len(self.task_names)
        ).to(self.device)

        print(f"\n✅ 模型构建完成")
        print(f"   运行设备: {self.device}")

        return self.model

    def train(self, train_loader, val_loader, epochs=200, lr=0.0005,
              constraint_weight=0.01, patience=80):
        """训练模型"""
        # 初始化训练器
        self.trainer = PhotocatalysisTrainer(
            self.model,
            self.label_processor or PhotocatalysisLabelProcessor(self.task_names),
            device=self.device
        )

        # 训练
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            constraint_weight=constraint_weight,
            patience=patience
        )

        return history

    def evaluate(self, test_loader):
        """评估模型"""
        if test_loader is None:
            print("\n⚠️ 无测试集，跳过评估")
            return None, None

        if self.trainer is None:
            self.trainer = PhotocatalysisTrainer(
                self.model,
                self.label_processor or PhotocatalysisLabelProcessor(self.task_names),
                device=self.device
            )

        task_weights = self.label_processor.get_task_weights() if self.label_processor else [1.0] * len(self.task_names)
        task_weights = torch.tensor(task_weights, device=self.device)
        test_loss, test_mae = self.trainer.evaluate(test_loader, task_weights)

        print("\n" + "=" * 60)
        print("测试集评估结果")
        print("=" * 60)
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试MAE: {test_mae:.4f}")

        # 逐任务评估
        try:
            predictions = self.trainer.predict(test_loader)

            # 收集所有目标值
            all_targets = []
            for batch in test_loader:
                all_targets.append(batch.y)
            targets = torch.cat(all_targets, dim=0)

            # 确保 targets 是正确形状
            if targets.dim() == 1:
                try:
                    targets = targets.view(-1, len(self.task_names))
                except:
                    targets = targets.unsqueeze(1)

            print("\n📊 各任务性能:")
            for i, task_name in enumerate(self.task_names):
                if i >= predictions.shape[1] or i >= targets.shape[1]:
                    continue

                task_pred = predictions[:, i].cpu().numpy()
                task_target = targets[:, i].cpu().numpy()
                valid_mask = ~np.isnan(task_target)

                if valid_mask.sum() > 0:
                    mae = mean_absolute_error(task_target[valid_mask], task_pred[valid_mask])
                    r2 = r2_score(task_target[valid_mask], task_pred[valid_mask])

                    # 转换回原始尺度（如果有scaler）
                    if self.label_processor and task_name in self.label_processor.scalers:
                        try:
                            pred_raw = self.label_processor.inverse_transform(task_pred, task_name)
                            target_raw = self.label_processor.inverse_transform(task_target, task_name)
                            mae_raw = mean_absolute_error(target_raw[valid_mask], pred_raw[valid_mask])
                        except:
                            pred_raw = task_pred
                            target_raw = task_target
                            mae_raw = mae
                    else:
                        mae_raw = mae

                    print(f"  {task_name}:")
                    print(f"    MAE (归一化): {mae:.4f}")
                    print(f"    MAE (原始): {mae_raw:.4f}")
                    print(f"    R²: {r2:.4f}")
        except Exception as e:
            print(f"  详细评估时出错: {e}")

        return test_loss, test_mae

    def predict_and_interpret(self, data_loader, structure_info=None):
        """预测并生成可解释报告"""
        if self.trainer is None:
            self.trainer = PhotocatalysisTrainer(
                self.model,
                self.label_processor or PhotocatalysisLabelProcessor(self.task_names),
                device=self.device
            )

        predictions = self.trainer.predict(data_loader)

        # 创建解释器（如果没有）
        if self.interpreter is None and self.label_processor:
            self.interpreter = PhotocatalysisInterpreter(self.label_processor)

        reports = []
        for i in range(len(predictions)):
            pred_dict = {
                task_name: predictions[i, j].item()
                for j, task_name in enumerate(self.task_names)
                if j < predictions.shape[1]
            }

            if self.interpreter:
                report = self.interpreter.generate_report(pred_dict, structure_info)
            else:
                # 简单报告
                report = f"预测结果 (样本 {i}):\n"
                for task, value in pred_dict.items():
                    report += f"  {task}: {value:.4f}\n"
            reports.append(report)

        return reports

    def save(self, model_path=None, pipeline_path=None):
        """保存整个流水线到 runs/ 目录"""
        # 设置默认路径
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'gat_model.pt')
        if pipeline_path is None:
            pipeline_path = os.path.join(self.output_dir, 'pipeline.pt')

        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)

        if self.trainer:
            self.trainer.save_model(model_path)

        pipeline_data = {
            'task_names': self.task_names,
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'global_dim': self.global_dim,
            'label_processor': self.label_processor,
            'device': str(self.device),
            'run_name': self.run_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        torch.save(pipeline_data, pipeline_path)
        print(f"✅ 流水线已保存到: {pipeline_path}")

    def load(self, model_path, pipeline_path):
        """加载流水线"""
        pipeline_data = torch.load(pipeline_path, map_location=self.device)
        self.task_names = pipeline_data['task_names']
        self.node_dim = pipeline_data['node_dim']
        self.edge_dim = pipeline_data['edge_dim']
        self.global_dim = pipeline_data['global_dim']
        self.label_processor = pipeline_data['label_processor']
        self.run_name = pipeline_data.get('run_name', self.run_name)

        self.build_model()
        self.trainer = PhotocatalysisTrainer(
            self.model,
            self.label_processor,
            device=self.device
        )
        self.trainer.load_model(model_path)
        self.interpreter = PhotocatalysisInterpreter(self.label_processor)

        print(f"✅ 流水线已加载")


# ==================== 主程序示例 ====================

def plot_training_history(history, save_dir):
    """绘制训练历史曲线"""
    if not history or 'train_loss' not in history or not history['train_loss']:
        print("⚠️ 没有训练历史数据可绘制")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制 MAE 曲线
    plt.subplot(1, 2, 2)
    if 'val_mae' in history and history['val_mae']:
        plt.plot(epochs, history['val_mae'], 'g-', label='Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 保存到 runs/ 目录
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path)
    print(f"📈 训练曲线图已保存为 '{save_path}'")
    plt.show()


def main():
    """主程序示例 - 与特征工程输出完美对应"""
    print("=" * 70)
    print("光催化性能预测GAT模型训练流水线")
    print("(与特征工程准确特征输出完全对应)")
    print("=" * 70)

    # 生成运行名称
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # 文件路径配置
    config = {
        # 输入文件 (从features/目录读取)
        'graph_data_path': os.path.join("features", 'photocatalysis_graph.pt'),
        'targets_path': os.path.join("features", 'photocatalysis_targets.csv'),
        'features_path': os.path.join("features", 'photocatalysis_features.csv'),

        # 输出文件 (保存到runs/目录)
        'model_save_path': os.path.join(output_dir, 'gat_model.pt'),
        'pipeline_save_path': os.path.join(output_dir, 'pipeline.pt'),

        # 要预测的任务 - 3个准确计算的目标特征
        'task_names': [
            'band_gap_pred',      # 带隙 (MEGNet预测)
            'distortion_pred',    # 结构畸变 (q6参数)
            'ionization_energy',  # 电离能 (准确)
        ],

        # 模型参数
        'hidden_dim': 64,
        'num_heads': 4,

        # 训练参数
        'batch_size': 16,
        'epochs': 150,
        'learning_rate': 0.0005,
        'constraint_weight': 0.01,
        'patience': 20,
        'val_split': 0.15,
        'test_split': 0.15,

        # 设备
        'device': 'cpu',  # 如果没有GPU，使用CPU

        # 运行名称
        'run_name': run_name
    }

    print(f"\n📁 运行目录: {output_dir}")

    try:
        # 检查文件是否存在
        if not os.path.exists(config['graph_data_path']):
            print(f"❌ 错误: 图数据文件 '{config['graph_data_path']}' 不存在")
            print("请先运行特征工程脚本生成数据文件")
            return

        # 1. 初始化流水线
        pipeline = PhotocatalysisPipeline(
            task_names=config['task_names'],
            megnet_model_path=None,
            device=config['device'],
            run_name=config['run_name']
        )

        # 2. 加载数据
        pipeline.load_data(
            graph_data_path=config['graph_data_path'],
            targets_path=config['targets_path']
        )

        # 3. 准备标签
        pipeline.prepare_labels()

        # 4. 构建模型
        pipeline.build_model(
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads']
        )

        # 5. 准备数据加载器
        train_loader, val_loader, test_loader = pipeline.prepare_data_loaders(
            batch_size=config['batch_size'],
            val_split=config['val_split'],
            test_split=config['test_split']
        )

        if train_loader is None:
            print("❌ 无法创建数据加载器，退出")
            return

        # 6. 训练模型
        history = pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            lr=config['learning_rate'],
            constraint_weight=config['constraint_weight'],
            patience=config['patience']
        )

        # 7. 绘制训练曲线
        if history:
            plot_training_history(history, output_dir)

        # 8. 评估模型
        pipeline.evaluate(test_loader)

        # 9. 保存模型
        pipeline.save(
            model_path=config['model_save_path'],
            pipeline_path=config['pipeline_save_path']
        )

        # 10. 示例预测和解释
        if test_loader and len(test_loader.dataset) > 0:
            print("\n" + "=" * 70)
            print("预测结果解释示例")
            print("=" * 70)

            # 使用第一个测试样本
            sample_loader = DataLoader([test_loader.dataset[0]], batch_size=1)
            sample_data = test_loader.dataset[0]

            # 获取结构信息（如果有）
            structure_info = {
                'formula': getattr(sample_data, 'formula', 'Unknown'),
                'spacegroup': getattr(sample_data, 'spacegroup', 'Unknown'),
                'crystal_system': getattr(sample_data, 'crystal_system', 'Unknown')
            }

            reports = pipeline.predict_and_interpret(sample_loader, structure_info)
            if reports:
                print(reports[0])

        print("\n✅ 训练完成！")
        print(f"📁 所有输出文件保存在: {output_dir}/")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
光催化GAT模型预训练模块 V2.1
修复：Loss Components显示 + 增强DOS预测精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from scipy.interpolate import interp1d
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import glob
import warnings

warnings.filterwarnings('ignore')


# ==================== DOS专用数据封装 ====================

class DOSData(Data):
    """态密度预训练数据封装类"""

    def __init__(self, x=None, edge_index=None, edge_attr=None, pos=None,
                 dos_energy=None, dos_total=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs)
        self.dos_energy = dos_energy
        self.dos_total = dos_total

    def __inc__(self, key, value, *args, **kwargs):
        if key in ['dos_energy', 'dos_total']:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['dos_energy', 'dos_total']:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


# ==================== DOS曲线处理工具 ====================

class DOSProcessor:
    """态密度曲线处理器"""

    def __init__(self, energy_range=(-8, 8), num_points=128, smooth_sigma=0.5):
        self.energy_min, self.energy_max = energy_range
        self.num_points = num_points
        self.smooth_sigma = smooth_sigma
        self.energy_axis = np.linspace(energy_range[0], energy_range[1], num_points)

    def process_dos_curve(self, energy, dos, normalize=True):
        energy = np.array(energy).flatten()
        dos = np.array(dos).flatten()

        try:
            f = interp1d(energy, dos, kind='cubic', bounds_error=False, fill_value=0.0)
            dos_interp = f(self.energy_axis)
        except Exception:
            f = interp1d(energy, dos, kind='linear', bounds_error=False, fill_value=0.0)
            dos_interp = f(self.energy_axis)

        if self.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            dos_interp = gaussian_filter1d(dos_interp, self.smooth_sigma, mode='constant')

        if normalize:
            area = np.trapz(np.abs(dos_interp), self.energy_axis)
            if area > 0:
                dos_interp = dos_interp / area

        return dos_interp.astype(np.float32)


# ==================== 可视化工具（修复版）====================

class DOSVisualizer:
    """态密度曲线可视化工具"""

    @staticmethod
    def plot_dos_comparison(energy_axis, target_dos, pred_dos,
                            title="DOS Prediction", save_path=None):
        """绘制DOS预测对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 左图：整体对比
        ax = axes[0]
        ax.plot(energy_axis, target_dos, 'b-', label='Target', linewidth=2, alpha=0.8)
        ax.plot(energy_axis, pred_dos, 'r--', label='Prediction', linewidth=2, alpha=0.8)
        ax.axvline(x=0, color='k', linestyle=':', alpha=0.5, label='Fermi Level')
        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.set_ylabel('DOS (arb. units)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 中图：误差
        ax = axes[1]
        error = pred_dos - target_dos
        ax.fill_between(energy_axis, 0, error, alpha=0.3, color='red')
        ax.plot(energy_axis, error, 'k-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('Prediction Error', fontsize=14)
        ax.grid(True, alpha=0.3)

        # 右图：散点图（相关性）
        ax = axes[2]
        ax.scatter(target_dos, pred_dos, alpha=0.5, s=10)
        min_val = min(target_dos.min(), pred_dos.min())
        max_val = max(target_dos.max(), pred_dos.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
        ax.set_xlabel('Target DOS', fontsize=12)
        ax.set_ylabel('Predicted DOS', fontsize=12)
        ax.set_title('Correlation', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 计算R²
        ss_res = np.sum((target_dos - pred_dos) ** 2)
        ss_tot = np.sum((target_dos - np.mean(target_dos)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  💾 图片已保存: {save_path}")
        plt.show()

        return fig

    @staticmethod
    def plot_training_history(history, save_path='dos_training_history.png'):
        """绘制训练历史（修复版）"""
        if not history.get('epochs'):
            print("⚠️ 没有训练历史数据")
            return None

        epochs = history['epochs']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 总损失
        ax = axes[0, 0]
        if history.get('train_total'):
            ax.plot(epochs, history['train_total'], 'b-', label='Train', linewidth=2)
        if history.get('val_total'):
            ax.plot(epochs, history['val_total'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 对数坐标更容易看趋势

        # 2. 损失分量（修复键名）
        ax = axes[0, 1]
        # 注意：历史记录中的键名是 'train_mse', 'train_grad', 'train_spectral' 等
        loss_components = ['mse', 'grad', 'spectral', 'integral']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for comp, color in zip(loss_components, colors):
            key = f'train_{comp}'
            if key in history and history[key]:
                ax.plot(epochs, history[key], label=comp.capitalize(),
                       color=color, alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 3. MAE对比
        ax = axes[1, 0]
        if history.get('train_mae'):
            ax.plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
        if history.get('val_mae'):
            ax.plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('DOS MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 学习率
        ax = axes[1, 1]
        if history.get('lr'):
            ax.plot(epochs, history['lr'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 训练历史图已保存: {save_path}")
        plt.show()

        return fig

class DOSPretrainGAT(nn.Module):
    """DOS预训练模型 V3.0 — 可学习边编码 + 跳跃知识聚合"""

    def __init__(self, base_gat: nn.Module,
                 dos_points: int = 128,
                 hidden_dim: int = 64,
                 energy_range: Tuple[float, float] = (-8, 8)):
        super().__init__()

        self.base_encoder = base_gat.float()
        self.dos_points = dos_points
        self.hidden_dim = hidden_dim
        self.energy_min, self.energy_max = energy_range

        # ===== 能量位置编码 =====
        self.energy_pos_encoding = self._create_energy_encoding()

        # ===== 可学习边编码器 =====
        # 从 base_gat 获取原始边特征维度
        edge_in_dim = self.base_encoder.gat1.edge_dim  # GATv2Conv 的 edge_dim 参数
        self.edge_encoder = LearnableEdgeEncoder(edge_in_dim)

        # ===== 增强的1D卷积解码器 =====
        combined_dim = hidden_dim * 2

        self.dos_conv_decoder = nn.Sequential(
            # Block 1
            nn.Conv1d(combined_dim, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Residual Block 1
            ResidualBlock1D(128, 128, kernel_size=7),

            # Block 3
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Residual Block 2
            ResidualBlock1D(128, 128, kernel_size=7),

            # Block 4: 降维
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 5: 输出
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Softplus()
        )

    def _create_energy_encoding(self):
        energy_axis = torch.linspace(self.energy_min, self.energy_max, self.dos_points)
        energy_norm = (energy_axis - self.energy_min) / (self.energy_max - self.energy_min)

        encoding = torch.zeros(self.dos_points, self.hidden_dim)
        for i in range(0, self.hidden_dim, 2):
            freq = 10000 ** (2 * i / self.hidden_dim)
            encoding[:, i] = torch.sin(energy_norm * freq)
            if i + 1 < self.hidden_dim:
                encoding[:, i + 1] = torch.cos(energy_norm * freq)

        return nn.Parameter(encoding, requires_grad=False)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = x.float()
        # 防止模型不关注键长：引入可学习边编码MLP（RBF（d））
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr.float())

        # 防止注意力极化于Ti-O键：训练时进行随机edge drop out正则：
        if self.training and edge_attr is not None:
            from torch_geometric.utils import dropout_adj
            edge_index,edge_attr=dropout_adj(
                edge_index,edge_attr=edge_attr,p=0.1,force_undirected=True
            )
        # ===== 编码器 =====
        x = self.base_encoder.node_encoder(x)
        x = self.base_encoder.node_norm(x)
        x = F.relu(x)

        # ===== GAT层1 =====
        out = self.base_encoder.gat1(x, edge_index, edge_attr)
        x1 = out[0] if isinstance(out, tuple) else out
        x1 = self.base_encoder.norm1(x1)
        x1 = F.relu(x1)

        # ===== GAT层2 =====
        out = self.base_encoder.gat2(x1, edge_index, edge_attr)
        x2 = out[0] if isinstance(out, tuple) else out
        x2 = self.base_encoder.norm2(x2)
        x2 = F.relu(x2)

        # ===== GAT层3 =====
        out = self.base_encoder.gat3(x2, edge_index, edge_attr)
        x3 = out[0] if isinstance(out, tuple) else out
        x3 = self.base_encoder.norm3(x3)
        x3 = F.relu(x3)
        node_features = x1 + x2 + x3  # 三层特征直接叠加

        # 图池化
        graph_features = global_mean_pool(node_features, batch)
        batch_size = graph_features.size(0)

        # DOS预测
        graph_features_expanded = graph_features.unsqueeze(1).expand(-1, self.dos_points, -1)
        energy_encoding_expanded = self.energy_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([graph_features_expanded, energy_encoding_expanded], dim=-1)
        combined = combined.transpose(1, 2)

        dos_pred = self.dos_conv_decoder(combined)
        dos_pred = dos_pred.squeeze(1)

        return dos_pred, node_features

    def get_pretrained_encoder(self):
        return self.base_encoder


class LearnableEdgeEncoder(nn.Module):
    """可学习的 RBF 边编码器 """

    def __init__(self, raw_edge_dim, hidden=32):
        super().__init__()
        self.transform=nn.Sequential(
            nn.Linear(raw_edge_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,raw_edge_dim)
        )

    def forward(self, edge_attr):
        return edge_attr + self.transform(edge_attr)


class ResidualBlock1D(nn.Module):
    """1D残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# ==================== 损失函数 V2.1 ====================

class DOSLoss(nn.Module):
    """DOS预测损失函数 V2.1"""

    def __init__(self, mse_weight=1.0, grad_weight=0.3,
                 spectral_weight=0.5, integral_weight=0.1,
                 peak_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.spectral_weight = spectral_weight
        self.integral_weight = integral_weight
        self.peak_weight = peak_weight

    def forward(self, dos_pred, dos_target):
        """
        计算损失
        """
        # 确保类型正确
        dos_pred = dos_pred.float()
        dos_target = dos_target.float()

        losses = {}

        # 1. 逐点MSE
        mse_loss = F.mse_loss(dos_pred, dos_target)
        losses['mse'] = mse_loss

        # 2. 梯度损失（一阶导数）
        dos_grad_pred = torch.diff(dos_pred, dim=1)
        dos_grad_target = torch.diff(dos_target, dim=1)
        grad_loss = F.mse_loss(dos_grad_pred, dos_grad_target)
        losses['grad'] = grad_loss

        # 3. 谱损失（频域约束）
        spectral_loss = self._spectral_loss(dos_pred, dos_target)
        losses['spectral'] = spectral_loss

        # 4. 积分面积损失 --- 方案A修复：比较平均值而非总和 ---
        # 通过除以点数，将尺度从总面积归一化到单个点的平均值，与MSE损失的量级相匹配
        integral_pred = torch.mean(dos_pred, dim=1) # [batch_size]
        integral_target = torch.mean(dos_target, dim=1) # [batch_size]
        integral_loss = F.mse_loss(integral_pred, integral_target)
        losses['integral'] = integral_loss

        # 5. 峰值损失（关注重要的DOS峰）
        peak_loss = self._peak_loss(dos_pred, dos_target)
        losses['peak'] = peak_loss

        # 总损失
        total_loss = (self.mse_weight * mse_loss +
                      self.grad_weight * grad_loss +
                      self.spectral_weight * spectral_loss +
                      self.integral_weight * integral_loss +
                      self.peak_weight * peak_loss)

        losses['total'] = total_loss

        return losses

    def _spectral_loss(self, pred, target):
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)

        amp_pred = torch.abs(pred_fft)
        amp_target = torch.abs(target_fft)

        n_freqs = amp_pred.size(1)
        weights = torch.linspace(1.0, 0.1, n_freqs, device=pred.device)

        return F.mse_loss(
            amp_pred * weights.unsqueeze(0),
            amp_target * weights.unsqueeze(0)
        )

    def _peak_loss(self, pred, target):
        """关注峰值的损失"""
        # 找到目标中的峰值区域（超过均值+0.5*标准差）
        target_mean = target.mean(dim=1, keepdim=True)
        target_std = target.std(dim=1, keepdim=True)
        peak_mask = (target > target_mean + 0.5 * target_std).float()

        # 对峰值区域加权
        weighted_diff = (pred - target) ** 2 * (1 + peak_mask)
        return weighted_diff.mean()


# ==================== 预训练器 V2.1 ====================

class DOSPretrainer:
    """DOS预训练器 V2.1"""

    def __init__(self, model: DOSPretrainGAT,
                 loss_fn: DOSLoss,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.history = defaultdict(list)
        self.visualizer = DOSVisualizer()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    def train(self, train_loader, val_loader=None,
              epochs=150, lr=0.001, weight_decay=1e-5,
              patience=30, warmup_epochs=5, grad_clip=1.0):
        """训练模型"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print("\n" + "=" * 60)
        print("开始DOS预训练 V2.1")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"学习率: {lr}, 预热: {warmup_epochs} epochs")
        print(f"损失权重: MSE={self.loss_fn.mse_weight}, "
              f"Grad={self.loss_fn.grad_weight}, "
              f"Spectral={self.loss_fn.spectral_weight}, "
              f"Peak={self.loss_fn.peak_weight}")

        for epoch in range(epochs):
            # 学习率预热
            if epoch < warmup_epochs:
                warmup_lr = lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # 训练
            self.model.train()
            train_losses = self._train_epoch(train_loader, optimizer, grad_clip, epoch)

            # 验证
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(val_loader)
                val_loss = val_metrics.get('val_total', float('inf'))

                if epoch >= warmup_epochs:
                    scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone()
                                        for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n⏹️ 早停于 epoch {epoch + 1}")
                        break

            # 记录历史
            self.history['epochs'].append(epoch + 1)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])

            for key, value in train_losses.items():
                self.history[f'train_{key}'].append(value)

            for key, value in val_metrics.items():
                clean_key = key.replace('val_', '')
                self.history[f'val_{clean_key}'].append(value)

            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\n📈 Epoch {epoch + 1}/{epochs}")
                print(f"  Train - Total: {train_losses.get('total', 0):.4f}, "
                      f"MSE: {train_losses.get('mse', 0):.4f}, "
                      f"Grad: {train_losses.get('grad', 0):.4f}, "
                      f"MAE: {train_losses.get('mae', 0):.4f}")
                if val_loader:
                    print(f"  Val   - Total: {val_metrics.get('val_total', 0):.4f}, "
                          f"MSE: {val_metrics.get('val_mse', 0):.4f}, "
                          f"MAE: {val_metrics.get('val_mae', 0):.4f}")

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ 最佳验证损失: {best_val_loss:.4f}")

        return dict(self.history)

    def _train_epoch(self, train_loader, optimizer, grad_clip, epoch):
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)

            if hasattr(batch, 'x'):
                batch.x = batch.x.float()
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr.float()
            if hasattr(batch, 'dos_total'):
                batch.dos_total = batch.dos_total.float()

            optimizer.zero_grad()

            dos_pred, _ = self.model(batch)
            dos_target = batch.dos_total

            if dos_pred.shape != dos_target.shape:
                dos_target = self._align_dimensions(dos_target, dos_pred.shape)

            losses = self.loss_fn(dos_pred, dos_target)
            print(f"  dos_pred stats: min={dos_pred.min():.4f}, max={dos_pred.max():.4f}, mean={dos_pred.mean():.4f}")
            print(
                f"  dos_target stats: min={dos_target.min():.4f}, max={dos_target.max():.4f}, mean={dos_target.mean():.4f}")
            print(f"  MSE loss value: {losses['mse']:.4f}")

            losses['total'].backward()

            if batch_idx == 0 and epoch == 0:
                self._monitor_gradients()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            optimizer.step()

            for key, value in losses.items():
                epoch_losses[key] += value.item()

            with torch.no_grad():
                mae = F.l1_loss(dos_pred, dos_target).item()
                epoch_losses['mae'] += mae

            n_batches += 1

        return {k: v / n_batches for k, v in epoch_losses.items()}

    def _validate(self, val_loader):
        self.model.eval()
        epoch_losses = defaultdict(float)
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                if hasattr(batch, 'x'):
                    batch.x = batch.x.float()
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    batch.edge_attr = batch.edge_attr.float()
                if hasattr(batch, 'dos_total'):
                    batch.dos_total = batch.dos_total.float()

                dos_pred, _ = self.model(batch)
                dos_target = batch.dos_total

                if dos_pred.shape != dos_target.shape:
                    dos_target = self._align_dimensions(dos_target, dos_pred.shape)

                losses = self.loss_fn(dos_pred, dos_target)

                for key, value in losses.items():
                    epoch_losses[f'val_{key}'] += value.item()

                mae = F.l1_loss(dos_pred, dos_target).item()
                epoch_losses['val_mae'] += mae

                n_batches += 1

        return {k: v / n_batches for k, v in epoch_losses.items()}

    def _align_dimensions(self, target, pred_shape):
        if target.dim() == 1:
            if target.size(0) == pred_shape[0] * pred_shape[1]:
                target = target.view(pred_shape)
            else:
                target = target[:pred_shape[1]].unsqueeze(0).expand(pred_shape[0], -1)
        elif target.dim() == 2:
            if target.size(1) != pred_shape[1]:
                if target.size(1) > pred_shape[1]:
                    target = target[:, :pred_shape[1]]
                else:
                    padded = torch.zeros(pred_shape, device=target.device)
                    padded[:, :target.size(1)] = target
                    target = padded
        return target

    def _monitor_gradients(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"\n🔍 梯度范数: {total_norm:.4f}")
        if total_norm < 1e-5:
            print("  ⚠️ 警告：梯度消失！")

    def visualize_predictions(self, data_loader, num_samples=3, save_dir='./dos_predictions'):
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()

        with torch.no_grad():
            sample_count = 0
            for batch in data_loader:
                if sample_count >= num_samples:
                    break

                batch = batch.to(self.device)
                if hasattr(batch, 'x'):
                    batch.x = batch.x.float()

                dos_pred, _ = self.model(batch)

                for i in range(min(batch.num_graphs, num_samples - sample_count)):
                    pred = dos_pred[i].cpu().numpy()

                    if hasattr(batch, 'dos_total'):
                        target = batch.dos_total
                        if target.dim() == 1:
                            target = target.view(batch.num_graphs, -1)
                        target = target[i].cpu().numpy()

                        if hasattr(batch, 'dos_energy'):
                            energy = batch.dos_energy
                            if energy.dim() == 1:
                                energy = energy.view(batch.num_graphs, -1)
                            energy = energy[i].cpu().numpy()
                        else:
                            energy = np.linspace(-8, 8, len(pred))

                        material_id = batch.filename[i] if hasattr(batch, 'filename') else f"sample_{sample_count}"

                        save_path = os.path.join(save_dir, f'dos_{material_id}.png')
                        self.visualizer.plot_dos_comparison(
                            energy, target, pred,
                            title=f"{material_id}",
                            save_path=save_path
                        )

                    sample_count += 1

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'base_encoder_state': self.model.base_encoder.state_dict(),
            'history': dict(self.history)
        }, path)
        print(f"✅ 模型已保存: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = defaultdict(list, checkpoint['history'])
        print(f"✅ 模型已加载: {path}")


# ==================== 单样本过拟合测试 ====================

def test_overfit_single_sample(data_path='dos_data.pt', save_dir='overfit_test'):
    """
    单样本过拟合测试
    用于验证模型架构是否正确
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("单样本过拟合测试")
    print("=" * 60)

    # 加载数据
    data_list = torch.load(data_path, map_location='cpu')
    sample = data_list[0]
    print(f"测试样本: {sample.filename}")

    # 创建单样本加载器
    train_loader = DataLoader([sample], batch_size=1)

    # 创建模型
    from process_label_train import PhotocatalysisGAT

    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1) if sample.edge_attr is not None else 9

    base_gat = PhotocatalysisGAT(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        global_in_dim=32,
        hidden_dim=64,
        num_heads=4,
        output_dim=9
    )

    model = DOSPretrainGAT(
        base_gat=base_gat,
        dos_points=128,
        hidden_dim=64
    )

    loss_fn = DOSLoss()
    trainer = DOSPretrainer(model, loss_fn)

    # 训练（更多epochs，更小学习率）
    history = trainer.train(
        train_loader=train_loader,
        val_loader=None,
        epochs=200,
        lr=0.0005,
        patience=50,
        warmup_epochs=10
    )

    # 可视化
    DOSVisualizer.plot_training_history(
        trainer.history,
        save_path=os.path.join(save_dir, 'overfit_history.png')
    )

    trainer.visualize_predictions(
        train_loader,
        num_samples=1,
        save_dir=save_dir
    )

    return trainer


# ==================== 主程序 ====================

def prepare_dos_data(cif_dir="data/cif_files",
                     dos_dir="data/dos_data",
                     output_path="dos_data.pt",
                     dos_points=128,
                     energy_range=(-8, 8)):
    """准备DOS预训练数据"""
    from pymatgen.core import Structure
    from basics_feature_extractor import GraphDataBuilder
    import glob

    print("=" * 60)
    print("准备DOS预训练数据")
    print("=" * 60)

    graph_builder = GraphDataBuilder(cutoff=5.0, max_neighbors=12)
    dos_processor = DOSProcessor(
        energy_range=energy_range,
        num_points=dos_points
    )

    cif_files = glob.glob(os.path.join(cif_dir, "*.cif"))

    available_dos = set()
    for f in os.listdir(dos_dir):
        if f.endswith('_dos.npy'):
            mp_id = f.replace('_dos.npy', '')
            if os.path.exists(os.path.join(dos_dir, f"{mp_id}_energy.npy")):
                available_dos.add(mp_id)

    print(f"找到 {len(cif_files)} 个CIF文件")
    print(f"找到 {len(available_dos)} 个可用的DOS数据")

    dos_data_list = []
    successful = 0

    for cif_file in cif_files:
        mp_id = os.path.basename(cif_file).replace('.cif', '')
        if mp_id not in available_dos:
            continue

        energy_file = os.path.join(dos_dir, f"{mp_id}_energy.npy")
        dos_file = os.path.join(dos_dir, f"{mp_id}_dos.npy")

        try:
            structure = Structure.from_file(cif_file)
            graph_data = graph_builder.build_from_structure(structure)

            # ============================================
            # ===== 在这里添加位置编码！ =====
            # ============================================

            # 获取原子坐标
            coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
            num_nodes = len(coords)

            # 方法1：简单的坐标拼接（最快速）
            # 将坐标归一化到 [-1, 1] 范围
            coords_norm = coords / (coords.max() - coords.min() + 1e-8) * 2 - 1

            # 到中心的距离
            center = coords.mean(dim=0)
            dist_to_center = torch.norm(coords - center, dim=1, keepdim=True)
            dist_norm = dist_to_center / (dist_to_center.max() + 1e-8)

            # 拼接到原节点特征
            enhanced_x = torch.cat([
                graph_data.x,
                coords_norm,  # [N, 3] 归一化坐标
                dist_norm,  # [N, 1] 到中心距离
            ], dim=1)

            # 更新节点特征
            graph_data.x = enhanced_x

            # 可选：打印特征维度变化
            # print(f"    节点特征维度: {graph_data.x.shape[1]} (原12 + 位置4 = 16)")

            # ============================================
            # ===== 位置编码添加完成 =====
            # ============================================

            energy = np.load(energy_file, allow_pickle=True)
            dos = np.load(dos_file, allow_pickle=True)

            if energy.dtype == np.dtype('O'):
                energy = np.array(energy.tolist(), dtype=np.float32).flatten()
            if dos.dtype == np.dtype('O'):
                dos = np.array(dos.tolist(), dtype=np.float32).flatten()

            processed_dos = dos_processor.process_dos_curve(energy, dos)

            dos_data = DOSData(
                x=graph_data.x.float(),
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr.float() if graph_data.edge_attr is not None else None,
                dos_energy=torch.tensor(dos_processor.energy_axis, dtype=torch.float32),
                dos_total=torch.tensor(processed_dos, dtype=torch.float32),
                filename=mp_id
            )

            dos_data_list.append(dos_data)
            successful += 1

            if successful % 10 == 0:
                print(f"  ✓ 已处理 {successful} 个样本...")

        except Exception as e:
            print(f"  ✗ {mp_id}: {str(e)[:50]}")
            continue

    if dos_data_list:
        torch.save(dos_data_list, output_path)
        print(f"\n✅ 已保存 {successful} 个数据到 {output_path}")

    return dos_data_list

def main():
    """主程序"""
    print("=" * 70)
    print("DOS预训练 V2.1")
    print("=" * 70)

    config = {
        'cif_dir': '/Users/luming/PycharmProjects/gnn_mat/data/cif_files',
        'dos_dir': '/Users/luming/PycharmProjects/gnn_mat/data/dos_data',
        'data_path': 'dos_data.pt',
        'model_save_path': 'dos_pretrained_gat_v2.pt',

        'dos_points': 128,
        'energy_range': (-8, 8),
        'hidden_dim': 64,
        'num_heads': 4,
        'batch_size': 16,
        'epochs': 150,
        'lr': 0.001,
    }

    # 选择运行模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--overfit':
        # 单样本过拟合测试
        test_overfit_single_sample(config['data_path'])
        return

    try:
        # 1. 数据
        if not os.path.exists(config['data_path']):
            dos_data_list = prepare_dos_data(
                cif_dir=config['cif_dir'],
                dos_dir=config['dos_dir'],
                output_path=config['data_path'],
                dos_points=config['dos_points'],
                energy_range=config['energy_range']
            )
        else:
            print(f"\n📂 加载数据: {config['data_path']}")
            dos_data_list = torch.load(config['data_path'])

        if not dos_data_list:
            print("❌ 无有效数据")
            return

        print(f"\n📊 总样本数: {len(dos_data_list)}")

        # 2. 划分
        n_total = len(dos_data_list)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        indices = np.random.permutation(n_total)
        train_data = [dos_data_list[i] for i in indices[:n_train]]
        val_data = [dos_data_list[i] for i in indices[n_train:n_train+n_val]]
        test_data = [dos_data_list[i] for i in indices[n_train+n_val:]]

        print(f"  训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}")

        # 3. 加载器
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'])
        test_loader = DataLoader(test_data, batch_size=config['batch_size'])

        # 4. 模型
        from process_label_train import PhotocatalysisGAT

        sample = dos_data_list[0]
        print(f"DOS target range: [{sample.dos_total.min():.4f}, {sample.dos_total.max():.4f}]")
        node_dim = sample.x.size(1)
        edge_dim = sample.edge_attr.size(1) if sample.edge_attr is not None else 9
        print(f"节点特征维度: {node_dim}")  # 应该输出 16
        print(f"边特征维度: {edge_dim}")

        base_gat = PhotocatalysisGAT(
            node_in_dim=node_dim,
            edge_in_dim=edge_dim,
            global_in_dim=32,
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            output_dim=9
        )

        model = DOSPretrainGAT(
            base_gat=base_gat,
            dos_points=config['dos_points'],
            hidden_dim=config['hidden_dim'],
            energy_range=config['energy_range']
        )

        # 5. 训练
        loss_fn = DOSLoss()
        trainer = DOSPretrainer(model, loss_fn)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            lr=config['lr']
        )

        # 6. 可视化
        DOSVisualizer.plot_training_history(trainer.history)
        trainer.visualize_predictions(test_loader, num_samples=5)

        # 7. 保存
        trainer.save_model(config['model_save_path'])

        print("\n✅ DOS预训练完成!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
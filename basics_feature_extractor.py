"""
增强型特征提取器 - 面向光催化性能预测的图数据特征工程
支持节点特征、边特征、全局特征的提取和加权
"""
import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import warnings
import os
import glob
import traceback
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from scipy.stats import skew, kurtosis
from abc import ABC, abstractmethod
from pymatgen.analysis.local_env import LocalStructOrderParams
import json

warnings.filterwarnings('ignore')


class FeatureExtractorBase(ABC):
    """特征提取器基类"""

    def __init__(self, name):
        self.name = name
        self.success_count = 0
        self.fail_count = 0
        self.failed_files = []

    @abstractmethod
    def _extract_impl(self, structure):
        """子类必须实现的提取方法"""
        pass

    def extract(self, structure):
        """提取特征的公共接口"""
        try:
            features = self._extract_impl(structure)
            self.success_count += 1
            return features
        except Exception as e:
            self.fail_count += 1
            return {}


class CompositionalExtractor(FeatureExtractorBase):
    """成分特征提取器"""

    def __init__(self):
        super().__init__("Compositional")

    def _extract_impl(self, structure):
        features = {}

        # 元素组成
        elements = [site.specie for site in structure]
        symbols = [el.symbol for el in elements]
        unique_elements = set(symbols)

        features['num_elements'] = len(unique_elements)
        features['num_atoms'] = len(structure)

        # 电负性统计
        electroneg = [el.X for el in elements if el.X]
        if electroneg:
            features['avg_electroneg'] = np.mean(electroneg)
            features['std_electroneg'] = np.std(electroneg)
            features['max_electroneg'] = max(electroneg)
            features['min_electroneg'] = min(electroneg)
            features['range_electroneg'] = max(electroneg) - min(electroneg)
        else:
            features.update({'avg_electroneg': 0, 'std_electroneg': 0, 'max_electroneg': 0,
                             'min_electroneg': 0, 'range_electroneg': 0})

        # 原子序数统计
        atomic_nums = [el.number for el in elements]
        features['avg_atomic_num'] = np.mean(atomic_nums)
        features['std_atomic_num'] = np.std(atomic_nums)

        # 原子质量统计
        atomic_masses = [el.atomic_mass for el in elements]
        features['avg_atomic_mass'] = np.mean(atomic_masses)
        features['total_mass'] = np.sum(atomic_masses)

        # 元素周期分布
        rows = [el.row for el in elements if el.row]
        if rows:
            features['avg_period'] = np.mean(rows)
            features['period_range'] = max(rows) - min(rows)
        else:
            features['avg_period'] = 0
            features['period_range'] = 0

        # 族分布
        groups = [el.group for el in elements if el.group]
        if groups:
            features['avg_group'] = np.mean(groups)
            features['group_range'] = max(groups) - min(groups)
        else:
            features['avg_group'] = 0
            features['group_range'] = 0

        # 金属/非金属比例
        metal_count = sum(1 for el in elements if self._is_metal(el))
        features['metal_ratio'] = metal_count / len(elements)

        # 过渡金属含量
        tm_count = sum(1 for el in elements if self._is_transition_metal(el))
        features['transition_metal_ratio'] = tm_count / len(elements)

        return features

    def _is_metal(self, element):
        """判断是否为金属"""
        if element.group:
            return element.group <= 2 or element.group >= 12 or element.symbol in ['Al', 'Ga', 'In', 'Sn', 'Pb', 'Bi']
        return False

    def _is_transition_metal(self, element):
        """判断是否为过渡金属"""
        if element.group:
            return 3 <= element.group <= 12
        return False


class StructuralExtractor(FeatureExtractorBase):
    """结构特征提取器"""

    def __init__(self):
        super().__init__("Structural")
        self.cnn = CrystalNN()
        self.lsop = LocalStructOrderParams(types=['q6'])

    def _extract_impl(self, structure):
        features = {}
        cn_list = []
        distortion_list = []

        for i in range(len(structure)):
            # 1. 提取配位数
            cn = len(self.cnn.get_nn_info(structure, i))
            cn_list.append(cn)

            # 2. 提取畸变 (q6)
            try:
                op = self.lsop.get_order_parameters(structure, i, indices_neighs=None)
                dist = op[0] if (op and op[0] is not None) else 0.5
            except:
                dist = 0.5
            distortion_list.append(dist)

        # 晶胞参数
        lattice = structure.lattice
        features['a'] = lattice.a
        features['b'] = lattice.b
        features['c'] = lattice.c
        features['alpha'] = lattice.alpha
        features['beta'] = lattice.beta
        features['gamma'] = lattice.gamma
        features['volume'] = structure.volume
        features['density'] = structure.density

        # 配位数和q6畸变
        features['avg_coordination'] = np.mean(cn_list)
        features["distortion"] = np.mean(distortion_list)
        features["volume_per_atom"] = structure.volume / len(structure)

        # 晶格比率
        features['a_b_ratio'] = lattice.a / lattice.b if lattice.b != 0 else 0
        features['a_c_ratio'] = lattice.a / lattice.c if lattice.c != 0 else 0
        features['b_c_ratio'] = lattice.b / lattice.c if lattice.c != 0 else 0

        # 晶胞角度特征
        angles = [lattice.alpha, lattice.beta, lattice.gamma]
        features['angle_sum'] = np.sum(angles)
        features['angle_product'] = np.prod(angles) / 1000000  # 缩放

        # 对称性分析
        try:
            spg_analyzer = SpacegroupAnalyzer(structure)
            features['spacegroup'] = spg_analyzer.get_space_group_number()
            features['spacegroup_symbol'] = spg_analyzer.get_space_group_symbol()

            crystal_system = spg_analyzer.get_crystal_system()
            cs_code = {
                'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7
            }.get(crystal_system, 0)
            features['crystal_system_code'] = cs_code

            # 点群
            point_group = spg_analyzer.get_point_group_symbol()
            features['point_group_complexity'] = len(point_group) if point_group else 0

            # Wyckoff位置多样性
            sym_data = spg_analyzer.get_symmetry_dataset()
            if sym_data and 'wyckoffs' in sym_data:
                wyckoffs = sym_data['wyckoffs']
                features['wyckoff_diversity'] = len(set(wyckoffs)) / len(wyckoffs)
            else:
                features['wyckoff_diversity'] = 0

        except Exception as e:
            features.update({'spacegroup': 0, 'crystal_system_code': 0, 'point_group_complexity': 0,
                             'wyckoff_diversity': 0})

        # 原子位置分布
        coords = np.array([site.coords for site in structure])

        # 原子间距统计
        if len(structure) > 1:
            distances = []
            for i in range(min(len(structure), 50)):
                for j in range(i + 1, min(len(structure), 50)):
                    distances.append(structure.get_distance(i, j))

            if distances:
                features['min_atom_distance'] = np.min(distances)
                features['max_atom_distance'] = np.max(distances)
                features['avg_atom_distance'] = np.mean(distances)
                features['std_atom_distance'] = np.std(distances)
            else:
                features.update({'min_atom_distance': 0, 'max_atom_distance': 0,
                                 'avg_atom_distance': 0, 'std_atom_distance': 0})

        # 晶格畸变
        features['lattice_anisotropy'] = np.std([lattice.a, lattice.b, lattice.c]) / np.mean(
            [lattice.a, lattice.b, lattice.c]) if np.mean([lattice.a, lattice.b, lattice.c]) > 0 else 0

        return features


class BondingExtractor(FeatureExtractorBase):
    """成键特征提取器"""

    def __init__(self, cutoff=3.0):
        super().__init__("Bonding")
        self.cutoff = cutoff

    def _extract_impl(self, structure):
        features = {}

        try:
            cnn = CrystalNN()
            coordination_nums = []
            valid_atoms = 0  # 统计有效原子数

            for i, site in enumerate(structure):
                try:
                    # 尝试方法1: get_local_order_parameters
                    local_env = cnn.get_local_order_parameters(structure, i)
                    if local_env and 'CN' in local_env and local_env['CN'] is not None:
                        cn = local_env['CN']
                        if cn > 0:  # 只添加有效值
                            coordination_nums.append(cn)
                            valid_atoms += 1
                            continue

                    # 方法2: 直接获取邻居数量
                    nn_info = cnn.get_nn_info(structure, i)
                    if nn_info:
                        coordination_nums.append(len(nn_info))
                        valid_atoms += 1
                        continue

                except Exception as e1:
                    pass

                # 备选方法：VoronoiNN
                try:
                    vnn = VoronoiNN()
                    nn_info = vnn.get_nn_info(structure, i)
                    if nn_info:
                        coordination_nums.append(len(nn_info))
                        valid_atoms += 1
                        continue
                except:
                    pass

                # 如果都失败，记录但不添加（避免0值污染）
                print(f"警告: 原子 {i} ({site.specie.symbol}) 配位数计算失败")

            # 只在有有效数据时计算统计量
            if coordination_nums and len(coordination_nums) >= len(structure) * 0.5:
                features['avg_coordination'] = np.mean(coordination_nums)
                features['std_coordination'] = np.std(coordination_nums)
                features['min_coordination'] = np.min(coordination_nums)
                features['max_coordination'] = np.max(coordination_nums)
                features['coordination_valid_ratio'] = valid_atoms / len(structure)
            else:
                # 如果没有足够有效数据，使用合理默认值
                # 根据元素类型估计典型配位数
                default_cn = self._estimate_typical_coordination(structure)
                features['avg_coordination'] = default_cn
                features['std_coordination'] = default_cn * 0.3
                features['min_coordination'] = max(1, default_cn - 2)
                features['max_coordination'] = default_cn + 2
                features['coordination_valid_ratio'] = 0

        except Exception as e:
            # 整体失败时的回退值
            default_cn = self._estimate_typical_coordination(structure)
            features.update({
                'avg_coordination': default_cn,
                'std_coordination': default_cn * 0.3,
                'min_coordination': max(1, default_cn - 2),
                'max_coordination': default_cn + 2,
                'coordination_valid_ratio': 0
            })

        return features

    def _estimate_typical_coordination(self, structure):
        """根据元素类型估计典型配位数"""
        elements = [site.specie.symbol for site in structure]

        # 常见元素的典型配位数
        typical_cn = {
            'O': 6, 'Ti': 6, 'Si': 4, 'Al': 4, 'Fe': 6,
            'Mg': 6, 'Ca': 6, 'Na': 6, 'K': 8, 'C': 4,
            'N': 3, 'P': 4, 'S': 4, 'Cl': 1, 'F': 1
        }

        # 计算加权平均
        total_weight = 0
        weighted_sum = 0

        for el in elements:
            if el in typical_cn:
                weighted_sum += typical_cn[el]
                total_weight += 1

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # 默认值
            return 4.0


class ElectronicExtractor(FeatureExtractorBase):
    """电子结构特征提取器"""

    def __init__(self, model_path=None):
        super().__init__("Electronic")
        self.model_path = model_path
        self.megnet_model = None
        self.use_megnet = model_path is not None

        if model_path:
            try:
                import matgl
                self.megnet_model = matgl.load_model(model_path)
                print(f"✅ ElectronicExtractor: 已加载MEGNet模型")
            except Exception as e:
                print(f"⚠️ ElectronicExtractor: 加载MEGNet模型失败: {e}")
                self.use_megnet = False

    def _extract_impl(self, structure):
        features = {}

        elements = [site.specie for site in structure]

        # 总价电子数
        total_valence = 0
        for el in elements:
            if el.group:
                if el.group <= 2:
                    total_valence += el.group
                elif el.group >= 13:
                    total_valence += el.group - 10
                else:
                    total_valence += 2
            else:
                total_valence += 2

        features['total_valence_electrons'] = total_valence
        features['valence_electrons_per_atom'] = total_valence / len(elements)

        # 电离能特征
        ionization_energies = []
        for el in elements:
            if el.ionization_energies:
                ionization_energies.append(el.ionization_energies[0])

        if ionization_energies:
            features['avg_ionization_energy'] = np.mean(ionization_energies)
            features['min_ionization_energy'] = np.min(ionization_energies)
            features['max_ionization_energy'] = np.max(ionization_energies)
        else:
            features.update({'avg_ionization_energy': 0, 'min_ionization_energy': 0, 'max_ionization_energy': 0})

        # 极化率估计
        polarizabilities = []
        for el in elements:
            if el.atomic_radius:
                polarizabilities.append(el.atomic_radius ** 3)
            else:
                polarizabilities.append(0)

        features['avg_polarizability'] = np.mean(polarizabilities)
        features['total_polarizability'] = np.sum(polarizabilities)

        # d电子特征
        d_electrons_list = []
        for el in elements:
            if el.group and 3 <= el.group <= 12:
                d_electrons_list.append(el.group - 2)
            else:
                d_electrons_list.append(0)

        features['total_d_electrons'] = np.sum(d_electrons_list)
        features['avg_d_electrons'] = np.mean(d_electrons_list) if d_electrons_list else 0
        features['has_d_electrons'] = 1 if np.sum(d_electrons_list) > 0 else 0

        # 电负性列表 - 在全局作用域定义，供后面使用
        electroneg = [el.X for el in elements if el.X]

        # 增强的带隙估计 - 使用MEGNet
        if self.use_megnet and self.megnet_model:
            try:
                # 多保真度预测
                methods = {0: "PBE", 1: "GLLB-SC", 2: "HSE", 3: "SCAN"}

                for method_id, method_name in methods.items():
                    try:
                        graph_attrs = torch.tensor([method_id])
                        bandgap = self.megnet_model.predict_structure(
                            structure=structure,
                            state_attr=graph_attrs
                        )
                        features[f'band_gap_{method_name}'] = float(bandgap)
                        print(f"      MEGNet预测 {method_name} 带隙: {float(bandgap)}")
                    except Exception as e:
                        print(f"      MEGNet预测 {method_name} 失败: {e}")
                        features[f'band_gap_{method_name}'] = 0.0

                # 使用HSE作为主要带隙估计
                features['band_gap_est'] = features.get('band_gap_HSE',
                                                        features.get('band_gap_PBE', 0))
            except Exception as e:
                print(f"      MEGNet预测整体失败: {e}")
                # 回退到简化估计
                if electroneg:
                    x_range = max(electroneg) - min(electroneg)
                    features['band_gap_est'] = 0.5 + 0.5 * x_range
                else:
                    features['band_gap_est'] = 1.0
        else:
            # 使用简化估计
            if electroneg:
                x_range = max(electroneg) - min(electroneg)
                features['band_gap_est'] = 0.5 + 0.5 * x_range
            else:
                features['band_gap_est'] = 1.0

        # d带中心估计 - 使用之前定义的 electroneg
        if np.sum(d_electrons_list) > 0 and electroneg:
            d_atoms = [i for i, d in enumerate(d_electrons_list) if d > 0]
            if d_atoms:
                # 确保索引不超出 electroneg 的范围
                valid_indices = [i for i in d_atoms if i < len(electroneg)]
                if valid_indices:
                    d_band_center = np.mean([electroneg[i] * d_electrons_list[i] for i in valid_indices])
                    features['d_band_center_est'] = d_band_center

                    # d带宽度
                    d_electroneg_values = [electroneg[i] for i in valid_indices]
                    features['d_band_width_est'] = np.std(d_electroneg_values) if len(d_electroneg_values) > 1 else 0
                else:
                    features['d_band_center_est'] = 0
                    features['d_band_width_est'] = 0
            else:
                features['d_band_center_est'] = 0
                features['d_band_width_est'] = 0
        else:
            features['d_band_center_est'] = 0
            features['d_band_width_est'] = 0

        # 元素电负性加权
        if electroneg:
            try:
                features['electroneg_product'] = np.prod(electroneg) / (10 ** (len(electroneg) - 1))
                weights = [el.number for el in elements if el.X]  # 只对有电负性的元素加权
                if weights:
                    features['electroneg_weighted_avg'] = np.average(electroneg, weights=weights)
                else:
                    features['electroneg_weighted_avg'] = np.mean(electroneg)
            except:
                features['electroneg_product'] = 0
                features['electroneg_weighted_avg'] = np.mean(electroneg) if electroneg else 0

        return features


class ThermodynamicExtractor(FeatureExtractorBase):
    """热力学特征提取器"""

    def __init__(self):
        super().__init__("Thermodynamic")

    def _extract_impl(self, structure):
        features = {}

        # Ewald能量计算
        try:
            ewald = EwaldSummation(structure)
            features['ewald_energy'] = ewald.total_energy
            features['ewald_energy_per_atom'] = ewald.total_energy / structure.num_sites

            # 能量的实空间和倒空间分量
            features['ewald_real'] = ewald.real_space_energy
            features['ewald_recip'] = ewald.reciprocal_space_energy
        except Exception as e:
            features.update({'ewald_energy': 0, 'ewald_energy_per_atom': 0,
                             'ewald_real': 0, 'ewald_recip': 0})

        # 形成能估计（基于元素参考态）
        formation_energy_est = 0
        elements = [site.specie for site in structure]

        # 简化模型：基于电负性和原子半径
        for el in elements:
            if el.X and el.atomic_radius:
                # 负值表示稳定
                formation_energy_est -= el.X * el.atomic_radius * 0.1

        features['formation_energy_est'] = formation_energy_est / len(elements)

        # 内聚能估计
        cohesive_energy_est = 0
        for el in elements:
            if el.atomic_mass:
                cohesive_energy_est += el.atomic_mass * 0.01

        features['cohesive_energy_est'] = cohesive_energy_est / len(elements)

        # 晶格能密度
        features['lattice_energy_density'] = features['ewald_energy'] / structure.volume if structure.volume > 0 else 0

        # 体积模量估计（基于密度和Ewald能量）
        if structure.density > 0 and features['ewald_energy_per_atom'] < 0:
            features['bulk_modulus_est'] = abs(features['ewald_energy_per_atom']) * structure.density * 10
        else:
            features['bulk_modulus_est'] = 0

        # 热膨胀系数估计（简化）
        features['thermal_expansion_est'] = 1.0 / (features['bulk_modulus_est'] + 1e-8) * 1000

        return features


class SurfaceExtractor(FeatureExtractorBase):
    """表面特征提取器 - 用于表面/界面相关特征"""

    def __init__(self):
        super().__init__("Surface")

    def _extract_impl(self, structure):
        features = {}

        # 注意：完整的表面计算需要专门的表面结构
        # 这里提供基于体相结构的表面特性估计

        # 表面能估计（基于配位数和键能）
        elements = [site.specie for site in structure]

        # 估计表面原子比例
        features['estimated_surface_atoms_ratio'] = 2.0 / (structure.num_sites ** (1 / 3))

        # 表面能估计
        bond_energy_est = 0
        for el in elements:
            if el.X:
                bond_energy_est += el.X * 10  # 粗略估计

        features['surface_energy_est'] = bond_energy_est / len(elements) * features['estimated_surface_atoms_ratio']

        # 活性位点密度估计（基于d电子和电负性）
        d_electrons = 0
        for el in elements:
            if el.group and 3 <= el.group <= 12:
                d_electrons += el.group - 2

        features['active_site_density_est'] = d_electrons / structure.volume if structure.volume > 0 else 0

        # 表面极性估计（基于电负性差异）
        electroneg = [el.X for el in elements if el.X]
        if electroneg:
            features['surface_polarity_est'] = np.std(electroneg) * 10
        else:
            features['surface_polarity_est'] = 0

        # 氧空位形成能估计（如果含氧）
        if any(el.symbol == 'O' for el in elements):
            # 基于平均电负性的简化估计
            avg_x = np.mean([el.X for el in elements if el.X])
            features['oxygen_vacancy_energy_est'] = 5.0 - avg_x * 2  # 经验公式
        else:
            features['oxygen_vacancy_energy_est'] = 0

        # 表面粗糙度估计
        coord_variation = 0
        try:
            cnn = CrystalNN()
            coords = []
            for i in range(min(len(structure), 20)):
                local_env = cnn.get_local_order_parameters(structure, i)
                if local_env and 'CN' in local_env:
                    coords.append(local_env['CN'])

            if coords:
                coord_variation = np.std(coords) / (np.mean(coords) + 1e-8)
        except:
            pass

        features['surface_roughness_est'] = coord_variation

        return features


class PhotocatalysisWeighting:
    """
    光催化性能关键特征权重分配器
    基于光催化机理：带隙、光吸收、载流子分离、表面反应活性
    """

    # 光催化关键特征权重映射
    FEATURE_WEIGHTS = {
        # 电子结构特征（权重最高）- 直接影响带隙和光吸收
        'band_gap_est': 1.0,
        'band_gap_HSE': 1.0,  # HSE方法最准确
        'band_gap_PBE': 0.8,  # PBE方法低估带隙
        'band_gap_SCAN': 0.9,  # SCAN方法较准确
        'band_gap_GLLB-SC': 0.85,
        'total_valence_electrons': 0.8,
        'valence_electrons_per_atom': 0.85,
        'avg_ionization_energy': 0.75,
        'min_ionization_energy': 0.7,
        'max_ionization_energy': 0.7,
        'avg_polarizability': 0.65,
        'total_polarizability': 0.6,
        'd_band_center_est': 0.9,  # d带中心（对过渡金属重要）
        'd_band_width_est': 0.85,
        'has_d_electrons': 0.8,
        'total_d_electrons': 0.85,
        'avg_d_electrons': 0.85,

        # 结构特征 - 影响能带结构和载流子迁移
        'volume': 0.5,
        'density': 0.4,
        'volume_per_atom': 0.45,
        'spacegroup': 0.3,
        'crystal_system_code': 0.35,
        'a': 0.3, 'b': 0.3, 'c': 0.3,
        'alpha': 0.2, 'beta': 0.2, 'gamma': 0.2,
        'lattice_anisotropy': 0.5,

        # 成键特征 - 影响载流子迁移率
        'avg_bond_length': 0.55,
        'std_bond_length': 0.5,
        'avg_coordination': 0.6,
        'std_coordination': 0.5,
        'min_coordination': 0.4,
        'max_coordination': 0.4,
        'bond_angle_distortion': 0.7,  # 键角畸变
        'hetero_bond_ratio': 0.65,

        # 热力学特征 - 影响稳定性
        'ewald_energy': 0.5,
        'ewald_energy_per_atom': 0.55,
        'formation_energy_est': 0.8,  # 形成能估计
        'cohesive_energy_est': 0.75,  # 内聚能估计
        'lattice_energy_density': 0.6,

        # 表面特征 - 直接影响光催化反应
        'surface_energy_est': 0.85,
        'active_site_density_est': 0.95,  # 活性位点密度
        'oxygen_vacancy_energy_est': 0.9,  # 氧空位形成能
        'surface_polarity_est': 0.7,
        'surface_roughness_est': 0.65,
        'estimated_surface_atoms_ratio': 0.6,

        # 元素特征
        'num_elements': 0.4,
        'metal_ratio': 0.5,
        'transition_metal_ratio': 0.7,
        'avg_electroneg': 0.6,
        'range_electroneg': 0.55,
    }

    @classmethod
    def get_weight(cls, feature_name):
        """获取特征权重，默认权重0.5"""
        for key, weight in cls.FEATURE_WEIGHTS.items():
            if key in feature_name:
                return weight
        return 0.5

    @classmethod
    def apply_weights(cls, features_dict):
        return features_dict


class GraphDataBuilder:
    """图数据构建器 - 将晶体结构转换为PyG图数据格式"""

    def __init__(self, cutoff=5.0, max_neighbors=12, model_path=None):
        """
        参数:
            cutoff: 截断半径(Å)，用于确定原子间连接
            max_neighbors: 每个原子的最大邻居数
            model_path: MEGNet预训练模型路径，如果为None则使用简化估计
        """
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.node_feature_dim = 12  # 节点特征维度
        self.edge_feature_dim = 9  # 边特征维度
        self.model_path = model_path
        self.megnet_model = None
        self.use_megnet = model_path is not None

        # 如果提供了模型路径，加载MEGNet模型
        if model_path:
            try:
                import matgl
                self.megnet_model = matgl.load_model(model_path)
                print(f"✅ 已加载MEGNet模型: {model_path}")
            except Exception as e:
                print(f"⚠️ 加载MEGNet模型失败: {e}，将使用简化估计")
                self.use_megnet = False

    def build_from_structure(self, structure, target_property=None):
        """
        从pymatgen Structure构建图数据

        返回:
            data: torch_geometric.data.Data对象，包含:
                - x: 节点特征 [num_nodes, node_features]
                - edge_index: 边索引 [2, num_edges]
                - edge_attr: 边特征 [num_edges, edge_features]
                - y: 全局目标值（如果有）
                - global_features: 全局特征向量
        """
        # 1. 提取节点特征
        node_features = self._extract_node_features(structure)

        # 2. 构建边连接和边特征
        edge_index, edge_features = self._build_edges(structure)

        # 3. 提取全局特征
        global_features = self._extract_global_features(structure)

        # 4. 构建PyG Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float) if len(edge_features) > 0 else torch.zeros(
                (0, self.edge_feature_dim), dtype=torch.float),
        )

        # 添加全局特征
        data.global_features = torch.tensor(global_features, dtype=torch.float)

        if target_property is not None:
            data.y = torch.tensor([target_property], dtype=torch.float)

        # 添加晶体学信息作为附加属性
        data.num_nodes = structure.num_sites
        data.spacegroup = self._get_spacegroup_number(structure)

        return data

    def _extract_node_features(self, structure):
        """提取原子级节点特征"""
        node_features = []

        for site in structure:
            element = site.specie
            features = []

            # 原子基本属性
            features.append(float(element.number))  # 原子序数
            features.append(float(element.group) if element.group else 0.0)  # 族
            features.append(float(element.row) if element.row else 0.0)  # 周期
            features.append(float(element.X) if element.X else 0.0)  # 电负性
            features.append(float(element.atomic_mass))  # 原子质量
            features.append(float(element.atomic_radius) if element.atomic_radius else 0.0)  # 原子半径

            # 电子结构相关
            features.append(float(self._get_valence_electrons(element)))  # 价电子数
            features.append(float(getattr(element, 'mendeleev_no', 0)))  # 门捷列夫数

            # 轨道信息
            d_electrons = self._get_d_electrons(element)
            features.append(float(d_electrons))  # d电子数
            features.append(1.0 if d_electrons > 0 else 0.0)  # 是否有d电子

            # 电离能
            if element.ionization_energies:
                features.append(float(element.ionization_energies[0]))
            else:
                features.append(0.0)

            # 极化率估计
            if element.atomic_radius:
                features.append(float(element.atomic_radius ** 3))
            else:
                features.append(0.0)

            node_features.append(features)

        # 归一化节点特征
        node_features = np.array(node_features, dtype=np.float32)

        # 逐列归一化
        for j in range(node_features.shape[1]):
            col = node_features[:, j]
            if np.std(col) > 1e-8:
                node_features[:, j] = (col - np.mean(col)) / (np.std(col) + 1e-8)
            else:
                node_features[:, j] = 0.0

        return node_features

    def _build_edges(self, structure):
        """使用 pymatgen 的 PBC 感知搜索构建边"""
        edge_index = []
        edge_features = []

        # 使用 get_all_neighbors 处理周期性边界条件 (PBC)
        # 这比 KDTree 更适合晶体结构
        all_neighbors = structure.get_all_neighbors(self.cutoff)

        for i, neighbors in enumerate(all_neighbors):
            # 限制最大邻居数，防止某些结构过密
            neighbors = sorted(neighbors, key=lambda x: x.nn_distance)
            for neighbor in neighbors[:self.max_neighbors]:
                j = neighbor.index
                dist = neighbor.nn_distance

                edge_index.append([i, j])
                # 提取边特征
                feat = self._get_edge_features(structure, i, j, dist)
                edge_features.append(feat)

        return np.array(edge_index).T, np.array(edge_features)

    def _get_edge_features(self, structure, i, j, distance):
        """计算边特征"""
        site_i = structure[i]
        site_j = structure[j]

        features = []

        # 几何特征
        features.append(float(distance))  # 键长
        features.append(1.0 / (distance + 1e-8))  # 倒空间距离
        features.append(float(np.exp(-distance)))  # 高斯衰减

        # 化学特征
        x_i = site_i.specie.X if site_i.specie.X else 0
        x_j = site_j.specie.X if site_j.specie.X else 0
        features.append(float(abs(x_i - x_j)))  # 电负性差
        features.append(float(site_i.specie.number + site_j.specie.number))  # 原子序数和
        features.append(float(abs(site_i.specie.number - site_j.specie.number)))  # 原子序数差

        # 成键类型指示
        features.append(1.0 if (site_i.specie.symbol == 'O' or site_j.specie.symbol == 'O') else 0.0)  # 是否涉及O（对光催化重要）
        features.append(1.0 if (self._is_transition_metal(site_i.specie) or self._is_transition_metal(
            site_j.specie)) else 0.0)  # 是否涉及过渡金属

        # 键级估计（简化）
        features.append(float(1.0 / (1 + abs(x_i - x_j))))  # 基于电负性差的键级估计

        return features

    def _extract_global_features(self, structure):
        """提取全局图级别特征"""
        global_features = []

        # 基本结构特征（保持不变）
        lattice = structure.lattice
        global_features.extend([
            float(lattice.a), float(lattice.b), float(lattice.c),
            float(lattice.alpha), float(lattice.beta), float(lattice.gamma),
            float(structure.volume),
            float(structure.density),
            float(structure.volume / structure.num_sites)
        ])

        # 对称性（保持不变）
        try:
            spg_analyzer = SpacegroupAnalyzer(structure)
            global_features.append(float(spg_analyzer.get_space_group_number()))
            crystal_system = spg_analyzer.get_crystal_system()
            cs_code = {
                'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7
            }.get(crystal_system, 0)
            global_features.append(float(cs_code))
        except:
            global_features.extend([0.0, 0.0])

        # 统计特征（保持不变）
        elements = set([site.specie.symbol for site in structure])
        global_features.append(float(len(elements)))

        electroneg = [site.specie.X for site in structure if site.specie.X]
        if electroneg:
            global_features.extend([
                float(np.mean(electroneg)),
                float(np.std(electroneg)),
                float(max(electroneg) - min(electroneg))
            ])
        else:
            global_features.extend([0.0, 0.0, 0.0])

        # 光催化关键特征 - 增强版带隙估计
        band_gap_results = self._estimate_band_gap_with_details(structure)

        if isinstance(band_gap_results, dict):
            # 如果是多方法预测结果
            global_features.append(float(band_gap_results.get('band_gap_HSE', 0)))
            global_features.append(float(band_gap_results.get('band_gap_PBE', 0)))
            global_features.append(float(band_gap_results.get('band_gap_SCAN', 0)))
        else:
            # 如果是单一值
            global_features.append(float(band_gap_results))
            global_features.append(0.0)  # 占位
            global_features.append(0.0)  # 占位

        # 其他特征（保持不变）
        d_band_center = self._estimate_d_band_center(structure)
        global_features.append(float(d_band_center))

        formation_energy = self._estimate_formation_energy(structure)
        global_features.append(float(formation_energy))

        tm_count = sum(1 for site in structure if self._is_transition_metal(site.specie))
        global_features.append(float(tm_count / structure.num_sites))

        return np.array(global_features, dtype=np.float32)

    def _estimate_band_gap_with_details(self, structure):
        """带细节的带隙估计，返回多方法预测结果"""
        if self.use_megnet and self.megnet_model:
            try:
                band_gaps = {}
                methods = {0: "PBE", 1: "GLLB-SC", 2: "HSE", 3: "SCAN"}

                for method_id, method_name in methods.items():
                    graph_attrs = torch.tensor([method_id])
                    bandgap = self.megnet_model.predict_structure(
                        structure=structure,
                        state_attr=graph_attrs
                    )
                    band_gaps[f'band_gap_{method_name}'] = float(bandgap)

                return band_gaps
            except:
                return self._estimate_band_gap_simple(structure)
        else:
            return self._estimate_band_gap_simple(structure)

    def _estimate_band_gap_simple(self, structure):
        """简化的带隙估计（原始方法）"""
        elements = [site.specie for site in structure]
        electroneg = [el.X for el in elements if el.X]

        if electroneg:
            x_diff = max(electroneg) - min(electroneg)
            estimated_gap = 0.5 + 0.5 * x_diff
        else:
            estimated_gap = 1.0

        if self._get_spacegroup_number(structure) > 0:
            estimated_gap *= 1.1

        return estimated_gap

    def _get_valence_electrons(self, element):
        """获取价电子数"""
        if element.group:
            if element.group <= 2:
                return element.group
            elif element.group >= 13:
                return element.group - 10
            elif 3 <= element.group <= 12:
                # 过渡金属
                return 2  # 简化处理
        return 2

    def _get_d_electrons(self, element):
        """获取d电子数"""
        if element.group and 3 <= element.group <= 12:
            # 过渡金属
            return element.group - 2
        return 0

    def _is_transition_metal(self, element):
        """判断是否为过渡金属"""
        if element.group:
            return 3 <= element.group <= 12
        return False

    def _get_spacegroup_number(self, structure):
        """获取空间群编号"""
        try:
            spg_analyzer = SpacegroupAnalyzer(structure)
            return spg_analyzer.get_space_group_number()
        except:
            return 0

    def _estimate_band_gap(self, structure):
        """估计带隙 - 使用MEGNet模型或简化模型"""

        # 如果可用，使用MEGNet模型进行多保真度预测
        if self.use_megnet and self.megnet_model:
            try:
                # 对不同计算方法进行预测
                band_gaps = {}

                # 多保真度预测：0:PBE, 1:GLLB-SC, 2:HSE, 3:SCAN
                methods = {
                    0: "PBE",
                    1: "GLLB-SC",
                    2: "HSE",
                    3: "SCAN"
                }

                for method_id, method_name in methods.items():
                    graph_attrs = torch.tensor([method_id])
                    bandgap = self.megnet_model.predict_structure(
                        structure=structure,
                        state_attr=graph_attrs
                    )
                    band_gaps[f'band_gap_{method_name}'] = float(bandgap)

                # 返回HSE作为主要带隙估计（最准确），如果可用
                if 'band_gap_HSE' in band_gaps:
                    return band_gaps['band_gap_HSE']
                else:
                    # 返回第一个可用的带隙
                    return list(band_gaps.values())[0]

            except Exception as e:
                print(f"⚠️ MEGNet预测失败: {e}，使用简化估计")
                # 回退到简化模型

        # 简化模型（原始方法）
        elements = [site.specie for site in structure]
        electroneg = [el.X for el in elements if el.X]

        if electroneg:
            x_diff = max(electroneg) - min(electroneg)
            estimated_gap = 0.5 + 0.5 * x_diff
        else:
            estimated_gap = 1.0

        # 考虑晶体结构的影响
        if self._get_spacegroup_number(structure) > 0:
            estimated_gap *= 1.1

        return estimated_gap

    def _estimate_d_band_center(self, structure):
        """估计d带中心"""
        d_electron_total = 0
        d_atom_count = 0

        for site in structure:
            d_electrons = self._get_d_electrons(site.specie)
            if d_electrons > 0:
                d_electron_total += d_electrons
                d_atom_count += 1

        if d_atom_count > 0:
            return d_electron_total / d_atom_count
        return 0

    def _estimate_formation_energy(self, structure):
        """估计形成能"""
        # 简化的形成能估计
        total_energy = 0
        for site in structure:
            # 基于原子序数的简单能量估计
            total_energy += site.specie.number * 0.1

        # 考虑晶格能
        lattice_energy = structure.volume / structure.num_sites * 0.01

        return -(total_energy + lattice_energy) / structure.num_sites


class EnhancedFeatureExtractor(FeatureExtractorBase):
    """增强版特征提取器 - 支持特征工程和加权"""

    def __init__(self, base_extractor, apply_weighting=True, normalize=True):
        super().__init__(base_extractor.name + "_enhanced")
        self.base_extractor = base_extractor
        self.apply_weighting = apply_weighting
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None

    def _extract_impl(self, structure):
        """实现基类的抽象方法"""
        # 基础特征提取
        base_features = self.base_extractor._extract_impl(structure)

        # 特征工程：创建交互特征
        engineered = self._create_interaction_features(structure, base_features)
        base_features.update(engineered)

        # 特征工程：创建统计特征
        stats_features = self._create_statistical_features(structure)
        base_features.update(stats_features)

        # 应用光催化权重
        if self.apply_weighting:
            base_features = PhotocatalysisWeighting.apply_weights(base_features)

        return base_features

    def extract_with_engineering(self, structure, cif_path="data/cif_files"):
        """提取特征并应用特征工程（兼容旧接口）"""
        try:
            features = self._extract_impl(structure)
            self.success_count += 1
            return features
        except Exception as e:
            self.fail_count += 1
            file_name = os.path.basename(cif_path) if cif_path else "unknown"
            self.failed_files.append(file_name)
            return {}

    def _create_interaction_features(self, structure, base_features):
        """创建特征交互项"""
        features = {}

        # 电负性与配位数交互
        if 'avg_electroneg' in base_features and 'avg_coordination' in base_features:
            features['electroneg_coord_product'] = base_features['avg_electroneg'] * base_features['avg_coordination']

        # 体积与带隙交互
        if 'volume' in base_features and 'band_gap_est' in base_features:
            features['volume_bandgap_ratio'] = base_features['volume'] / (base_features['band_gap_est'] + 1e-8)

        # 密度与形成能交互
        if 'density' in base_features and 'formation_energy_est' in base_features:
            features['density_formation_product'] = base_features['density'] * base_features['formation_energy_est']

        # 键长与配位数交互
        if 'avg_bond_length' in base_features and 'avg_coordination' in base_features:
            features['bond_length_coord_ratio'] = base_features['avg_bond_length'] / (
                    base_features['avg_coordination'] + 1e-8)

        # 对称性与电负性范围交互
        if 'spacegroup' in base_features and 'range_electroneg' in base_features:
            features['symmetry_electroneg_product'] = base_features['spacegroup'] * base_features['range_electroneg']

        # 过渡金属与d带中心交互
        if 'transition_metal_ratio' in base_features and 'd_band_center_est' in base_features:
            features['tm_dband_product'] = base_features['transition_metal_ratio'] * base_features['d_band_center_est']

        # 体积与配位数交互
        if 'volume_per_atom' in base_features and 'avg_coordination' in base_features:
            features['volume_coord_ratio'] = base_features['volume_per_atom'] / (
                    base_features['avg_coordination'] + 1e-8)

        return features

    def _create_statistical_features(self, structure):
        """创建统计特征"""
        features = {}

        # 原子间距离分布统计
        distances = []
        for i in range(min(len(structure), 50)):  # 限制计算量
            for j in range(i + 1, min(len(structure), 50)):
                try:
                    distances.append(structure.get_distance(i, j))
                except:
                    pass

        if len(distances) > 1:
            features['dist_skewness'] = float(skew(distances))
            features['dist_kurtosis'] = float(kurtosis(distances))
            features['dist_entropy'] = float(self._calculate_entropy(distances))
        else:
            features.update({'dist_skewness': 0, 'dist_kurtosis': 0, 'dist_entropy': 0})

        # 元素分布熵
        element_counts = defaultdict(int)
        for site in structure:
            element_counts[site.specie.symbol] += 1

        probs = [count / len(structure) for count in element_counts.values()]
        if probs:
            features['element_entropy'] = float(-sum(p * np.log(p) for p in probs))
        else:
            features['element_entropy'] = 0

        return features

    def _calculate_entropy(self, data, bins=20):
        """计算数据分布的熵"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            return -sum(hist * np.log(hist))
        return 0


class PhotocatalysisData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'global_features':
            return None  # None 表示使用 stack (增加一维)，而不是 concat
        return super().__cat_dim__(key, value, *args, **kwargs)


class PhotocatalysisFeaturePipeline:
    """
    光催化性能预测完整特征工程流水线
    支持设置多个目标特征作为y值
    """

    # 您指定的9个目标特征
    TARGET_FEATURES = {
        'band_gap_pred': 'electronic_band_gap_HSE',  # 带隙
        'surface_energy_pred': 'surface_surface_energy_est',  # 表面能
        'active_site_pred': 'surface_active_site_density_est',  # 活性位点密度
        'vacancy_energy_pred': 'surface_oxygen_vacancy_energy_est',  # 氧空位形成能
        'distortion_pred': 'structural_distortion',  # 结构畸变
        'ionization_energy': 'electronic_avg_ionization_energy',  # 电离能
        'polarizability': 'electronic_avg_polarizability',  # 极化率
        'formation_energy': 'thermodynamic_formation_energy_est',  # 形成能
        'bulk_modulus': 'thermodynamic_bulk_modulus_est',  # 体积模量
    }

    def __init__(self, target_property='band_gap_pred', megnet_model_path=None, multi_target=True):
        """
        参数:
            target_property: 主要目标属性名（在TARGET_FEATURES中）
            megnet_model_path: MEGNet模型路径
            multi_target: 是否保存所有9个特征作为y值
        """
        self.target_property = target_property
        self.multi_target = multi_target
        self.megnet_model_path = megnet_model_path or "/Users/luming/PycharmProjects/gnn_mat/pretrained_models/MEGNet-MP-2019.4.1-BandGap-mfi"
        self.graph_builder = GraphDataBuilder(model_path=self.megnet_model_path)
        self.extractors = []
        self.graph_data_list = []
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.all_feature_keys = None
        self.builder_global_dim = None

        # 存储目标值
        self.target_values = {name: [] for name in self.TARGET_FEATURES.keys()}

        # 存储DataFrame（将在batch_process中设置）
        self.features_df = None
        self.targets_df = None

        # 初始化提取器
        self._init_extractors()

    def _init_extractors(self):
        """初始化所有特征提取器"""
        extractor_classes = [
            CompositionalExtractor,
            StructuralExtractor,
            BondingExtractor,
            ElectronicExtractor,
            ThermodynamicExtractor,
            SurfaceExtractor
        ]

        for ext_class in extractor_classes:
            if ext_class == ElectronicExtractor:
                base_ext = ext_class(model_path=self.megnet_model_path)
            else:
                base_ext = ext_class()
            enhanced = EnhancedFeatureExtractor(base_ext, apply_weighting=True, normalize=True)
            self.extractors.append(enhanced)

    def extract_target_values(self, features_dict):
        """
        从特征字典中提取9个目标特征的值

        参数:
            features_dict: 包含所有特征的字典（带前缀）

        返回:
            target_dict: 9个目标特征的值
        """
        target_dict = {}

        for target_name, feature_name in self.TARGET_FEATURES.items():
            # 在特征字典中查找对应的特征
            found_value = None

            # 直接匹配完整特征名
            if feature_name in features_dict:
                found_value = features_dict[feature_name]
            else:
                # 尝试匹配带前缀的版本
                for key in features_dict.keys():
                    if feature_name.split('_')[-1] in key:  # 匹配最后一部分
                        found_value = features_dict[key]
                        break

            # 如果找到值，保存；否则保存NaN
            if found_value is not None and isinstance(found_value, (int, float, np.number)):
                target_dict[target_name] = float(found_value)
            else:
                target_dict[target_name] = float('nan')
                print(f"   ⚠️ 未找到目标特征: {target_name} ({feature_name})")

        return target_dict

    def process_cif(self, cif_path, external_targets=None):
        """
        处理单个CIF文件，提取特征并设置y值

        参数:
            cif_path: CIF文件路径
            external_targets: 可选的外部目标值字典（如果不想使用内部计算的特征）

        返回:
            final_data: 包含y值的图数据
            base_features: 所有特征字典
            target_dict: 9个目标特征的值
        """
        try:
            # 1. 加载结构
            structure = Structure.from_file(cif_path)
            base_features = {'filename': os.path.basename(cif_path)}

            # 2. 调用图构建器
            graph_data = self.graph_builder.build_from_structure(structure)

            # 3. 提取所有增强特征
            all_enhanced_features = {}
            for extractor in self.extractors:
                try:
                    features = extractor.extract(structure)
                    # 添加前缀
                    if hasattr(extractor, 'base_extractor'):
                        prefix = extractor.base_extractor.name.lower() + '_'
                    else:
                        prefix = extractor.name.lower() + '_'

                    prefixed_features = {prefix + k: v for k, v in features.items()}
                    all_enhanced_features.update(prefixed_features)
                except Exception as ext_e:
                    continue

            base_features.update(all_enhanced_features)

            # 4. 提取目标特征值
            target_dict = {}
            if external_targets is not None:
                # 如果提供了外部目标值，使用外部值
                target_dict = external_targets
            else:
                # 否则从计算的特征中提取
                target_dict = self.extract_target_values(all_enhanced_features)

            # 5. 构建特征向量（用于全局特征）
            if self.all_feature_keys is None:
                # 首次处理文件时初始化特征键列表
                self.all_feature_keys = sorted([
                    k for k, v in all_enhanced_features.items()
                    if isinstance(v, (int, float, np.number))
                       and k not in self.TARGET_FEATURES.values()  # 排除目标特征
                ])

            enhanced_vector = []
            for k in self.all_feature_keys:
                value = all_enhanced_features.get(k, 0.0)
                enhanced_vector.append(float(value))
            # 6. 合并特征
            builder_global = graph_data.global_features.detach().cpu().numpy().flatten()

            # 如果是第一次处理文件，记录builder_global的维度
            if self.builder_global_dim is None:
                self.builder_global_dim = len(builder_global)
                print(f"   设置 builder_global_dim = {self.builder_global_dim}")

            # 确保所有文件的builder_global维度一致
            if len(builder_global) != self.builder_global_dim:
                if len(builder_global) < self.builder_global_dim:
                    # 如果当前维度较小，进行填充
                    builder_global = np.pad(builder_global,
                                            (0, self.builder_global_dim - len(builder_global)),
                                            mode='constant',
                                            constant_values=0)
                else:
                    # 如果当前维度较大，进行截断
                    builder_global = builder_global[:self.builder_global_dim]
            combined_global = np.concatenate([
                builder_global,
                np.array(enhanced_vector, dtype=np.float32)
            ])
            combined_global = np.nan_to_num(combined_global, nan=0.0).astype(np.float32)
            global_tensor = torch.tensor(combined_global, dtype=torch.float).view(1, -1)

            # 7. 创建图数据，并设置y值
            final_data = PhotocatalysisData(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr,
                global_features=global_tensor,
                filename=os.path.basename(cif_path)
            )

            # 设置y值 - 根据multi_target决定是单目标还是多目标
            if self.multi_target:
                # 多目标：创建一个9维的y向量
                y_values = []
                for target_name in self.TARGET_FEATURES.keys():
                    value = target_dict.get(target_name, float('nan'))
                    if np.isnan(value):
                        print(f"   ⚠️ {cif_path} 的 {target_name} 为NaN，使用0填充")
                        value = 0.0
                    y_values.append(float(value))

                final_data.y = torch.tensor([y_values], dtype=torch.float)  # [1, 9]
                final_data.y_names = list(self.TARGET_FEATURES.keys())  # 保存y值名称
            else:
                # 单目标：使用指定的target_property
                target_value = target_dict.get(self.target_property, float('nan'))
                if np.isnan(target_value):
                    print(f"   ⚠️ {cif_path} 的 {self.target_property} 为NaN，使用0填充")
                    target_value = 0.0
                final_data.y = torch.tensor([target_value], dtype=torch.float)

            # 保存目标值到字典（用于后续分析）
            for name, value in target_dict.items():
                if name in self.target_values:
                    self.target_values[name].append(value)

            return final_data, base_features, target_dict

        except Exception as e:
            print(f"❌ 处理文件 {cif_path} 失败: {str(e)}")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误详情: {traceback.format_exc()}")
            raise e

    def batch_process(self, folder_path, external_targets_dict=None):
        """
        批量处理文件夹

        参数:
            folder_path: CIF文件夹路径
            external_targets_dict: 外部目标值字典 {filename: {target_name: value}}

        返回:
            df: 特征DataFrame
            graph_data_list: 图数据列表
            targets_df: 目标值DataFrame
        """
        cif_files = glob.glob(os.path.join(folder_path, '*.cif'))
        cif_files.extend(glob.glob(os.path.join(folder_path, '*.CIF')))
        cif_files = list(set(cif_files))
        cif_files.sort()

        print(f"\n📥 开始处理 {len(cif_files)} 个CIF文件...")

        all_graph_data = []
        all_features_dict = []
        all_targets_dict = []
        all_filenames = []
        successful_files = 0

        for i, cif_file in enumerate(cif_files, 1):
            filename = os.path.basename(cif_file)
            external_targets = external_targets_dict.get(filename) if external_targets_dict else None

            print(f"[{i}/{len(cif_files)}] 处理: {filename}")

            try:
                final_data, features_dict, target_dict = self.process_cif(
                    cif_file, external_targets
                )

                all_graph_data.append(final_data)
                all_features_dict.append(features_dict)
                all_targets_dict.append(target_dict)
                all_filenames.append(filename)
                successful_files += 1

                # 打印第一个文件的y值
                if i == 1:
                    print(f"\n🔍 第一个文件的y值:")
                    for name, value in target_dict.items():
                        print(f"   {name}: {value:.4f}")

            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
                continue

        if not all_features_dict:
            print("❌ 没有成功处理的文件")
            return pd.DataFrame(), [], pd.DataFrame()

        print(f"\n✅ 成功处理 {successful_files} 个文件")

        # 构建特征DataFrame
        self.features_df = self._build_features_dataframe(all_features_dict, all_filenames)

        # 构建目标值DataFrame
        self.targets_df = pd.DataFrame(all_targets_dict)
        self.targets_df['filename'] = all_filenames

        # 统计目标值的可用性
        print(f"\n📊 目标值统计:")
        for target_name in self.TARGET_FEATURES.keys():
            valid_count = self.targets_df[target_name].notna().sum()
            print(f"   {target_name}: {valid_count}/{len(self.targets_df)} 可用")

        # 保存图数据
        self.graph_data_list = all_graph_data

        return self.features_df, self.graph_data_list, self.targets_df

    def _build_features_dataframe(self, all_features_dict, all_filenames):
        """构建特征DataFrame"""
        # 找出所有特征键
        all_keys = set()
        for features in all_features_dict:
            numeric_keys = [k for k, v in features.items()
                            if isinstance(v, (int, float, np.number))]
            all_keys.update(numeric_keys)

        all_keys = sorted(all_keys)

        # 构建DataFrame
        results_df = []
        for filename, features_dict in zip(all_filenames, all_features_dict):
            df_row = {'filename': filename}
            for k in all_keys:
                value = features_dict.get(k, 0.0)
                if isinstance(value, (int, float, np.number)):
                    df_row[k] = float(value)
                else:
                    df_row[k] = 0.0
            results_df.append(df_row)

        df = pd.DataFrame(results_df)
        df = self._handle_missing_values(df)

        return df

    def _handle_missing_values(self, df):
        """处理缺失值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
        return df

    def save_data(self, graph_path="graph_data.pt", features_path="features.csv", targets_path="targets.csv"):
        """
        保存所有数据到文件

        参数:
            graph_path: 图数据保存路径 (.pt文件)
            features_path: 特征DataFrame保存路径 (.csv文件)
            targets_path: 目标值DataFrame保存路径 (.csv文件)

        返回:
            bool: 是否成功保存
        """
        success = True

        # 1. 保存图数据
        if self.graph_data_list:
            try:
                # 保存单个图数据列表
                torch.save(self.graph_data_list, graph_path)
                print(f"✅ 图数据已保存到: {graph_path}")
                print(f"   - 图数量: {len(self.graph_data_list)}")

                # 同时保存为批次格式（便于GAT训练）
                try:
                    batch = Batch.from_data_list(self.graph_data_list)
                    batch_path = graph_path.replace('.pt', '_batch.pt')
                    torch.save(batch, batch_path)
                    print(f"✅ 批次图数据已保存到: {batch_path}")
                    print(f"   - 批次中的图数量: {batch.num_graphs}")
                    print(f"   - 总节点数: {batch.x.shape[0]}")
                    print(f"   - 总边数: {batch.edge_index.shape[1]}")

                    # 如果是多目标模式，保存y_names到单独的文件
                    if self.multi_target and hasattr(self.graph_data_list[0], 'y_names'):
                        y_names_path = graph_path.replace('.pt', '_y_names.json')
                        with open(y_names_path, 'w') as f:
                            json.dump(self.graph_data_list[0].y_names, f)
                        print(f"✅ y值名称已保存到: {y_names_path}")

                except Exception as e:
                    print(f"⚠️ 创建批次数据失败: {e}")
                    print(f"   详细错误: {traceback.format_exc()}")
                    success = False

            except Exception as e:
                print(f"❌ 保存图数据失败: {e}")
                success = False
        else:
            print("⚠️ 没有图数据可保存")
            success = False

        # 2. 保存特征DataFrame
        if self.features_df is not None and not self.features_df.empty:
            try:
                self.features_df.to_csv(features_path, index=False)
                print(f"✅ 特征DataFrame已保存到: {features_path}")
                print(f"   - 样本数: {len(self.features_df)}")
                print(f"   - 特征数: {len(self.features_df.columns) - 1}")  # 减去filename列
            except Exception as e:
                print(f"❌ 保存特征DataFrame失败: {e}")
                success = False
        else:
            print("⚠️ 没有特征DataFrame可保存")
            success = False

        # 3. 保存目标值DataFrame
        if self.targets_df is not None and not self.targets_df.empty:
            try:
                self.targets_df.to_csv(targets_path, index=False)
                print(f"✅ 目标值DataFrame已保存到: {targets_path}")
                print(f"   - 样本数: {len(self.targets_df)}")
                print(f"   - 目标变量数: {len([c for c in self.targets_df.columns if c != 'filename'])}")

                # 保存目标值统计信息
                stats_path = targets_path.replace('.csv', '_stats.csv')
                stats_df = self.targets_df.describe()
                stats_df.to_csv(stats_path)
                print(f"✅ 目标值统计信息已保存到: {stats_path}")

            except Exception as e:
                print(f"❌ 保存目标值DataFrame失败: {e}")
                success = False
        else:
            print("⚠️ 没有目标值DataFrame可保存")
            success = False

        # 4. 保存特征重要性（如果已计算）
        if self.feature_importance:
            try:
                importance_path = features_path.replace('.csv', '_importance.csv')
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v}
                    for k, v in self.feature_importance.items()
                ])
                importance_df.to_csv(importance_path, index=False)
                print(f"✅ 特征重要性已保存到: {importance_path}")
            except Exception as e:
                print(f"⚠️ 保存特征重要性失败: {e}")

        # 5. 保存元数据
        try:
            metadata_path = graph_path.replace('.pt', '_metadata.json')
            metadata = {
                'num_graphs': len(self.graph_data_list) if self.graph_data_list else 0,
                'multi_target': self.multi_target,
                'target_property': self.target_property,
                'target_features': list(self.TARGET_FEATURES.keys()),
                'source_features': list(self.TARGET_FEATURES.values()),
                'builder_global_dim': self.builder_global_dim,
                'total_feature_dim': len(
                    self.all_feature_keys) + self.builder_global_dim if self.all_feature_keys else 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ 元数据已保存到: {metadata_path}")

        except Exception as e:
            print(f"⚠️ 保存元数据失败: {e}")

        return success

    def load_data(self, graph_path="graph_data.pt", features_path="features.csv", targets_path="targets.csv"):
        """
        从文件加载数据

        参数:
            graph_path: 图数据路径
            features_path: 特征DataFrame路径
            targets_path: 目标值DataFrame路径

        返回:
            tuple: (graph_data_list, features_df, targets_df)
        """
        # 加载图数据
        if os.path.exists(graph_path):
            try:
                self.graph_data_list = torch.load(graph_path)
                print(f"✅ 已加载图数据: {graph_path}")
                print(f"   - 图数量: {len(self.graph_data_list)}")
            except Exception as e:
                print(f"❌ 加载图数据失败: {e}")
                self.graph_data_list = []

        # 加载特征DataFrame
        if os.path.exists(features_path):
            try:
                self.features_df = pd.read_csv(features_path)
                print(f"✅ 已加载特征DataFrame: {features_path}")
                print(f"   - 形状: {self.features_df.shape}")
            except Exception as e:
                print(f"❌ 加载特征DataFrame失败: {e}")
                self.features_df = None

        # 加载目标值DataFrame
        if os.path.exists(targets_path):
            try:
                self.targets_df = pd.read_csv(targets_path)
                print(f"✅ 已加载目标值DataFrame: {targets_path}")
                print(f"   - 形状: {self.targets_df.shape}")
            except Exception as e:
                print(f"❌ 加载目标值DataFrame失败: {e}")
                self.targets_df = None

        return self.graph_data_list, self.features_df, self.targets_df


def normalize_graph_data(graph_list):
    """
    对 graph_list 中的节点特征进行归一化
    """
    # 1. 提取所有图中所有的节点特征
    all_features = []
    for data in graph_list:
        all_features.append(data.x.numpy())

    all_features_flattened = np.vstack(all_features)

    # 2. 计算均值和标准差
    scaler = StandardScaler()
    scaler.fit(all_features_flattened)

    # 3. 应用归一化并更新 graph_list
    for data in graph_list:
        normalized_x = scaler.transform(data.x.numpy())
        data.x = torch.tensor(normalized_x, dtype=torch.float)

    print("特征归一化完成。")
    return graph_list, scaler


def main_with_targets():
    """主程序：提取特征并设置9个目标值"""

    print("=" * 70)
    print("🚀 光催化性能预测特征工程流水线（多目标版）")
    print("=" * 70)

    # 显示目标特征
    print("\n🎯 目标特征（9个）:")
    for name, feature in PhotocatalysisFeaturePipeline.TARGET_FEATURES.items():
        print(f"   {name:20} <- {feature}")

    # 初始化流水线
    megnet_model_path = "/Users/luming/PycharmProjects/gnn_mat/pretrained_models/MEGNet-MP-2019.4.1-BandGap-mfi"
    pipeline = PhotocatalysisFeaturePipeline(
        target_property='band_gap_pred',
        megnet_model_path=megnet_model_path,
        multi_target=True  # 启用多目标模式
    )

    # 设置输入输出路径
    input_folder = "data/cif_files/"
    output_prefix = "photocatalysis"

    if not os.path.exists(input_folder):
        print(f"❌ 输入文件夹不存在: {input_folder}")
        os.makedirs(input_folder, exist_ok=True)
        print(f"✅ 已创建文件夹: {input_folder}")
        print("请将CIF文件放入该文件夹后重新运行")
        return

    # 可选：从外部文件加载目标值
    external_targets = None
    target_file = os.path.join(input_folder, "targets.csv")
    if os.path.exists(target_file):
        try:
            target_df = pd.read_csv(target_file)
            # 期望格式：filename, band_gap_pred, surface_energy_pred, ...
            external_targets = {}
            for _, row in target_df.iterrows():
                filename = row['filename']
                targets = {}
                for target_name in PhotocatalysisFeaturePipeline.TARGET_FEATURES.keys():
                    if target_name in row and pd.notna(row[target_name]):
                        targets[target_name] = row[target_name]
                if targets:  # 只添加有目标值的文件
                    external_targets[filename] = targets
            print(f"✅ 已加载外部目标值，共 {len(external_targets)} 个文件")
        except Exception as e:
            print(f"⚠️ 目标文件加载失败: {e}")
            print("将使用内部计算的特征作为目标值")

    # 批量处理
    print("\n📥 开始批量处理CIF文件...")
    df, graph_data_list, targets_df = pipeline.batch_process(input_folder, external_targets)

    if len(df) == 0:
        print("❌ 没有成功处理的文件")
        return

    # 保存结果
    print("\n💾 正在保存结果...")
    pipeline.save_data(
        graph_path=f"{output_prefix}_graph.pt",
        features_path=f"{output_prefix}_features.csv",
        targets_path=f"{output_prefix}_targets.csv"
    )

    # 输出统计信息
    print(f"\n📊 处理统计:")
    print(f"   - 成功处理文件数: {len(df)}")
    print(f"   - 特征维度: {df.shape[1] - 1} (不包括filename)")

    # 显示目标值统计
    print(f"\n📈 目标值统计:")
    for target_name in pipeline.TARGET_FEATURES.keys():
        if target_name in targets_df.columns:
            valid_values = targets_df[target_name].dropna()
            if len(valid_values) > 0:
                print(f"   {target_name:20}: 均值={valid_values.mean():.4f}, "
                      f"标准差={valid_values.std():.4f}, "
                      f"范围=[{valid_values.min():.4f}, {valid_values.max():.4f}]")
            else:
                print(f"   {target_name:20}: 无有效值")

    return df, graph_data_list, targets_df


if __name__ == "__main__":
    # 运行多目标版本
    df, graph_data, targets_df = main_with_targets()


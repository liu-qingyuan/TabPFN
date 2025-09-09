"""
UDA Medical Imbalance Project - 医疗数据加载器

基于TabPFN项目的数据加载模式，使用预定义的特征集和数据路径
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MedicalDataLoader:
    """
    医疗数据加载器
    
    支持多医院数据集：
    - A: AI4health数据集 (源域)
    - B: HenanCancerHospital数据集 (目标域)
    - C: GuangzhouMedicalHospital数据集 (目标域)
    
    使用预定义的特征集，无需额外特征选择
    """
    
    # 数据集映射
    DATASET_MAPPING = {
        'A': 'AI4health',
        'B': 'HenanCancerHospital', 
        'C': 'GuangzhouMedicalHospital'
    }
    
    # 数据文件路径 - 使用相对路径，自动适配项目位置
    @property
    def DATA_PATHS(self):
        # 获取项目根目录
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # 从 uda_medical_imbalance_project/data/ 回到项目根目录
        data_dir = project_root / "data"
        
        return {
            'A': str(data_dir / "AI4healthcare.xlsx"),
            'B': str(data_dir / "HenanCancerHospital_features63_58.xlsx"),
            'C': str(data_dir / "GuangzhouMedicalHospital_features23_no_nan_new_fixed.xlsx")
        }
    
    # 全部63个特征
    ALL_63_FEATURES = [
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
        'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20',
        'Feature21', 'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30',
        'Feature31', 'Feature32', 'Feature33', 'Feature34', 'Feature35', 'Feature36', 'Feature37', 'Feature38', 'Feature39', 'Feature40',
        'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45', 'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50',
        'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60',
        'Feature61', 'Feature62', 'Feature63'
    ]
    
    # 经过RFE筛选的58个特征
    SELECTED_58_FEATURES = [
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
        'Feature11', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20', 'Feature21',
        'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31',
        'Feature32', 'Feature35', 'Feature37', 'Feature38', 'Feature39', 'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55',
        'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61', 'Feature62', 'Feature63'
    ]
    
    # 最佳特征集
    BEST_7_FEATURES = [
        'Feature63', 'Feature2', 'Feature46', 
        'Feature56', 'Feature42', 'Feature39', 'Feature43'
    ]
    
    BEST_8_FEATURES = [
        'Feature63', 'Feature2', 'Feature46', 'Feature61',
        'Feature56', 'Feature42', 'Feature39', 'Feature43'
    ]
    
    BEST_9_FEATURES = [
        'Feature63', 'Feature2', 'Feature46', 'Feature61',
        'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48'
    ]
    
    BEST_10_FEATURES = [
        'Feature63', 'Feature2', 'Feature46', 'Feature61', 
        'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48', 'Feature5'
    ]
    
    # 类别特征
    CAT_FEATURE_NAMES = [
        'Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11',
        'Feature45', 'Feature46', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature63'
    ]
    
    def __init__(self, data_root: Optional[str] = None):
        """初始化数据加载器"""
        self.data_root = Path(data_root) if data_root else Path('.')
        self.loaded_datasets = {}
        self.target_column = 'Label'  # 基于settings.py中的LABEL_COL
        
        logger.info(f"医疗数据加载器初始化，数据根目录: {self.data_root}")
    
    def load_dataset(self, dataset_id: str, feature_type: str = 'best10') -> Dict[str, Any]:
        """
        加载指定数据集
        
        Args:
            dataset_id: 数据集ID ('A', 'B', 'C')
            feature_type: 特征集类型 ('best7', 'best10', 'all')
            
        Returns:
            包含X, y, feature_names等信息的字典
        """
        if dataset_id not in self.DATASET_MAPPING:
            raise ValueError(f"不支持的数据集ID: {dataset_id}. 支持的ID: {list(self.DATASET_MAPPING.keys())}")
        
        dataset_name = self.DATASET_MAPPING[dataset_id]
        cache_key = f"{dataset_id}_{feature_type}"
        
        # 检查是否已加载
        if cache_key in self.loaded_datasets:
            logger.info(f"使用缓存的数据集: {dataset_name} ({feature_type})")
            return self.loaded_datasets[cache_key]
        
        logger.info(f"加载数据集: {dataset_name} (ID: {dataset_id}, 特征: {feature_type})")
        
        # 获取数据文件路径
        data_path = Path(self.DATA_PATHS[dataset_id])
        
        # 加载原始数据
        raw_data = self._load_raw_data(data_path)
        
        # 选择特征（使用预定义特征集）
        selected_features = self._get_features_by_type(feature_type)
        
        # 提取特征和标签
        X, y = self._extract_features_and_labels(raw_data, selected_features)
        
        # 数据验证
        self._validate_data(X, y, dataset_name)
        
        # 获取类别特征索引
        categorical_indices = self._get_categorical_indices(selected_features)
        
        # 构建结果
        dataset_info = {
            'X': X,
            'y': y,
            'feature_names': selected_features,
            'categorical_indices': categorical_indices,
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,
            'feature_type': feature_type,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': self._get_class_distribution(y),
            'data_path': str(data_path)
        }
        
        # 缓存结果
        self.loaded_datasets[cache_key] = dataset_info
        
        logger.info(f"数据集加载完成: {dataset_name}")
        logger.info(f"  样本数: {dataset_info['n_samples']}")
        logger.info(f"  特征数: {dataset_info['n_features']}")
        logger.info(f"  类别分布: {dataset_info['class_distribution']}")
        logger.info(f"  类别特征数: {len(categorical_indices)}")
        
        return dataset_info
    
    def load_source_target_data(self, source_id: str = 'A', target_id: str = 'B', 
                               feature_type: str = 'best10') -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        加载源域和目标域数据
        
        Args:
            source_id: 源域数据集ID
            target_id: 目标域数据集ID
            feature_type: 特征集类型
            
        Returns:
            (源域数据信息, 目标域数据信息)
        """
        logger.info(f"加载源域和目标域数据: {source_id} → {target_id}")
        
        source_data = self.load_dataset(source_id, feature_type)
        target_data = self.load_dataset(target_id, feature_type)
        
        # 验证特征一致性
        if source_data['feature_names'] != target_data['feature_names']:
            raise ValueError("源域和目标域的特征不一致")
        
        logger.info(f"源域→目标域数据加载完成: {self.DATASET_MAPPING[source_id]} → {self.DATASET_MAPPING[target_id]}")
        
        return source_data, target_data
    
    def _load_raw_data(self, data_path: Path) -> pd.DataFrame:
        """加载原始数据文件"""
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        try:
            # 加载Excel文件
            if data_path.suffix.lower() == '.xlsx':
                data = pd.read_excel(data_path)
            elif data_path.suffix.lower() == '.csv':
                data = pd.read_csv(data_path)
            else:
                raise ValueError(f"不支持的文件格式: {data_path.suffix}")
            
            logger.info(f"成功加载数据文件: {data_path}")
            logger.info(f"原始数据形状: {data.shape}")
            logger.info(f"原始数据列: {list(data.columns)}")
            
            return data
            
        except Exception as e:
            logger.error(f"加载数据文件失败: {data_path}, 错误: {e}")
            raise
    
    def _get_features_by_type(self, feature_type: str) -> List[str]:
        """根据类型获取特征列表（使用RFE预筛选的特征集）"""
        # 导入统一的特征配置 - 直接导入避免yaml依赖
        try:
            from pathlib import Path
            
            # 直接加载settings.py文件避免__init__.py的yaml依赖
            project_root = Path(__file__).parent.parent
            settings_path = project_root / "config" / "settings.py"
            
            import importlib.util
            spec = importlib.util.spec_from_file_location("settings", settings_path)
            settings = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(settings)
            
            features = settings.get_features_by_type(feature_type)
            if features:
                return features.copy()
        except Exception:
            # 如果导入失败，使用本地定义
            pass
        
        # 本地备份定义（保持向后兼容）
        if feature_type == 'all63':
            return self.ALL_63_FEATURES.copy()
        elif feature_type == 'selected58':
            return self.SELECTED_58_FEATURES.copy()
        elif feature_type == 'best7':
            return self.BEST_7_FEATURES.copy()
        elif feature_type == 'best8':
            return self.BEST_8_FEATURES.copy()
        elif feature_type == 'best9':
            return self.BEST_9_FEATURES.copy()
        elif feature_type == 'best10':
            return self.BEST_10_FEATURES.copy()
        else:
            # 完整支持的特征类型 (best3-best58)
            supported_types = ['all63', 'selected58'] + [f'best{i}' for i in range(3, 59)]
            raise ValueError(f"不支持的特征类型: {feature_type}. 支持的类型包括: all63, selected58, best3-best58")
    
    def _extract_features_and_labels(self, data: pd.DataFrame, 
                                   selected_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """提取特征和标签"""
        if self.target_column not in data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不存在于数据中")
        
        # 检查特征是否存在
        missing_features = [f for f in selected_features if f not in data.columns]
        if missing_features:
            logger.warning(f"以下特征在数据中不存在: {missing_features}")
            # 只使用存在的特征
            selected_features = [f for f in selected_features if f in data.columns]
            logger.info(f"使用存在的特征: {len(selected_features)} 个")
        
        X = data[selected_features].values
        y = data[self.target_column].values
        
        # 处理缺失值
        if np.isnan(X).any():
            logger.warning("特征中存在缺失值，将用均值填充")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        # 确保标签是二分类
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            logger.warning(f"标签不是二分类: {unique_labels}，将进行二值化处理")
            y = (y > np.median(y)).astype(int)
        
        return X, y
    
    def _get_categorical_indices(self, selected_features: List[str]) -> List[int]:
        """获取类别特征在选定特征中的索引"""
        categorical_indices = []
        for i, feature in enumerate(selected_features):
            if feature in self.CAT_FEATURE_NAMES:
                categorical_indices.append(i)
        return categorical_indices
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray, dataset_name: str):
        """验证数据质量"""
        # 检查数据形状
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"特征和标签的样本数不匹配: {X.shape[0]} vs {y.shape[0]}")
        
        if X.shape[0] == 0:
            raise ValueError(f"数据集 {dataset_name} 为空")
        
        # 检查异常值
        if np.isinf(X).any() or np.isnan(X).any():
            raise ValueError(f"特征中包含无穷大或NaN值")
        
        # 检查类别平衡
        class_counts = np.bincount(y)
        imbalance_ratio = np.min(class_counts) / np.max(class_counts)
        if imbalance_ratio < 0.1:
            logger.warning(f"数据集 {dataset_name} 存在严重的类别不平衡: {class_counts}")
        
        logger.info(f"数据验证通过: {dataset_name}")
    
    def _get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """获取类别分布"""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique.astype(str), counts))
    
    def get_available_datasets(self) -> Dict[str, str]:
        """获取可用的数据集"""
        return self.DATASET_MAPPING.copy()
    
    def get_available_feature_types(self) -> List[str]:
        """获取可用的特征类型（RFE预筛选）"""
        return ['all63', 'selected58', 'best7', 'best8', 'best9', 'best10']
    
    def get_categorical_features(self, feature_type: str) -> List[str]:
        """
        获取指定特征集中的类别特征名称
        
        Args:
            feature_type: 特征集类型 ('best7', 'best8', 'best9', 'best10', 'all')
            
        Returns:
            类别特征名称列表
        """
        selected_features = self._get_features_by_type(feature_type)
        categorical_features = []
        
        for feature in selected_features:
            if feature in self.CAT_FEATURE_NAMES:
                categorical_features.append(feature)
        
        return categorical_features


# 便捷函数
def load_medical_data(source_id: str = 'A', target_id: str = 'B', 
                     feature_type: str = 'best10', 
                     data_root: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    便捷的医疗数据加载函数
    
    Args:
        source_id: 源域ID
        target_id: 目标域ID  
        feature_type: 特征类型
        data_root: 数据根目录
        
    Returns:
        (源域数据, 目标域数据)
    """
    loader = MedicalDataLoader(data_root)
    return loader.load_source_target_data(source_id, target_id, feature_type)


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据加载器
    loader = MedicalDataLoader()
    
    print("=" * 60)
    print("医疗数据加载器测试")
    print("=" * 60)
    
    # 测试单个数据集加载
    print("\n1. 测试单个数据集加载")
    try:
        dataset_a = loader.load_dataset('A', 'best7')
        print(f"数据集A加载成功: {dataset_a['dataset_name']}")
        print(f"  形状: {dataset_a['X'].shape}")
        print(f"  特征: {dataset_a['feature_names']}")
        print(f"  类别分布: {dataset_a['class_distribution']}")
        print(f"  类别特征索引: {dataset_a['categorical_indices']}")
    except Exception as e:
        print(f"数据集A加载失败: {e}")
    
    # 测试源域-目标域数据加载
    print("\n2. 测试源域-目标域数据加载")
    try:
        source_data, target_data = loader.load_source_target_data('A', 'B', 'best10')
        print(f"源域-目标域数据加载成功")
        print(f"  源域: {source_data['dataset_name']} {source_data['X'].shape}")
        print(f"  目标域: {target_data['dataset_name']} {target_data['X'].shape}")
    except Exception as e:
        print(f"源域-目标域数据加载失败: {e}")
    
    print("\n" + "=" * 60)
    print("医疗数据加载器测试完成")
    print("=" * 60) 
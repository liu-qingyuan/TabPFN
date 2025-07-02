"""
UDA Medical Imbalance Project - 基于Adapt库的域适应方法

本模块基于adapt-python库实现各种域适应算法，包括：
- 实例重加权方法 (KMM, KLIEP等)
- 特征对齐方法 (CORAL, SA等)
- 深度域适应方法 (DANN, ADDA等)

参考文献:
- Adapt库文档: https://adapt-python.github.io/adapt/
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler  # 不需要标准化

# Adapt库导入
try:
    # 实例重加权方法
    from adapt.instance_based import (
        KMM, KLIEP, LDM, ULSIF, RULSIF, 
        NearestNeighborsWeighting, IWC, IWN
    )
    
    # 特征对齐方法
    from adapt.feature_based import (
        CORAL, SA, TCA, fMMD, PRED,
        DeepCORAL, DANN, ADDA, WDGRL, CDAN, MCD, MDD, CCSA
    )
    
    # 参数迁移方法
    from adapt.parameter_based import (
        RegularTransferLR, RegularTransferLC, RegularTransferNN,
        FineTuning, TransferTreeClassifier, TransferForestClassifier
    )
    
    ADAPT_AVAILABLE = True
    
except ImportError as e:
    logging.warning(f"Adapt库导入失败: {e}")
    ADAPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdaptUDAMethod:
    """基于Adapt库的UDA方法包装器"""
    
    def __init__(self, method_name: str, estimator=None, **params):
        """
        初始化Adapt UDA方法
        
        参数:
        - method_name: 方法名称
        - estimator: 基础分类器
        - **params: 方法特定参数
        """
        if not ADAPT_AVAILABLE:
            raise ImportError("Adapt库未安装，请运行: pip install adapt-python")
        
        self.method_name = method_name.upper()
        self.params = params
        self.estimator = estimator or LogisticRegression(penalty=None, random_state=42)
        # 不使用标准化
        # self.scaler = StandardScaler()
        self.adapt_model = None
        self.is_fitted = False
        
        # 创建Adapt模型
        self._create_adapt_model()
    
    def _create_adapt_model(self):
        """创建Adapt模型实例"""
        method_name = self.method_name
        
        if method_name == 'KMM':
            # Kernel Mean Matching
            # 注意：KMM需要在fit时设置Xt参数，这里先创建基础模型
            # KMM不接受gamma参数，只有kernel参数
            self.adapt_model = KMM(
                estimator=self.estimator,
                kernel=self.params.get('kernel', 'linear'),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'KLIEP':
            # Kullback-Leibler Importance Estimation Procedure
            self.adapt_model = KLIEP(
                estimator=self.estimator,
                gamma=self.params.get('gamma', 1.0),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'CORAL':
            # CORrelation ALignment
            self.adapt_model = CORAL(
                estimator=self.estimator,
                lambda_=self.params.get('lambda_', 1.0),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'SA':
            # Subspace Alignment
            self.adapt_model = SA(
                estimator=self.estimator,
                n_components=self.params.get('n_components', None),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'FMMD':
            # feature-based Maximum Mean Discrepancy
            self.adapt_model = fMMD(
                estimator=self.estimator,
                gamma=self.params.get('gamma', 1.0),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'TCA':
            # Transfer Component Analysis
            self.adapt_model = TCA(
                estimator=self.estimator,
                n_components=self.params.get('n_components', None),
                mu=self.params.get('mu', 1.0),
                kernel=self.params.get('kernel', 'linear'),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'LDM':
            # Linear Discrepancy Minimization
            self.adapt_model = LDM(
                estimator=self.estimator,
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'ULSIF':
            # Unconstrained Least-Squares Importance Fitting
            # 确保参数正确传递给adapt库进行内置优化
            ulsif_params = {
                'estimator': self.estimator,
                'kernel': self.params.get('kernel', 'rbf'),
                'lambdas': self.params.get('lambdas', [0.001, 0.01, 0.1, 1.0, 10.0]),
                'max_centers': self.params.get('max_centers', 100),
                'verbose': self.params.get('verbose', 1),
                'random_state': self.params.get('random_state', 42)
            }
            
            # gamma参数需要直接设置为属性，而不是通过构造函数
            gamma_values = self.params.get('gamma', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            
            self.adapt_model = ULSIF(**ulsif_params)
            # 手动设置gamma参数以确保交叉验证生效
            self.adapt_model.gamma = gamma_values
            
        elif method_name == 'RULSIF':
            # Relative Unconstrained Least-Squares Importance Fitting
            # 确保参数正确传递给adapt库进行内置优化
            rulsif_params = {
                'estimator': self.estimator,
                'kernel': self.params.get('kernel', 'rbf'),
                'alpha': self.params.get('alpha', 0.1),
                'lambdas': self.params.get('lambdas', [0.001, 0.01, 0.1, 1.0, 10.0]),
                'max_centers': self.params.get('max_centers', 100),
                'verbose': self.params.get('verbose', 1),
                'random_state': self.params.get('random_state', 42)
            }
            
            # gamma参数需要直接设置为属性，而不是通过构造函数
            gamma_values = self.params.get('gamma', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            
            self.adapt_model = RULSIF(**rulsif_params)
            # 手动设置gamma参数以确保交叉验证生效
            self.adapt_model.gamma = gamma_values
            
        elif method_name == 'NNW':
            # Nearest Neighbors Weighting
            self.adapt_model = NearestNeighborsWeighting(
                estimator=self.estimator,
                n_neighbors=self.params.get('n_neighbors', 5),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'IWC':
            # Importance Weighting Classifier
            # 使用传入的estimator作为classifier（域分类器）
            import copy
            classifier_estimator = copy.deepcopy(self.estimator)
            
            self.adapt_model = IWC(
                estimator=self.estimator,
                classifier=classifier_estimator,  # 使用我们的estimator作为classifier
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'IWN':
            # Importance Weighting Network
            # IWN是基于TensorFlow的深度学习方法，需要编译
            # 即使estimator是sklearn模型，IWN内部的weighter仍然是TensorFlow网络
            try:
                import tensorflow as tf
                
                self.adapt_model = IWN(
                    estimator=self.estimator,
                    weighter=self.params.get('weighter', None),  # 默认会创建TF网络
                    pretrain=self.params.get('pretrain', True),
                    sigma_init=self.params.get('sigma_init', 0.1),
                    update_sigma=self.params.get('update_sigma', True),
                    verbose=self.params.get('verbose', 0),
                    random_state=self.params.get('random_state', 42)
                )
                
                # IWN必须编译才能使用，因为它继承自BaseAdaptDeep
                # 在TF 2.15+中需要使用legacy optimizer来避免变量识别问题
                try:
                    # 首先尝试使用legacy optimizer（推荐用于adapt库）
                    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
                except AttributeError:
                    # 如果legacy不可用，尝试标准optimizer并构建变量
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                
                self.adapt_model.compile(
                    optimizer=optimizer,
                    loss=None,  # IWN使用自定义MMD损失函数
                    metrics=None
                )
                logger.info("IWN模型编译成功")
                
            except ImportError:
                logger.error("IWN需要TensorFlow支持，但TensorFlow不可用")
                # 回退到其他方法或跳过
                raise ImportError("IWN方法需要TensorFlow支持")
            except Exception as e:
                logger.error(f"IWN初始化或编译失败: {e}")
                raise
            

            
        elif method_name == 'PRED':
            # Feature Augmentation with SrcOnly Prediction
            # PRED需要特殊处理，因为它需要目标域数据和标签
            self.adapt_model = PRED(
                estimator=self.estimator,
                pretrain=True,  # 需要预训练源域模型
                copy=True,
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'DANN':
            # Domain-Adversarial Neural Networks
            self.adapt_model = DANN(
                lambda_=self.params.get('lambda_', 1.0),
                lr=self.params.get('lr', 0.001),
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'ADDA':
            # Adversarial Discriminative Domain Adaptation
            self.adapt_model = ADDA(
                lambda_=self.params.get('lambda_', 1.0),
                lr=self.params.get('lr', 0.001),
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'WDGRL':
            # Wasserstein Distance Guided Representation Learning
            self.adapt_model = WDGRL(
                lambda_=self.params.get('lambda_', 1.0),
                lr=self.params.get('lr', 0.001),
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
        elif method_name == 'DEEPCORAL':
            # Deep CORAL
            self.adapt_model = DeepCORAL(
                lambda_=self.params.get('lambda_', 1.0),
                lr=self.params.get('lr', 0.001),
                epochs=self.params.get('epochs', 100),
                batch_size=self.params.get('batch_size', 32),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            

            
        else:
            raise ValueError(f"不支持的Adapt方法: {method_name}")
    
    def fit(self, X_source: np.ndarray, y_source: np.ndarray, 
            X_target: np.ndarray, y_target: Optional[np.ndarray] = None):
        """
        拟合域适应模型
        
        参数:
        - X_source: 源域特征
        - y_source: 源域标签
        - X_target: 目标域特征
        - y_target: 目标域标签（可选，用于监督方法）
        """
        logger.info(f"开始拟合{self.method_name}域适应模型...")
        
        # 不使用标准化，直接使用原始数据
        X_source_scaled = X_source
        X_target_scaled = X_target
        
        # 特殊处理KMM方法
        if self.method_name == 'KMM':
            # KMM不需要gamma参数优化，直接设置目标域数据并拟合
            self.adapt_model.Xt = X_target_scaled
            self.adapt_model.fit(X_source_scaled, y_source)
        elif self.method_name in ['ULSIF', 'RULSIF']:
            # ULSIF和RULSIF使用内置优化，只需设置目标域数据
            self.adapt_model.Xt = X_target_scaled
            self.adapt_model.fit(X_source_scaled, y_source)
        elif self.method_name == 'PRED':
            # PRED方法需要特殊处理：它是基于特征增强的方法
            # 先设置目标域数据
            self.adapt_model.Xt = X_target_scaled
            
            # PRED方法需要目标域数据进行特征增强，但不一定需要目标域标签
            # 根据PRED源码，它会先训练源域模型，然后用其预测来增强目标域特征
            
            # 直接调用fit方法，PRED内部会处理特征增强逻辑
            try:
                # PRED的fit方法接受源域和目标域数据
                self.adapt_model.fit(X_source_scaled, y_source, X_target_scaled)
            except Exception as e:
                logger.warning(f"PRED fit失败，尝试备用方法: {e}")
                # 备用方法：手动设置一些伪标签
                import copy
                temp_estimator = copy.deepcopy(self.estimator)
                temp_estimator.fit(X_source_scaled, y_source)
                y_target_pseudo = temp_estimator.predict(X_target_scaled)
                self.adapt_model.yt = y_target_pseudo
                self.adapt_model.fit(X_source_scaled, y_source)
        elif self.method_name == 'IWN':
            # IWN方法需要特殊处理：它是基于TensorFlow的深度学习方法
            # IWN使用fit_weights和fit_estimator的两阶段训练
            try:
                import tensorflow as tf
                
                # 设置目标域数据
                self.adapt_model.Xt = X_target_scaled
                
                # IWN的训练参数
                epochs = self.params.get('epochs', 50)  # 减少epochs避免训练时间过长
                batch_size = self.params.get('batch_size', 256)
                
                # 在TF 2.15+中，确保optimizer正确构建变量
                # 首先进行一次前向传播来初始化所有变量
                logger.info("IWN初始化变量...")
                dummy_batch_size = min(32, X_source_scaled.shape[0])
                dummy_source = X_source_scaled[:dummy_batch_size]
                dummy_target = X_target_scaled[:dummy_batch_size] if X_target_scaled.shape[0] >= dummy_batch_size else X_target_scaled
                
                # 手动触发一次前向传播来构建变量
                try:
                    with tf.GradientTape():
                        _ = self.adapt_model(dummy_source, training=True)
                    logger.info("IWN变量初始化成功")
                except Exception as init_error:
                    logger.warning(f"IWN变量初始化警告: {init_error}")
                
                # 第一阶段：训练权重网络
                logger.info("IWN第一阶段：训练权重网络...")
                weights = self.adapt_model.fit_weights(
                    X_source_scaled, X_target_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=self.params.get('verbose', 0)
                )
                
                # 第二阶段：使用权重训练估计器
                logger.info("IWN第二阶段：训练估计器...")
                self.adapt_model.fit_estimator(
                    X_source_scaled, y_source,
                    sample_weight=weights,
                    random_state=self.params.get('random_state', 42)
                )
                
                logger.info("IWN训练完成")
                
            except Exception as e:
                logger.error(f"IWN训练失败: {e}")
                # 如果是optimizer相关错误，尝试重新编译
                if "optimizer" in str(e).lower() or "variable" in str(e).lower():
                    logger.warning("检测到optimizer兼容性问题，尝试重新编译...")
                    try:
                        import tensorflow as tf
                        # 使用legacy optimizer重新编译
                        legacy_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
                        self.adapt_model.compile(
                            optimizer=legacy_optimizer,
                            loss=None,
                            metrics=None
                        )
                        logger.info("使用legacy optimizer重新编译成功")
                        # 重新尝试fit
                        self.adapt_model.fit(X_source_scaled, y_source, Xt=X_target_scaled)
                    except Exception as recompile_error:
                        logger.error(f"重新编译也失败: {recompile_error}")
                        raise e
                else:
                    # 对于其他错误，回退到普通fit方法
                    logger.warning("尝试使用IWN的标准fit方法...")
                    self.adapt_model.fit(X_source_scaled, y_source, Xt=X_target_scaled)
        else:
            # 对于需要目标域数据的方法，设置Xt参数
            if hasattr(self.adapt_model, 'Xt'):
                self.adapt_model.Xt = X_target_scaled
            
            # 拟合模型
            if self.method_name in ['DANN', 'ADDA', 'WDGRL', 'DEEPCORAL']:
                # 深度学习方法需要目标域数据
                self.adapt_model.fit(X_source_scaled, y_source, X_target_scaled)
            elif self.method_name in ['TCA', 'SA', 'FMMD', 'CORAL']:
                # 特征变换方法需要同时接收源域和目标域数据
                logger.info(f"拟合{self.method_name}特征变换方法，同时使用源域和目标域数据")
                try:
                    # 这些方法需要源域和目标域数据来学习特征变换
                    self.adapt_model.fit(X_source_scaled, y_source, X_target_scaled)
                except Exception as e:
                    logger.warning(f"{self.method_name}标准fit方法失败: {e}")
                    # 某些方法可能只接受两个参数
                    logger.info(f"尝试使用{self.method_name}的备用fit方法")
                    self.adapt_model.fit(X_source_scaled, y_source)
            else:
                # 其他方法（如KMM、KLIEP等实例重加权方法）
                self.adapt_model.fit(X_source_scaled, y_source)
        
        self.is_fitted = True
        logger.info(f"{self.method_name}域适应模型拟合完成")
        return self
    


    
    def predict(self, X_target: np.ndarray) -> np.ndarray:
        """预测目标域标签"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用fit方法")
        
        # 不使用标准化，直接使用原始数据
        X_target_scaled = X_target
        
        # 对于特征变换方法，需要先变换特征再预测
        if self.method_name in ['FMMD', 'SA', 'TCA', 'PRED']:
            try:
                # 使用transform方法获取变换后的特征
                if hasattr(self.adapt_model, 'transform'):
                    # SA方法需要特殊处理domain参数
                    if self.method_name == 'SA':
                        # 对于目标域数据，使用domain="tgt"（默认值）
                        X_transformed = self.adapt_model.transform(X_target_scaled, domain="tgt")
                    else:
                        X_transformed = self.adapt_model.transform(X_target_scaled)
                    
                    logger.debug(f"{self.method_name}特征变换: {X_target_scaled.shape} -> {X_transformed.shape}")
                    
                    # 然后使用estimator_进行预测
                    if hasattr(self.adapt_model, 'estimator_'):
                        pred = self.adapt_model.estimator_.predict(X_transformed)
                        logger.debug(f"{self.method_name}变换后预测成功")
                        return pred
            except Exception as e:
                logger.warning(f"{self.method_name}特征变换预测失败: {e}")
                # 回退到直接预测
                pass
        
        # 对于其他方法或变换失败的情况，直接使用adapt_model的predict方法
        return self.adapt_model.predict(X_target_scaled)
    
    def predict_proba(self, X_target: np.ndarray) -> Optional[np.ndarray]:
        """预测目标域标签概率"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用fit方法")
        
        # 不使用标准化，直接使用原始数据
        X_target_scaled = X_target
        
        # 方法1: 直接调用adapt_model的predict_proba（如果存在）
        try:
            # SA方法需要指定domain参数
            if self.method_name == 'SA':
                # SA方法可能没有predict_proba，跳过此方法
                pass
            else:
                return self.adapt_model.predict_proba(X_target_scaled)
        except (AttributeError, NotImplementedError, TypeError):
            pass
        
        # 方法2: 对于特征变换方法，先变换特征再预测概率
        if self.method_name in ['FMMD', 'SA', 'TCA', 'PRED']:
            try:
                # 使用transform方法获取变换后的特征
                if hasattr(self.adapt_model, 'transform'):
                    # SA方法需要特殊处理domain参数
                    if self.method_name == 'SA':
                        # 对于目标域数据，使用domain="tgt"（默认值）
                        X_transformed = self.adapt_model.transform(X_target_scaled, domain="tgt")
                    else:
                        X_transformed = self.adapt_model.transform(X_target_scaled)
                    
                    logger.debug(f"{self.method_name}特征变换: {X_target_scaled.shape} -> {X_transformed.shape}")
                    
                    # 然后使用estimator_进行概率预测
                    if hasattr(self.adapt_model, 'estimator_') and hasattr(self.adapt_model.estimator_, 'predict_proba'):
                        proba = self.adapt_model.estimator_.predict_proba(X_transformed)
                        logger.debug(f"{self.method_name}变换后概率预测成功: {proba.shape}")
                        return proba
            except Exception as e:
                logger.warning(f"{self.method_name}特征变换概率预测失败: {e}")
                pass
        
        # 方法3: 通过estimator_属性获取概率预测（未变换特征的情况）
        if hasattr(self.adapt_model, 'estimator_'):
            try:
                estimator = self.adapt_model.estimator_
                if hasattr(estimator, 'predict_proba'):
                    return estimator.predict_proba(X_target_scaled)
            except Exception:
                pass
        
        # 方法4: 对于实例重加权方法，使用训练后的estimator_进行预测
        # 这些方法通过重新加权训练数据，训练后的estimator_包含了域适应信息
        if self.method_name in ['KMM', 'KLIEP', 'LDM', 'ULSIF', 'RULSIF', 'NNW', 'IWC', 'IWN']:
            try:
                if hasattr(self.adapt_model, 'estimator_') and hasattr(self.adapt_model.estimator_, 'predict_proba'):
                    return self.adapt_model.estimator_.predict_proba(X_target_scaled)
            except Exception:
                pass
        
        # 方法5: 使用decision_function转换为概率
        try:
            if hasattr(self.adapt_model, 'decision_function'):
                decision_scores = self.adapt_model.decision_function(X_target_scaled)
                from scipy.special import expit
                probas = expit(decision_scores)
                return np.column_stack([1 - probas, probas])
            elif hasattr(self.adapt_model, 'estimator_') and hasattr(self.adapt_model.estimator_, 'decision_function'):
                decision_scores = self.adapt_model.estimator_.decision_function(X_target_scaled)
                from scipy.special import expit
                probas = expit(decision_scores)
                return np.column_stack([1 - probas, probas])
        except Exception:
            pass
        
        # 如果所有方法都失败，返回None
        logger.warning(f"{self.method_name}方法不支持概率预测，AUC将无法计算")
        return None
    
    def score(self, X_target: np.ndarray, y_target: np.ndarray) -> float:
        """计算目标域准确率"""
        if not self.is_fitted:
            raise ValueError("模型未拟合，请先调用fit方法")
        
        # 不使用标准化，直接使用原始数据
        X_target_scaled = X_target
        return self.adapt_model.score(X_target_scaled, y_target)
    
    def get_weights(self) -> Optional[np.ndarray]:
        """获取实例权重（如果适用）"""
        if self.method_name == 'KMM':
            # KMM特殊处理：使用fit_weights方法
            try:
                if hasattr(self.adapt_model, 'fit_weights'):
                    # 需要重新获取原始数据进行权重计算
                    # 这里返回None，建议用户直接调用fit_weights方法
                    logger.info("KMM权重需要通过fit_weights方法获取，请参考文档")
                    return None
            except Exception:
                pass
        
        # 其他方法的权重获取
        if hasattr(self.adapt_model, 'predict_weights'):
            return self.adapt_model.predict_weights()
        return None
    
    def fit_weights(self, X_source: np.ndarray, X_target: np.ndarray) -> Optional[np.ndarray]:
        """
        专门用于KMM的权重计算方法
        
        参数:
        - X_source: 源域特征
        - X_target: 目标域特征
        
        返回:
        - weights: 实例权重数组
        """
        if self.method_name != 'KMM':
            logger.warning("fit_weights方法仅适用于KMM")
            return None
        
        try:
            # 不使用标准化，直接使用原始数据
            X_source_scaled = X_source
            X_target_scaled = X_target
            
            # 创建专门用于权重计算的KMM模型
            # KMM不接受gamma参数，只有kernel参数
            kmm_weights = KMM(
                estimator=self.estimator,
                Xt=X_target_scaled,
                kernel=self.params.get('kernel', 'linear'),
                verbose=self.params.get('verbose', 0),
                random_state=self.params.get('random_state', 42)
            )
            
            # 计算权重
            weights = kmm_weights.fit_weights(X_source_scaled, X_target_scaled)
            logger.info(f"KMM权重计算完成，权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
            return weights
            
        except Exception as e:
            logger.error(f"KMM权重计算失败: {e}")
            return None


class AdaptUDAFactory:
    """基于Adapt库的UDA方法工厂"""
    
    # 支持的方法及其默认参数
    SUPPORTED_METHODS = {
        'KMM': {
            'description': 'Kernel Mean Matching - 核均值匹配',
            'type': 'instance_based',
            'default_params': {
                'kernel': 'linear',       # KMM核函数类型（不需要gamma参数）
                'B': 1000,                # 权重边界
                'eps': None,              # 自动计算约束参数
                'max_size': 1000,         # 批处理大小
                'tol': 1e-6,              # 优化阈值
                'max_iter': 100,          # 最大迭代次数
                'verbose': 0,
                'random_state': 42
            }
        },
        'KLIEP': {
            'description': 'Kullback-Leibler Importance Estimation Procedure',
            'type': 'instance_based',
            'default_params': {
                'gamma': 1.0,
                'verbose': 0,
                'random_state': 42
            }
        },
        'CORAL': {
            'description': 'CORrelation ALignment - 相关性对齐',
            'type': 'feature_based',
            'default_params': {
                'lambda_': 1.0,
                'verbose': 0,
                'random_state': 42
            }
        },
        'SA': {
            'description': 'Subspace Alignment - 子空间对齐',
            'type': 'feature_based',
            'default_params': {
                'n_components': None,
                'verbose': 0,
                'random_state': 42
            }
        },
        'FMMD': {
            'description': 'Feature-based Maximum Mean Discrepancy',
            'type': 'feature_based',
            'default_params': {
                'gamma': 1.0,
                'verbose': 0,
                'random_state': 42
            }
        },
        'DANN': {
            'description': 'Domain-Adversarial Neural Networks',
            'type': 'deep_learning',
            'default_params': {
                'lambda_': 1.0,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                'random_state': 42
            }
        },
        'ADDA': {
            'description': 'Adversarial Discriminative Domain Adaptation',
            'type': 'deep_learning',
            'default_params': {
                'lambda_': 1.0,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                'random_state': 42
            }
        },
        'WDGRL': {
            'description': 'Wasserstein Distance Guided Representation Learning',
            'type': 'deep_learning',
            'default_params': {
                'lambda_': 1.0,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                'random_state': 42
            }
        },
        'DEEPCORAL': {
            'description': 'Deep CORAL',
            'type': 'deep_learning',
            'default_params': {
                'lambda_': 1.0,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                'random_state': 42
            }
        },

        'TCA': {
            'description': 'Transfer Component Analysis - 迁移成分分析',
            'type': 'feature_based',
            'default_params': {
                'n_components': None,
                'mu': 1.0,
                'kernel': 'linear',
                'verbose': 0,
                'random_state': 42
            }
        },
        'LDM': {
            'description': 'Linear Discrepancy Minimization',
            'type': 'instance_based',
            'default_params': {
                'verbose': 0,
                'random_state': 42
            }
        },
        'ULSIF': {
            'description': 'Unconstrained Least-Squares Importance Fitting',
            'type': 'instance_based',
            'default_params': {
                'gamma': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # gamma列表，算法自动选择最佳值
                'lambdas': [0.001, 0.01, 0.1, 1.0, 10.0],       # lambdas列表，算法自动选择最佳值
                'max_centers': 100,    # 核函数中心数量
                'kernel': 'rbf',       # 核函数类型
                'verbose': 1,          # 开启详细输出以查看优化过程
                'random_state': 42
            }
        },
        'RULSIF': {
            'description': 'Relative Unconstrained Least-Squares Importance Fitting',
            'type': 'instance_based',
            'default_params': {
                'gamma': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # gamma列表，算法自动选择最佳值
                'lambdas': [0.001, 0.01, 0.1, 1.0, 10.0],       # lambdas列表，算法自动选择最佳值
                'alpha': 0.1,          # 相对重要性参数
                'max_centers': 100,    # 核函数中心数量
                'kernel': 'rbf',       # 核函数类型
                'verbose': 1,          # 开启详细输出以查看优化过程
                'random_state': 42
            }
        },
        'NNW': {
            'description': 'Nearest Neighbors Weighting',
            'type': 'instance_based',
            'default_params': {
                'n_neighbors': 5,
                'verbose': 0,
                'random_state': 42
            }
        },
        'IWC': {
            'description': 'Importance Weighting Classifier',
            'type': 'instance_based',
            'default_params': {
                'verbose': 0,
                'random_state': 42
            }
        },
        'IWN': {
            'description': 'Importance Weighting Network',
            'type': 'instance_based',
            'default_params': {
                'weighter': None,
                'pretrain': True,
                'sigma_init': 0.1,
                'update_sigma': True,
                'batch_size': 256,
                'verbose': 0,
                'random_state': 42
            }
        },

        'PRED': {
            'description': 'Feature Augmentation with SrcOnly Prediction - 仅使用源域预测的特征增强',
            'type': 'feature_based',
            'default_params': {
                'verbose': 0,
                'random_state': 42
            }
        },
        'CDAN': {
            'description': 'Conditional Adversarial Domain Adaptation',
            'type': 'deep_learning',
            'default_params': {
                'lambda_': 1.0,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                'random_state': 42
            }
        },
        'MCD': {
            'description': 'Maximum Classifier Discrepancy',
            'type': 'deep_learning',
            'default_params': {
                'lambda_': 1.0,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                'random_state': 42
            }
        },
        'MDD': {
            'description': 'Margin Disparity Discrepancy',
            'type': 'deep_learning',
            'default_params': {
                'lambda_': 1.0,
                'lr': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                'random_state': 42
            }
        }
    }
    
    @classmethod
    def create_method(cls, method_name: str, estimator=None, **params) -> AdaptUDAMethod:
        """创建Adapt UDA方法实例"""
        if not ADAPT_AVAILABLE:
            raise ImportError("Adapt库未安装，请运行: pip install adapt-python")
        
        method_name = method_name.upper()
        
        if method_name not in cls.SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的方法: {method_name}. "
                f"支持的方法: {list(cls.SUPPORTED_METHODS.keys())}"
            )
        
        # 合并默认参数
        default_params = cls.SUPPORTED_METHODS[method_name]['default_params']
        merged_params = {**default_params, **params}
        
        return AdaptUDAMethod(method_name, estimator, **merged_params)
    
    @classmethod
    def get_available_methods(cls) -> Dict[str, Dict[str, str]]:
        """获取可用方法列表"""
        if not ADAPT_AVAILABLE:
            return {}
        
        return {
            method: {
                'description': info['description'],
                'type': info['type']
            }
            for method, info in cls.SUPPORTED_METHODS.items()
        }
    
    @classmethod
    def get_methods_by_type(cls, method_type: str) -> Dict[str, str]:
        """根据类型获取方法"""
        if not ADAPT_AVAILABLE:
            return {}
        
        return {
            method: info['description']
            for method, info in cls.SUPPORTED_METHODS.items()
            if info['type'] == method_type
        }


# 便捷函数
def create_adapt_method(method_name: str, estimator=None, **params) -> AdaptUDAMethod:
    """创建Adapt UDA方法的便捷函数"""
    return AdaptUDAFactory.create_method(method_name, estimator, **params)


def get_available_adapt_methods() -> Dict[str, Dict[str, str]]:
    """获取可用Adapt方法的便捷函数"""
    return AdaptUDAFactory.get_available_methods()


def is_adapt_available() -> bool:
    """检查Adapt库是否可用"""
    return ADAPT_AVAILABLE


if __name__ == "__main__":
    # 测试示例
    if ADAPT_AVAILABLE:
        print("Adapt库可用")
        print("支持的方法:")
        for method, info in get_available_adapt_methods().items():
            print(f"  {method}: {info['description']} ({info['type']})")
    else:
        print("Adapt库不可用，请安装: pip install adapt-python") 
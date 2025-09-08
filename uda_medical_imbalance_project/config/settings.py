"""
UDA Medical Imbalance Project - 配置设置

包含预定义的特征集、类别特征和数据路径配置
基于RFE预筛选的最优特征组合
"""

import logging
from typing import List, Dict, Any

# 全部63个特征（Feature1到Feature63）
ALL_63_FEATURES = [
    'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20',
    'Feature21', 'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30',
    'Feature31', 'Feature32', 'Feature33', 'Feature34', 'Feature35', 'Feature36', 'Feature37', 'Feature38', 'Feature39', 'Feature40',
    'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45', 'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50',
    'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60',
    'Feature61', 'Feature62', 'Feature63'
]

# 经过RFE筛选的58个特征（移除了Feature12, Feature33, Feature34, Feature36, Feature40）
SELECTED_58_FEATURES = [
    'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20', 'Feature21',
    'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31',
    'Feature32', 'Feature35', 'Feature37', 'Feature38', 'Feature39', 'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45',
    'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55',
    'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61', 'Feature62', 'Feature63'
]

# 为了向后兼容，保留原名称
SELECTED_FEATURES = SELECTED_58_FEATURES

# 基于58个共同特征的RFE结果 (来源: feature_selection_evaluation_58features_20250906_003222)

# 最佳3特征配置 (AUC: 0.7795)
BEST_3_FEATURES = [
    'Feature63', 'Feature2', 'Feature46'
]

# 最佳4特征配置 (AUC: 0.7818)
BEST_4_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57'
]

# 最佳5特征配置 (AUC: 0.8023)
BEST_5_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61'
]

# 最佳6特征配置 (AUC: 0.8127)
BEST_6_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42'
]

# 最佳7特征配置 (AUC: 0.8270)
BEST_7_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
]

# 最佳8特征配置 (AUC: 0.8328)
BEST_8_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5'
]

# 最佳9特征配置 (AUC: 0.8393)
BEST_9_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5', 'Feature49'
]

# 最佳10特征配置 (AUC: 0.8414)
BEST_10_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5', 'Feature49', 'Feature48'
]

# 最佳11特征配置 (AUC: 0.8397)
BEST_11_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5', 'Feature49', 'Feature48', 'Feature22'
]

# 最佳12特征配置 (AUC: 0.8339)
BEST_12_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5', 'Feature49', 'Feature48', 'Feature22', 'Feature56'
]

# 最佳15特征配置 (AUC: 0.8299) - 高性能选择
BEST_15_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5', 'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44'
]

# 最佳20特征配置 (AUC: 0.8278) - 平衡选择
BEST_20_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5', 'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27', 'Feature52', 'Feature55', 'Feature47', 'Feature29'
]

# 最佳32特征配置 (AUC: 0.8207) - 扩展选择
BEST_32_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5', 'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27', 'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62', 'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19'
]

# 最佳13特征配置 (AUC: 0.8343)
BEST_13_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32'
]

# 最佳14特征配置 (AUC: 0.8317)
BEST_14_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17'
]

# 最佳16特征配置 (AUC: 0.8336)
BEST_16_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27'
]

# 最佳17特征配置 (AUC: 0.8317)
BEST_17_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52'
]

# 最佳18特征配置 (AUC: 0.8274)
BEST_18_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55'
]

# 最佳19特征配置 (AUC: 0.8320)
BEST_19_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47'
]

# 最佳21特征配置 (AUC: 0.8141)
BEST_21_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60'
]

# 最佳22特征配置 (AUC: 0.8184)
BEST_22_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14'
]

# 最佳23特征配置 (AUC: 0.8156)
BEST_23_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39'
]

# 最佳24特征配置 (AUC: 0.8159)
BEST_24_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62'
]

# 最佳25特征配置 (AUC: 0.8132)
BEST_25_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38'
]

# 最佳26特征配置 (AUC: 0.8164)
BEST_26_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7'
]

# 最佳27特征配置 (AUC: 0.8179)
BEST_27_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8'
]

# 最佳28特征配置 (AUC: 0.8200)
BEST_28_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58'
]

# 最佳29特征配置 (AUC: 0.8200)
BEST_29_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25'
]

# 最佳30特征配置 (AUC: 0.8197)
BEST_30_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23'
]

# 最佳31特征配置 (AUC: 0.8195)
BEST_31_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3'
]

# 最佳33特征配置 (AUC: 0.8168)
BEST_33_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35'
]

# 最佳34特征配置 (AUC: 0.8104)
BEST_34_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1'
]

# 最佳35特征配置 (AUC: 0.8095)
BEST_35_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53'
]

# 最佳36特征配置 (AUC: 0.8098)
BEST_36_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24'
]

# 最佳37特征配置 (AUC: 0.8097)
BEST_37_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24',
    'Feature30'
]

# 最佳38特征配置 (AUC: 0.8071)
BEST_38_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24',
    'Feature30', 'Feature21'
]

# 最佳39特征配置 (AUC: 0.8049)
BEST_39_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24',
    'Feature30', 'Feature21', 'Feature9'
]

# 最佳40特征配置 (AUC: 0.8089)
BEST_40_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24',
    'Feature30', 'Feature21', 'Feature9', 'Feature16'
]

# 最佳41特征配置 (AUC: 0.8091)
BEST_41_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24',
    'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59'
]

# 最佳42特征配置 (AUC: 0.8045)
BEST_42_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57',
    'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56',
    'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29',
    'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58',
    'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24',
    'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45'
]

# 最佳43特征配置 (AUC: 0.8036)
BEST_43_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26'
]

# 最佳44特征配置 (AUC: 0.7992)
BEST_44_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31'
]

# 最佳45特征配置 (AUC: 0.7967)
BEST_45_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20'
]

# 最佳46特征配置 (AUC: 0.8022)
BEST_46_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13'
]

# 最佳47特征配置 (AUC: 0.7979)
BEST_47_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4'
]

# 最佳48特征配置 (AUC: 0.7999)
BEST_48_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10'
]

# 最佳49特征配置 (AUC: 0.7971)
BEST_49_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6'
]

# 最佳50特征配置 (AUC: 0.7981)
BEST_50_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54'
]

# 最佳51特征配置 (AUC: 0.7976)
BEST_51_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18'
]

# 最佳52特征配置 (AUC: 0.7984)
BEST_52_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18', 'Feature28'
]

# 最佳53特征配置 (AUC: 0.8026)
BEST_53_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18', 'Feature28', 'Feature51'
]

# 最佳54特征配置 (AUC: 0.8009)
BEST_54_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18', 'Feature28', 'Feature51', 'Feature37'
]

# 最佳55特征配置 (AUC: 0.7954)
BEST_55_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18', 'Feature28', 'Feature51', 'Feature37', 'Feature41'
]

# 最佳56特征配置 (AUC: 0.7974)
BEST_56_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18', 'Feature28', 'Feature51', 'Feature37', 'Feature41', 'Feature15'
]

# 最佳57特征配置 (AUC: 0.7921)
BEST_57_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18', 'Feature28', 'Feature51', 'Feature37', 'Feature41', 'Feature15',
    'Feature50'
]

# 最佳58特征配置 (AUC: 0.7261) - 基于RFE排序的最优58特征
BEST_58_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature57', 'Feature61', 'Feature42', 'Feature43', 'Feature5',
    'Feature49', 'Feature48', 'Feature22', 'Feature56', 'Feature32', 'Feature17', 'Feature44', 'Feature27',
    'Feature52', 'Feature55', 'Feature47', 'Feature29', 'Feature60', 'Feature14', 'Feature39', 'Feature62',
    'Feature38', 'Feature7', 'Feature8', 'Feature58', 'Feature25', 'Feature23', 'Feature3', 'Feature19',
    'Feature35', 'Feature1', 'Feature53', 'Feature24', 'Feature30', 'Feature21', 'Feature9', 'Feature16',
    'Feature59', 'Feature45', 'Feature26', 'Feature31', 'Feature20', 'Feature13', 'Feature4', 'Feature10',
    'Feature6', 'Feature54', 'Feature18', 'Feature28', 'Feature51', 'Feature37', 'Feature41', 'Feature15',
    'Feature50', 'Feature11'
]

# 选定58特征 (包含所有A、B数据集共同特征) - 保持向后兼容
SELECTED_58_FEATURES_FULL = SELECTED_58_FEATURES

# 类别特征名称 (20个类别特征)
CAT_FEATURE_NAMES = [
    'Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11',
    'Feature45', 'Feature46', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature63'
]

# 数据文件路径
DATA_PATHS = {
    'A': "/home/24052432g/TabPFN/data/AI4healthcare.xlsx",
    'B': "/home/24052432g/TabPFN/data/HenanCancerHospital_features63_58.xlsx",
    'C': "/home/24052432g/TabPFN/data/GuangzhouMedicalHospital_features23_no_nan_new_fixed.xlsx"
}

# 标签列名
LABEL_COL = "Label"

# 数据集映射
DATASET_MAPPING = {
    'A': 'AI4health',
    'B': 'HenanCancerHospital', 
    'C': 'GuangzhouMedicalHospital'
}

def get_features_by_type(feature_type: str = 'best10') -> List[str]:
    """
    根据类型获取特征列表
    
    Args:
        feature_type: 特征集类型 ('all63', 'selected58', 'best3', 'best4', ..., 'best58')
        
    Returns:
        特征名称列表
    """
    if feature_type == 'all63':
        return ALL_63_FEATURES.copy()
    elif feature_type == 'selected58':
        return SELECTED_58_FEATURES.copy()
    elif feature_type == 'best3':
        return BEST_3_FEATURES.copy()
    elif feature_type == 'best4':
        return BEST_4_FEATURES.copy()
    elif feature_type == 'best5':
        return BEST_5_FEATURES.copy()
    elif feature_type == 'best6':
        return BEST_6_FEATURES.copy()
    elif feature_type == 'best7':
        return BEST_7_FEATURES.copy()
    elif feature_type == 'best8':
        return BEST_8_FEATURES.copy()
    elif feature_type == 'best9':
        return BEST_9_FEATURES.copy()
    elif feature_type == 'best10':
        return BEST_10_FEATURES.copy()
    elif feature_type == 'best11':
        return BEST_11_FEATURES.copy()
    elif feature_type == 'best12':
        return BEST_12_FEATURES.copy()
    elif feature_type == 'best13':
        return BEST_13_FEATURES.copy()
    elif feature_type == 'best14':
        return BEST_14_FEATURES.copy()
    elif feature_type == 'best15':
        return BEST_15_FEATURES.copy()
    elif feature_type == 'best16':
        return BEST_16_FEATURES.copy()
    elif feature_type == 'best17':
        return BEST_17_FEATURES.copy()
    elif feature_type == 'best18':
        return BEST_18_FEATURES.copy()
    elif feature_type == 'best19':
        return BEST_19_FEATURES.copy()
    elif feature_type == 'best20':
        return BEST_20_FEATURES.copy()
    elif feature_type == 'best21':
        return BEST_21_FEATURES.copy()
    elif feature_type == 'best22':
        return BEST_22_FEATURES.copy()
    elif feature_type == 'best23':
        return BEST_23_FEATURES.copy()
    elif feature_type == 'best24':
        return BEST_24_FEATURES.copy()
    elif feature_type == 'best25':
        return BEST_25_FEATURES.copy()
    elif feature_type == 'best26':
        return BEST_26_FEATURES.copy()
    elif feature_type == 'best27':
        return BEST_27_FEATURES.copy()
    elif feature_type == 'best28':
        return BEST_28_FEATURES.copy()
    elif feature_type == 'best29':
        return BEST_29_FEATURES.copy()
    elif feature_type == 'best30':
        return BEST_30_FEATURES.copy()
    elif feature_type == 'best31':
        return BEST_31_FEATURES.copy()
    elif feature_type == 'best32':
        return BEST_32_FEATURES.copy()
    elif feature_type == 'best33':
        return BEST_33_FEATURES.copy()
    elif feature_type == 'best34':
        return BEST_34_FEATURES.copy()
    elif feature_type == 'best35':
        return BEST_35_FEATURES.copy()
    elif feature_type == 'best36':
        return BEST_36_FEATURES.copy()
    elif feature_type == 'best37':
        return BEST_37_FEATURES.copy()
    elif feature_type == 'best38':
        return BEST_38_FEATURES.copy()
    elif feature_type == 'best39':
        return BEST_39_FEATURES.copy()
    elif feature_type == 'best40':
        return BEST_40_FEATURES.copy()
    elif feature_type == 'best41':
        return BEST_41_FEATURES.copy()
    elif feature_type == 'best42':
        return BEST_42_FEATURES.copy()
    elif feature_type == 'best43':
        return BEST_43_FEATURES.copy()
    elif feature_type == 'best44':
        return BEST_44_FEATURES.copy()
    elif feature_type == 'best45':
        return BEST_45_FEATURES.copy()
    elif feature_type == 'best46':
        return BEST_46_FEATURES.copy()
    elif feature_type == 'best47':
        return BEST_47_FEATURES.copy()
    elif feature_type == 'best48':
        return BEST_48_FEATURES.copy()
    elif feature_type == 'best49':
        return BEST_49_FEATURES.copy()
    elif feature_type == 'best50':
        return BEST_50_FEATURES.copy()
    elif feature_type == 'best51':
        return BEST_51_FEATURES.copy()
    elif feature_type == 'best52':
        return BEST_52_FEATURES.copy()
    elif feature_type == 'best53':
        return BEST_53_FEATURES.copy()
    elif feature_type == 'best54':
        return BEST_54_FEATURES.copy()
    elif feature_type == 'best55':
        return BEST_55_FEATURES.copy()
    elif feature_type == 'best56':
        return BEST_56_FEATURES.copy()
    elif feature_type == 'best57':
        return BEST_57_FEATURES.copy()
    elif feature_type == 'best58':
        return BEST_58_FEATURES.copy()
    else:
        # 提供支持的类型列表
        supported_types = ['all63', 'selected58'] + [f'best{i}' for i in range(3, 59)]
        raise ValueError(f"不支持的特征类型: {feature_type}. 支持的类型: {supported_types}")

def get_categorical_features(feature_type: str = 'best10') -> List[str]:
    """
    获取指定特征集中的类别特征
    
    Args:
        feature_type: 特征集类型
        
    Returns:
        类别特征名称列表
    """
    selected_features = get_features_by_type(feature_type)
    categorical_features = []
    
    for feature in selected_features:
        if feature in CAT_FEATURE_NAMES:
            categorical_features.append(feature)
    
    return categorical_features

def get_categorical_indices(feature_type: str = 'best10') -> List[int]:
    """
    获取类别特征在选定特征中的索引
    
    Args:
        feature_type: 特征集类型
        
    Returns:
        类别特征索引列表
    """
    selected_features = get_features_by_type(feature_type)
    categorical_indices = []
    
    for i, feature in enumerate(selected_features):
        if feature in CAT_FEATURE_NAMES:
            categorical_indices.append(i)
    
    return categorical_indices

def get_feature_set_info(feature_type: str = 'best10') -> Dict[str, Any]:
    """
    获取特征集的详细信息
    
    Args:
        feature_type: 特征集类型
        
    Returns:
        包含特征集信息的字典
    """
    features = get_features_by_type(feature_type)
    categorical_features = get_categorical_features(feature_type)
    categorical_indices = get_categorical_indices(feature_type)
    
    return {
        'feature_type': feature_type,
        'total_features': len(features),
        'feature_names': features,
        'categorical_features': categorical_features,
        'categorical_indices': categorical_indices,
        'categorical_count': len(categorical_features),
        'numerical_count': len(features) - len(categorical_features),
        'categorical_ratio': len(categorical_features) / len(features) * 100
    }

# 日志级别
LOG_LEVEL = logging.INFO

# 实验配置
EXPERIMENT_CONFIG = {
    'random_state': 42,
    'cv_folds': 10,
    'test_size': 0.2,
    'validation_size': 0.2
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (10, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'font_size': 12
}

# 支持的不平衡处理方法
IMBALANCE_METHODS = [
    'none',
    'smote',
    'smotenc',
    'borderline_smote',
    'kmeans_smote',
    'svm_smote',
    'adasyn',
    'smote_tomek',
    'smote_enn',
    'random_under',
    'edited_nn'
]

# 支持的标准化方法
SCALING_METHODS = [
    'none',
    'standard',
    'robust'
]

# 支持的UDA方法
UDA_METHODS = {
    'covariate_shift': ['DM'],
    'linear_kernel': ['SA', 'TCA', 'JDA', 'CORAL'],
    'deep_learning': ['DANN', 'ADDA', 'WDGRL', 'DeepCORAL', 'MCD', 'MDD', 'CDAN'],
    'optimal_transport': ['POT']
}

if __name__ == "__main__":
    # 测试特征集配置
    print("=" * 60)
    print("特征集配置测试")
    print("=" * 60)
    
    # 测试主要特征集配置
    main_feature_types = ['all63', 'selected58', 'best3', 'best5', 'best8', 'best10', 'best12', 'best15', 'best20', 'best32', 'best58']
    
    for feature_type in main_feature_types:
        info = get_feature_set_info(feature_type)
        print(f"\n{feature_type.upper()}特征集:")
        print(f"  总特征数: {info['total_features']}")
        print(f"  类别特征数: {info['categorical_count']}")
        print(f"  数值特征数: {info['numerical_count']}")
        print(f"  类别特征比例: {info['categorical_ratio']:.1f}%")
        print(f"  前5个特征: {info['feature_names'][:5]}")
    
    print(f"\n\n全部支持的特征类型:")
    supported = ['all63', 'selected58'] + [f'best{i}' for i in range(3, 59)]
    print(f"  总计: {len(supported)}个类型")
    print(f"  基础类型: ['all63', 'selected58']")
    print(f"  最佳特征类型: best3 到 best58 (共{58-3+1}个)")
    print(f"  完整覆盖: 3到58个特征的所有配置") 
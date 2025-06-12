import logging

# 全局常量
SELECTED_FEATURES = [
    'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20', 'Feature21',
    'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31',
    'Feature32', 'Feature35', 'Feature37', 'Feature38', 'Feature39', 'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45',
    'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55',
    'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61', 'Feature62', 'Feature63'
]

# 类别特征名称 (根据用户指定的列表)
# Note: This list includes 'Feature40', which is not present in the SELECTED_FEATURES list above.
# The original comment attempted to map indices from a previous feature set to the current 58 features.
# The list below is used as specified by the user instruction, despite potential inconsistencies with SELECTED_FEATURES.
CAT_FEATURE_NAMES = [
    'Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11',
    'Feature45', 'Feature46', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature63'
]

CAT_IDX = [SELECTED_FEATURES.index(f) for f in CAT_FEATURE_NAMES if f in SELECTED_FEATURES]

TABPFN_PARAMS = {'device': 'cuda', 'max_time': 60, 'random_state': 42}

# 项目根目录的相对路径，用于结果输出
# This might need adjustment if scripts are run from project_root/scripts/
# For now, assuming scripts are run from project_root.
# BASE_PATH = './results_analytical_coral_A2B_feature58' # Original suggestion
BASE_PATH = 'results_analytical_coral_A2B_feature58' # More robust if analytical_coral_A2B_feature58 is the root

# 数据文件路径 (相对于项目根目录 analytical_coral_A2B_feature58)
# These need to be relative to the new project root `analytical_coral_A2B_feature58`
# or absolute paths. Assuming the `data` folder from the original `TabPFN` project is accessible one level up.
# DATA_PATH_A = "../data/AI4healthcare.xlsx" # Original suggestion
# DATA_PATH_B = "../data/HenanCancerHospital_features63_58.xlsx" # Original suggestion

# Updated data paths relative to the main TabPFN project root
DATA_PATH_A = "data/AI4healthcare.xlsx"
DATA_PATH_B = "data/HenanCancerHospital_features63_58.xlsx"

LABEL_COL = "Label"

# 日志级别
LOG_LEVEL = logging.INFO 
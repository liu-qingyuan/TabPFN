#!/bin/bash
# 特征选择作业模板
# 用于执行特征选择实验的批处理作业

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT}
#SBATCH --error=${ERROR}
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1

# 加载必要的环境模块
# 可根据实际环境修改
module load anaconda
source activate tabpfn_env

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=${GPU}

# 记录作业开始信息
echo "Job started: $(date)"
echo "Running script: ${SCRIPT}"
echo "Config file: ${CONFIG}"
echo "Dataset: ${DATASET}"
echo "Feature selection method: ${FEATURE_METHOD}"
echo "Number of features: ${N_FEATURES}"

# 执行Python脚本
python ${SCRIPT} --config ${CONFIG} --dataset ${DATASET} --feature-method ${FEATURE_METHOD} --n-features ${N_FEATURES}

# 记录作业结束信息
echo "Job finished: $(date)" 
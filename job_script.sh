#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --gres=gpu:4g.40gb:1  # 申请1个GPU
#SBATCH --output=output.log    # 任务输出日志
#SBATCH --error=error.log      # 错误日志

python predict_healthcare_auto_predict_all_ABC.py
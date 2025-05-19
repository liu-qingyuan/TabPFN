#!/usr/bin/env python
"""
统一作业提交工具
用法: python submit.py --config configs/cross/AB_to_C.yaml [--gpu 0] [--dry-run]
"""
import argparse
import yaml
import os
import datetime
import sys

def main():
    parser = argparse.ArgumentParser(description="TabPFN实验作业提交工具")
    parser.add_argument("--config", required=True, help="作业配置文件")
    parser.add_argument("--gpu", help="指定GPU，覆盖配置文件")
    parser.add_argument("--dry-run", action="store_true", help="不实际提交，只打印命令")
    args = parser.parse_args()
    
    # 检查配置文件存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件 {args.config} 不存在")
        sys.exit(1)
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'experiment' not in config:
        print(f"错误：配置文件缺少 'experiment' 部分")
        sys.exit(1)
    
    # 生成日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config['experiment'].get('name', 'unknown')
    log_dir = f"logs/experiments/{experiment_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 选择作业模板
    template_type = config['experiment'].get('template', 'single_dataset')
    if template_type == 'cross_dataset':
        template = "jobs/templates/cross_dataset.sh"
    elif template_type == 'feature_selection':
        template = "jobs/templates/feature_selection.sh"
    else:
        template = "jobs/templates/single_dataset.sh"
    
    # 构建环境变量
    env_vars = {}
    env_vars['JOB_NAME'] = experiment_name
    env_vars['OUTPUT'] = f"{log_dir}/output.log"
    env_vars['ERROR'] = f"{log_dir}/error.log"
    env_vars['SCRIPT'] = config['experiment']['script']
    env_vars['CONFIG'] = args.config
    
    # GPU设置
    env_vars['GPU'] = args.gpu if args.gpu else config['experiment'].get('resources', {}).get('gpu', '0')
    
    # 数据集设置
    if template_type == 'cross_dataset' and 'datasets' in config['experiment']:
        env_vars['TRAIN_DATASETS'] = ",".join(config['experiment']['datasets'].get('train', []))
        env_vars['TEST_DATASET'] = config['experiment']['datasets'].get('test', '')
    elif 'dataset' in config['experiment']:
        env_vars['DATASET'] = config['experiment']['dataset']
    
    # 特征选择设置
    if template_type == 'feature_selection' and 'features' in config['experiment']:
        env_vars['FEATURE_METHOD'] = config['experiment']['features'].get('method', 'RFE')
        env_vars['N_FEATURES'] = str(config['experiment']['features'].get('n_features', '7'))
    
    # 构建导出变量字符串
    export_vars = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    
    # 构建作业提交命令
    cmd = f"sbatch {template} --export=\"{export_vars}\""
    
    # 保存作业配置副本
    config_copy = f"{log_dir}/job_config.yaml"
    with open(config_copy, 'w') as f:
        yaml.dump(config, f)
    
    if args.dry_run:
        print(f"作业配置:")
        for k, v in env_vars.items():
            print(f"  {k}: {v}")
        print(f"将执行: {cmd}")
    else:
        print(f"提交作业: {experiment_name}")
        exit_code = os.system(cmd)
        if exit_code == 0:
            print(f"作业已提交，日志目录: {log_dir}")
            # 创建最新结果的符号链接
            latest_link = f"logs/experiments/{experiment_name}_latest"
            if os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(log_dir, latest_link)
        else:
            print(f"作业提交失败，退出代码: {exit_code}")

if __name__ == "__main__":
    main() 
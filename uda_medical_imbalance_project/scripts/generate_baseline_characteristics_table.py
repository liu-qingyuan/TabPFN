#!/usr/bin/env python3
"""
生成基线特征表格（Baseline Characteristics Table）

用于生成临床研究中常见的基线特征表格，展示不同队列样本的人口统计学/临床特征分布。
常见于《Lancet》《JAMA》《NEJM》等临床研究中。

特征表达方式：
- 连续变量：均值 ± 标准差
- 类别变量：频数 + 百分比
- 分组：Train cohort (A) / Test cohort (B)

运行示例: python scripts/generate_baseline_characteristics_table.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import MedicalDataLoader
from config.settings import CAT_FEATURE_NAMES


class BaselineCharacteristicsGenerator:
    """基线特征表格生成器"""
    
    def __init__(
        self,
        feature_type: str = 'best8',
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化基线特征表格生成器
        
        Args:
            feature_type: 特征集类型 ('all63', 'selected58', 'best3', 'best4', ..., 'best58')
            output_dir: 输出目录
            verbose: 是否输出详细信息
        """
        self.feature_type = feature_type
        self.verbose = verbose
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root / "results" / f"baseline_characteristics_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载特征映射表
        self.feature_mapping = self._load_feature_mapping()
        
        if self.verbose:
            print(f"🏥 基线特征表格生成器初始化")
            print(f"   特征集: {feature_type}")
            print(f"   输出目录: {output_dir}")
    
    def _load_feature_mapping(self) -> Dict[str, str]:
        """加载特征映射表"""
        try:
            mapping_file = Path("data/Feature_Ranking_with_Original_Names.csv")
            if not mapping_file.exists():
                # 尝试从项目根目录查找
                mapping_file = project_root.parent / "data" / "Feature_Ranking_with_Original_Names.csv"
            
            if mapping_file.exists():
                df = pd.read_csv(mapping_file)
                mapping = dict(zip(df['Feature'], df['Original Feature Name']))
                if self.verbose:
                    print(f"✅ 特征映射表加载成功: {len(mapping)} 个特征")
                return mapping
            else:
                if self.verbose:
                    print("⚠ 特征映射表未找到，使用默认特征名")
                return {}
        except Exception as e:
            if self.verbose:
                print(f"⚠ 特征映射表加载失败: {e}")
            return {}
    
    def _get_original_feature_name(self, feature: str) -> str:
        """获取特征的原始名称"""
        if feature in self.feature_mapping:
            return self.feature_mapping[feature]
        else:
            # 如果没有映射，返回原始特征名
            return feature
    
    def _is_categorical_feature(self, feature: str) -> bool:
        """判断特征是否为类别特征"""
        return feature in CAT_FEATURE_NAMES
    
    def _calculate_continuous_stats(self, data: np.ndarray) -> Dict[str, float]:
        """计算连续变量统计量"""
        return {
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),  # 样本标准差
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'min': np.min(data),
            'max': np.max(data),
            'n_valid': len(data) - np.isnan(data).sum()
        }
    
    def _calculate_categorical_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """计算类别变量统计量"""
        # 移除缺失值
        valid_data = data[~np.isnan(data)]
        unique_values, counts = np.unique(valid_data, return_counts=True)
        
        total_valid = len(valid_data)
        
        # 计算频数和百分比
        freq_dict = {}
        for value, count in zip(unique_values, counts):
            freq_dict[f'category_{int(value)}'] = {
                'count': count,
                'percentage': (count / total_valid * 100) if total_valid > 0 else 0
            }
        
        return {
            'categories': freq_dict,
            'n_valid': total_valid,
            'n_missing': len(data) - total_valid,
            'n_categories': len(unique_values)
        }
    
    def _format_continuous_display(self, stats: Dict[str, float]) -> str:
        """格式化连续变量的显示"""
        return f"{stats['mean']:.2f} ± {stats['std']:.2f}"
    
    def _format_categorical_display(self, stats: Dict[str, Any]) -> List[str]:
        """格式化类别变量的显示"""
        display_lines = []
        
        # 获取所有类别值并排序
        categories = sorted(stats['categories'].items(), key=lambda x: int(x[0].replace('category_', '')))
        
        for category, info in categories:
            count = info['count']
            percentage = info['percentage']
            category_value = int(category.replace('category_', ''))
            
            # 判断是否为二分类特征
            total_categories = len(stats['categories'])
            
            if total_categories == 2:
                # 二分类特征：使用 No/Negative 和 Yes/Positive
                if category_value == 0:
                    category_display = "No/Negative"
                elif category_value == 1:
                    category_display = "Yes/Positive"
                else:
                    category_display = f"Category {category_value}"
            else:
                # 多分类特征：直接使用原始类别值
                category_display = f"Category {category_value}"
            
            display_lines.append(f"  {category_display}: {count} ({percentage:.1f}%)")
        
        return display_lines
    
    def load_datasets(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """加载数据集A和B"""
        if self.verbose:
            print(f"\n📊 加载数据集...")
        
        loader = MedicalDataLoader()
        
        # 加载数据集A（源域，Train cohort）
        data_A = loader.load_dataset('A', feature_type=self.feature_type)
        
        # 加载数据集B（目标域，Test cohort）
        data_B = loader.load_dataset('B', feature_type=self.feature_type)
        
        if self.verbose:
            print(f"✅ 数据集加载完成:")
            print(f"   Train cohort (A): {data_A['n_samples']} 样本")
            print(f"   Test cohort (B): {data_B['n_samples']} 样本")
            print(f"   特征数量: {data_A['n_features']}")
            print(f"   特征列表: {data_A['feature_names']}")
        
        return data_A, data_B
    
    def generate_baseline_table(self) -> pd.DataFrame:
        """生成基线特征表格"""
        if self.verbose:
            print(f"\n📋 生成基线特征表格...")
        
        # 加载数据
        data_A, data_B = self.load_datasets()
        
        # 提取特征和标签
        X_A = data_A['X']
        y_A = data_A['y']
        X_B = data_B['X']
        y_B = data_B['y']
        feature_names = data_A['feature_names']
        
        # 创建结果列表
        table_data = []
        
        # 处理每个特征
        for i, feature in enumerate(feature_names):
            # 获取原始特征名
            original_name = self._get_original_feature_name(feature)
            
            # 提取特征数据
            feature_data_A = X_A[:, i]
            feature_data_B = X_B[:, i]
            
            # 判断特征类型
            is_categorical = self._is_categorical_feature(feature)
            
            if is_categorical:
                # 类别特征
                stats_A = self._calculate_categorical_stats(feature_data_A)
                stats_B = self._calculate_categorical_stats(feature_data_B)
                
                # 格式化显示
                display_A = self._format_categorical_display(stats_A)
                display_B = self._format_categorical_display(stats_B)
                
                # 添加主行
                table_data.append({
                    'Feature': original_name,
                    'Feature_Code': feature,
                    'Type': 'Categorical',
                    'Train_Cohort_A': f"n = {stats_A['n_valid']}",
                    'Test_Cohort_B': f"n = {stats_B['n_valid']}",
                    'Is_Subrow': False
                })
                
                # 添加子行（各类别）
                max_categories = max(len(display_A), len(display_B))
                for j in range(max_categories):
                    sub_A = display_A[j] if j < len(display_A) else ""
                    sub_B = display_B[j] if j < len(display_B) else ""
                    
                    table_data.append({
                        'Feature': "",
                        'Feature_Code': feature,
                        'Type': 'Categorical_Sub',
                        'Train_Cohort_A': sub_A,
                        'Test_Cohort_B': sub_B,
                        'Is_Subrow': True
                    })
            
            else:
                # 连续特征
                stats_A = self._calculate_continuous_stats(feature_data_A)
                stats_B = self._calculate_continuous_stats(feature_data_B)
                
                # 格式化显示
                display_A = self._format_continuous_display(stats_A)
                display_B = self._format_continuous_display(stats_B)
                
                table_data.append({
                    'Feature': original_name,
                    'Feature_Code': feature,
                    'Type': 'Continuous',
                    'Train_Cohort_A': display_A,
                    'Test_Cohort_B': display_B,
                    'Is_Subrow': False
                })
        
        # 添加样本大小信息
        table_data.insert(0, {
            'Feature': 'Sample Size',
            'Feature_Code': 'N',
            'Type': 'Summary',
            'Train_Cohort_A': f"n = {len(y_A)}",
            'Test_Cohort_B': f"n = {len(y_B)}",
            'Is_Subrow': False
        })
        
        # 添加标签分布信息
        # 计算标签分布
        label_stats_A = self._calculate_categorical_stats(y_A)
        label_stats_B = self._calculate_categorical_stats(y_B)
        
        table_data.append({
            'Feature': 'Outcome (Malignant)',
            'Feature_Code': 'Label',
            'Type': 'Categorical',
            'Train_Cohort_A': f"n = {label_stats_A['n_valid']}",
            'Test_Cohort_B': f"n = {label_stats_B['n_valid']}",
            'Is_Subrow': False
        })
        
        # 添加标签子行
        label_display_A = self._format_categorical_display(label_stats_A)
        label_display_B = self._format_categorical_display(label_stats_B)
        
        max_labels = max(len(label_display_A), len(label_display_B))
        for j in range(max_labels):
            sub_A = label_display_A[j] if j < len(label_display_A) else ""
            sub_B = label_display_B[j] if j < len(label_display_B) else ""
            
            table_data.append({
                'Feature': "",
                'Feature_Code': 'Label',
                'Type': 'Categorical_Sub',
                'Train_Cohort_A': sub_A,
                'Test_Cohort_B': sub_B,
                'Is_Subrow': True
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(table_data)
        
        if self.verbose:
            print(f"✅ 基线特征表格生成完成")
            print(f"   表格行数: {len(df)}")
            print(f"   特征数量: {len(feature_names)}")
        
        return df
    
    def save_table(self, df: pd.DataFrame, formats: List[str] = ['csv', 'xlsx']) -> Dict[str, str]:
        """保存基线特征表格"""
        if self.verbose:
            print(f"\n💾 保存基线特征表格...")
        
        saved_files = {}
        
        # 创建显示用的DataFrame（移除辅助列）
        display_df = df[['Feature', 'Train_Cohort_A', 'Test_Cohort_B']].copy()
        display_df.columns = ['Characteristic', 'Train Cohort (A)', 'Test Cohort (B)']
        
        # 保存为不同格式
        for format_type in formats:
            if format_type == 'csv':
                file_path = self.output_dir / f"baseline_characteristics_{self.feature_type}.csv"
                display_df.to_csv(file_path, index=False, encoding='utf-8')
                saved_files['csv'] = str(file_path)
            
            elif format_type == 'xlsx':
                file_path = self.output_dir / f"baseline_characteristics_{self.feature_type}.xlsx"
                
                # 使用ExcelWriter进行格式化
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    display_df.to_excel(writer, sheet_name='Baseline_Characteristics', index=False)
                    
                    # 获取工作表
                    worksheet = writer.sheets['Baseline_Characteristics']
                    
                    # 设置列宽
                    worksheet.column_dimensions['A'].width = 30
                    worksheet.column_dimensions['B'].width = 20
                    worksheet.column_dimensions['C'].width = 20
                    
                    # 设置表头格式
                    from openpyxl.styles import Font, PatternFill, Alignment
                    
                    header_font = Font(bold=True, size=12)
                    header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
                    
                    for cell in worksheet[1]:
                        cell.font = header_font
                        cell.fill = header_fill
                        cell.alignment = Alignment(horizontal="center")
                
                saved_files['xlsx'] = str(file_path)
        
        if self.verbose:
            for format_type, file_path in saved_files.items():
                print(f"✅ {format_type.upper()} 文件已保存: {file_path}")
        
        return saved_files
    
    def generate_visualization(self, df: pd.DataFrame) -> str:
        """生成基线特征可视化图表"""
        if self.verbose:
            print(f"\n📊 生成基线特征可视化...")
        
        # 筛选连续特征进行可视化（排除标签和汇总行）
        continuous_features = df[
            (df['Type'] == 'Continuous') & 
            (~df['Is_Subrow']) &
            (df['Feature_Code'] != 'Label') &
            (df['Feature_Code'] != 'N')
        ].copy()
        
        # 筛选类别特征（排除标签和汇总行）
        categorical_features = df[
            (df['Type'] == 'Categorical') & 
            (~df['Is_Subrow']) &
            (df['Feature_Code'] != 'Label') &
            (df['Feature_Code'] != 'N')
        ].copy()
        
        if len(continuous_features) == 0 and len(categorical_features) == 0:
            if self.verbose:
                print("⚠ 没有特征可用于可视化")
            return ""
        
        if self.verbose:
            print(f"   连续特征数量: {len(continuous_features)}")
            print(f"   类别特征数量: {len(categorical_features)}")
            print(f"   总计可视化特征: {len(continuous_features) + len(categorical_features)}")
            print(f"   注意: 类别特征将显示为柱状图，连续特征显示为直方图")
        
        # 重新加载数据用于可视化
        data_A, data_B = self.load_datasets()
        X_A = data_A['X']
        X_B = data_B['X']
        feature_names = data_A['feature_names']
        
        # 合并所有要可视化的特征
        all_features = pd.concat([continuous_features, categorical_features], ignore_index=True)
        
        # 创建子图
        n_features = len(all_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制每个特征的分布
        for idx, (_, row) in enumerate(all_features.iterrows()):
            feature_code = row['Feature_Code']
            feature_name = row['Feature']
            feature_type = row['Type']
            
            # 获取特征在数组中的索引
            feature_idx = feature_names.index(feature_code)
            
            # 提取特征数据
            data_A_feature = X_A[:, feature_idx]
            data_B_feature = X_B[:, feature_idx]
            
            # 移除缺失值
            data_A_clean = data_A_feature[~np.isnan(data_A_feature)]
            data_B_clean = data_B_feature[~np.isnan(data_B_feature)]
            
            # 计算子图位置
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            
            if n_rows == 1:
                ax = axes[col_idx]
            else:
                ax = axes[row_idx, col_idx]
            
            # 根据特征类型绘制不同的图表
            if feature_type == 'Continuous':
                # 连续特征：直方图
                ax.hist(data_A_clean, bins=20, alpha=0.7, label='Train Cohort (A)', color='skyblue', density=True)
                ax.hist(data_B_clean, bins=20, alpha=0.7, label='Test Cohort (B)', color='lightcoral', density=True)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.set_title(f'{feature_name}\n(Continuous)', fontsize=10, fontweight='bold')
                
            elif feature_type == 'Categorical':
                # 类别特征：柱状图
                # 计算每个类别的频数
                unique_values = np.unique(np.concatenate([data_A_clean, data_B_clean]))
                
                counts_A = []
                counts_B = []
                for val in unique_values:
                    counts_A.append(np.sum(data_A_clean == val))
                    counts_B.append(np.sum(data_B_clean == val))
                
                # 计算百分比
                pct_A = np.array(counts_A) / len(data_A_clean) * 100
                pct_B = np.array(counts_B) / len(data_B_clean) * 100
                
                x = np.arange(len(unique_values))
                width = 0.35
                
                ax.bar(x - width/2, pct_A, width, label='Train Cohort (A)', color='skyblue', alpha=0.7)
                ax.bar(x + width/2, pct_B, width, label='Test Cohort (B)', color='lightcoral', alpha=0.7)
                
                # 设置x轴标签
                category_labels = []
                total_categories = len(unique_values)
                
                for val in unique_values:
                    if total_categories == 2:
                        # 二分类特征
                        if val == 0:
                            category_labels.append('No/Negative')
                        elif val == 1:
                            category_labels.append('Yes/Positive')
                        else:
                            category_labels.append(f'Cat {int(val)}')
                    else:
                        # 多分类特征：直接使用原始类别值
                        category_labels.append(f'Cat {int(val)}')
                
                ax.set_xticks(x)
                ax.set_xticklabels(category_labels)
                ax.set_xlabel('Category')
                ax.set_ylabel('Percentage (%)')
                ax.set_title(f'{feature_name}\n(Categorical)', fontsize=10, fontweight='bold')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_features, n_rows * n_cols):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            if n_rows == 1:
                axes[col_idx].set_visible(False)
            else:
                axes[row_idx, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        # 添加总标题
        fig.suptitle(f'Baseline Characteristics Distribution ({self.feature_type.upper()} Features)\n'
                    f'Continuous Features: {len(continuous_features)}, Categorical Features: {len(categorical_features)}', 
                    fontsize=14, fontweight='bold', y=0.96)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # 为总标题留出更多空间
        
        # 保存图表
        viz_file = self.output_dir / f"baseline_characteristics_all_features_{self.feature_type}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"✅ 可视化图表已保存: {viz_file}")
            print(f"   包含 {len(continuous_features)} 个连续特征和 {len(categorical_features)} 个类别特征")
        
        return str(viz_file)
    
    def generate_summary_report(self, df: pd.DataFrame, saved_files: Dict[str, str], viz_file: str) -> str:
        """生成汇总报告"""
        if self.verbose:
            print(f"\n📋 生成汇总报告...")
        
        # 统计信息
        total_features = len(df[~df['Is_Subrow']])
        categorical_features = len(df[(df['Type'] == 'Categorical') & (~df['Is_Subrow'])])
        continuous_features = len(df[(df['Type'] == 'Continuous') & (~df['Is_Subrow'])])
        
        # 生成报告内容
        report_content = []
        report_content.append("# 基线特征表格生成报告\n")
        report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 配置信息
        report_content.append("## 配置信息\n")
        report_content.append(f"- 特征集: {self.feature_type}")
        report_content.append(f"- 输出目录: {self.output_dir}")
        report_content.append("")
        
        # 数据集信息
        report_content.append("## 数据集信息\n")
        data_A, data_B = self.load_datasets()
        report_content.append(f"- Train Cohort (A): {data_A['n_samples']} 样本")
        report_content.append(f"- Test Cohort (B): {data_B['n_samples']} 样本")
        report_content.append(f"- 总特征数: {total_features}")
        report_content.append(f"- 连续特征数: {continuous_features}")
        report_content.append(f"- 类别特征数: {categorical_features}")
        report_content.append("")
        
        # 输出文件
        report_content.append("## 输出文件\n")
        for format_type, file_path in saved_files.items():
            report_content.append(f"- {format_type.upper()} 表格: {file_path}")
        
        if viz_file:
            report_content.append(f"- 可视化图表: {viz_file}")
        
        report_content.append("")
        
        # 使用说明
        report_content.append("## 使用说明\n")
        report_content.append("### 表格格式说明")
        report_content.append("- **连续变量**: 显示为 `均值 ± 标准差`")
        report_content.append("- **类别变量**: 显示为 `频数 (百分比%)`")
        report_content.append("- **Train Cohort (A)**: 源域数据集（AI4health）")
        report_content.append("- **Test Cohort (B)**: 目标域数据集（HenanCancerHospital）")
        report_content.append("")
        
        report_content.append("### 特征映射")
        report_content.append("特征代码到原始名称的映射基于 `data/Feature_Ranking_with_Original_Names.csv`")
        report_content.append("")
        
        report_content.append("### 引用格式")
        report_content.append("此表格适用于临床研究论文，格式参考《Lancet》《JAMA》《NEJM》等期刊的基线特征表格标准。")
        
        # 保存报告
        report_file = self.output_dir / f"baseline_characteristics_report_{self.feature_type}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        if self.verbose:
            print(f"✅ 汇总报告已保存: {report_file}")
        
        return str(report_file)
    
    def run_complete_analysis(self) -> Dict[str, str]:
        """运行完整的基线特征分析"""
        if self.verbose:
            print(f"🏥 开始基线特征表格生成")
            print("=" * 60)
        
        try:
            # 1. 生成基线特征表格
            df = self.generate_baseline_table()
            
            # 2. 保存表格
            saved_files = self.save_table(df, formats=['csv', 'xlsx'])
            
            # 3. 生成可视化
            viz_file = self.generate_visualization(df)
            
            # 4. 生成汇总报告
            report_file = self.generate_summary_report(df, saved_files, viz_file)
            
            # 5. 返回结果
            results = {
                'table_csv': saved_files.get('csv', ''),
                'table_xlsx': saved_files.get('xlsx', ''),
                'visualization': viz_file,
                'report': report_file,
                'output_dir': str(self.output_dir)
            }
            
            if self.verbose:
                print(f"\n✅ 基线特征表格生成完成！")
                print(f"📁 输出目录: {self.output_dir}")
                print(f"📊 CSV表格: {results['table_csv']}")
                print(f"📊 Excel表格: {results['table_xlsx']}")
                print(f"📈 可视化图表: {results['visualization']}")
                print(f"📋 汇总报告: {results['report']}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 基线特征表格生成失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


def main():
    """主函数"""
    print("🏥 基线特征表格生成器")
    print("=" * 60)
    
    # 创建生成器
    generator = BaselineCharacteristicsGenerator(
        feature_type='best8',
        verbose=True
    )
    
    # 运行完整分析
    results = generator.run_complete_analysis()
    
    if 'error' not in results:
        print(f"\n🎉 基线特征表格生成成功！")
        print(f"📁 查看输出目录: {results['output_dir']}")
    else:
        print(f"\n❌ 生成失败: {results['error']}")


if __name__ == "__main__":
    main() 
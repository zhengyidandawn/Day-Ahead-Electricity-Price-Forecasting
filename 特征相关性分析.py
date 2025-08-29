#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征相关性分析脚本
作者: AI助手
日期: 2024年
功能: 计算Pearson、Spearman和Kendall相关系数，分析特征与出清电价的相关性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
import logging
from datetime import datetime

# 设置中文字体和日志
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureCorrelationAnalyzer:
    """特征相关性分析器"""
    
    def __init__(self, data_path):
        """
        初始化分析器
        
        Args:
            data_path (str): 特征矩阵数据路径
        """
        self.data_path = data_path
        self.df = None
        self.correlation_results = {}
        
    def load_data(self):
        """加载数据"""
        logger.info("正在加载特征矩阵数据...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"数据加载成功，数据形状: {self.df.shape}")
            
            # 转换时间列
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            
            # 检查数据质量
            logger.info(f"数据时间范围: {self.df['datetime'].min()} 到 {self.df['datetime'].max()}")
            logger.info(f"缺失值统计:\n{self.df.isnull().sum().sum()}")
            
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return False
    
    def clean_data(self):
        """数据清理"""
        logger.info("清理数据...")
        
        # 替换无穷值
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # 获取数值列
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # 填充缺失值
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())
        
        logger.info(f"数据清理后缺失值统计: {self.df.isnull().sum().sum()}")
    
    def calculate_correlations(self):
        """计算三种相关系数"""
        logger.info("计算特征与电价的相关性...")
        
        # 获取数值特征列（排除datetime和price）
        feature_columns = [col for col in self.df.columns if col not in ['datetime', 'price'] and self.df[col].dtype in ['int64', 'float64']]
        
        # 获取目标变量
        target = self.df['price']
        
        # 存储结果
        correlation_data = []
        
        for feature in feature_columns:
            feature_data = self.df[feature]
            
            # 计算三种相关系数
            try:
                # Pearson相关系数
                pearson_corr, pearson_p = pearsonr(feature_data, target)
                
                # Spearman相关系数
                spearman_corr, spearman_p = spearmanr(feature_data, target)
                
                # Kendall相关系数
                kendall_corr, kendall_p = kendalltau(feature_data, target)
                
                correlation_data.append({
                    'feature': feature,
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'kendall_corr': kendall_corr,
                    'kendall_p': kendall_p
                })
                
            except Exception as e:
                logger.warning(f"计算特征 {feature} 相关性时出错: {e}")
                continue
        
        # 转换为DataFrame
        self.correlation_df = pd.DataFrame(correlation_data)
        
        # 按Pearson相关系数绝对值排序
        self.correlation_df['abs_pearson'] = abs(self.correlation_df['pearson_corr'])
        self.correlation_df = self.correlation_df.sort_values('abs_pearson', ascending=False)
        
        logger.info(f"成功计算了 {len(self.correlation_df)} 个特征的相关性")
        
        return self.correlation_df
    
    def create_correlation_matrix(self):
        """创建特征相关性矩阵"""
        logger.info("创建特征相关性矩阵...")
        
        # 获取数值特征列
        feature_columns = [col for col in self.df.columns if col not in ['datetime', 'price'] and self.df[col].dtype in ['int64', 'float64']]
        
        # 选择前20个最重要的特征（基于与电价的相关性）
        top_features = self.correlation_df.head(20)['feature'].tolist()
        
        # 添加price到特征列表
        analysis_features = top_features + ['price']
        
        # 创建相关性矩阵
        correlation_matrix = self.df[analysis_features].corr()
        
        return correlation_matrix, top_features
    
    def generate_visualizations(self):
        """生成可视化图表"""
        logger.info("生成相关性分析可视化...")
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('特征与出清电价相关性分析', fontsize=16, fontweight='bold')
        
        # 1. Pearson相关系数条形图
        top_20_pearson = self.correlation_df.head(20)
        axes[0, 0].barh(range(len(top_20_pearson)), top_20_pearson['pearson_corr'])
        axes[0, 0].set_yticks(range(len(top_20_pearson)))
        axes[0, 0].set_yticklabels(top_20_pearson['feature'], fontsize=10)
        axes[0, 0].set_xlabel('Pearson相关系数')
        axes[0, 0].set_title('Top 20特征 - Pearson相关系数')
        axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spearman相关系数条形图
        axes[0, 1].barh(range(len(top_20_pearson)), top_20_pearson['spearman_corr'])
        axes[0, 1].set_yticks(range(len(top_20_pearson)))
        axes[0, 1].set_yticklabels(top_20_pearson['feature'], fontsize=10)
        axes[0, 1].set_xlabel('Spearman相关系数')
        axes[0, 1].set_title('Top 20特征 - Spearman相关系数')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Kendall相关系数条形图
        axes[1, 0].barh(range(len(top_20_pearson)), top_20_pearson['kendall_corr'])
        axes[1, 0].set_yticks(range(len(top_20_pearson)))
        axes[1, 0].set_yticklabels(top_20_pearson['feature'], fontsize=10)
        axes[1, 0].set_xlabel('Kendall相关系数')
        axes[1, 0].set_title('Top 20特征 - Kendall相关系数')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 相关性矩阵热力图
        correlation_matrix, top_features = self.create_correlation_matrix()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
        axes[1, 1].set_title('特征相关性矩阵热力图')
        
        plt.tight_layout()
        plt.savefig('特征相关性分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("可视化图表已保存为 '特征相关性分析.png'")
    
    def generate_detailed_report(self):
        """生成详细的相关性分析报告"""
        logger.info("生成详细的相关性分析报告...")
        
        # 创建报告
        report = []
        report.append("# 特征与出清电价相关性分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据样本数: {len(self.df):,}")
        report.append(f"分析特征数: {len(self.correlation_df)}")
        report.append("")
        
        # 1. 高相关性特征（|相关系数| > 0.5）
        high_corr = self.correlation_df[abs(self.correlation_df['pearson_corr']) > 0.5]
        report.append("## 1. 高相关性特征 (|Pearson相关系数| > 0.5)")
        if len(high_corr) > 0:
            for _, row in high_corr.iterrows():
                report.append(f"- **{row['feature']}**:")
                report.append(f"  - Pearson: {row['pearson_corr']:.4f} (p={row['pearson_p']:.4f})")
                report.append(f"  - Spearman: {row['spearman_corr']:.4f} (p={row['spearman_p']:.4f})")
                report.append(f"  - Kendall: {row['kendall_corr']:.4f} (p={row['kendall_p']:.4f})")
                report.append("")
        else:
            report.append("无高相关性特征")
            report.append("")
        
        # 2. 中等相关性特征（0.3 < |相关系数| <= 0.5）
        medium_corr = self.correlation_df[(abs(self.correlation_df['pearson_corr']) > 0.3) & 
                                        (abs(self.correlation_df['pearson_corr']) <= 0.5)]
        report.append("## 2. 中等相关性特征 (0.3 < |Pearson相关系数| <= 0.5)")
        if len(medium_corr) > 0:
            for _, row in medium_corr.iterrows():
                report.append(f"- **{row['feature']}**:")
                report.append(f"  - Pearson: {row['pearson_corr']:.4f} (p={row['pearson_p']:.4f})")
                report.append(f"  - Spearman: {row['spearman_corr']:.4f} (p={row['spearman_p']:.4f})")
                report.append(f"  - Kendall: {row['kendall_corr']:.4f} (p={row['kendall_p']:.4f})")
                report.append("")
        else:
            report.append("无中等相关性特征")
            report.append("")
        
        # 3. 特征分类分析
        report.append("## 3. 特征分类相关性分析")
        
        # 时间特征
        time_features = [col for col in self.correlation_df['feature'] if any(x in col for x in ['month', 'day', 'hour', 'dayofweek', 'quarter', 'dayofyear', 'weekend', 'holiday'])]
        if time_features:
            report.append("### 时间特征")
            time_corr = self.correlation_df[self.correlation_df['feature'].isin(time_features)]
            for _, row in time_corr.iterrows():
                report.append(f"- {row['feature']}: Pearson={row['pearson_corr']:.4f}")
            report.append("")
        
        # 价格滞后特征
        lag_features = [col for col in self.correlation_df['feature'] if 'lag' in col]
        if lag_features:
            report.append("### 价格滞后特征")
            lag_corr = self.correlation_df[self.correlation_df['feature'].isin(lag_features)]
            for _, row in lag_corr.iterrows():
                report.append(f"- {row['feature']}: Pearson={row['pearson_corr']:.4f}")
            report.append("")
        
        # 容量比特征
        capacity_features = [col for col in self.correlation_df['feature'] if 'capacity_ratio' in col]
        if capacity_features:
            report.append("### 容量比特征")
            capacity_corr = self.correlation_df[self.correlation_df['feature'].isin(capacity_features)]
            for _, row in capacity_corr.iterrows():
                report.append(f"- {row['feature']}: Pearson={row['pearson_corr']:.4f}")
            report.append("")
        
        # 市场概率特征
        market_features = [col for col in self.correlation_df['feature'] if 'market_prob' in col]
        if market_features:
            report.append("### 市场概率特征")
            market_corr = self.correlation_df[self.correlation_df['feature'].isin(market_features)]
            for _, row in market_corr.iterrows():
                report.append(f"- {row['feature']}: Pearson={row['pearson_corr']:.4f}")
            report.append("")
        
        # 4. 统计摘要
        report.append("## 4. 相关性统计摘要")
        report.append(f"- 平均Pearson相关系数: {self.correlation_df['pearson_corr'].mean():.4f}")
        report.append(f"- 平均Spearman相关系数: {self.correlation_df['spearman_corr'].mean():.4f}")
        report.append(f"- 平均Kendall相关系数: {self.correlation_df['kendall_corr'].mean():.4f}")
        report.append(f"- 最大Pearson相关系数: {self.correlation_df['pearson_corr'].max():.4f}")
        report.append(f"- 最小Pearson相关系数: {self.correlation_df['pearson_corr'].min():.4f}")
        report.append("")
        
        # 5. 建议
        report.append("## 5. 建模建议")
        report.append("基于相关性分析结果，建议在电价预测模型中重点关注以下特征：")
        
        # 获取前10个最重要的特征
        top_10_features = self.correlation_df.head(10)['feature'].tolist()
        for i, feature in enumerate(top_10_features, 1):
            corr_value = self.correlation_df[self.correlation_df['feature'] == feature]['pearson_corr'].iloc[0]
            report.append(f"{i}. **{feature}** (相关系数: {corr_value:.4f})")
        
        # 保存报告
        with open('特征相关性分析报告.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info("详细报告已保存为 '特征相关性分析报告.md'")
        
        return '\n'.join(report)
    
    def save_correlation_data(self):
        """保存相关性数据到CSV文件"""
        logger.info("保存相关性数据...")
        
        # 保存完整相关性数据
        self.correlation_df.to_csv('特征相关性分析结果.csv', index=False, encoding='utf-8-sig')
        
        # 保存前20个最重要特征的相关性
        top_20 = self.correlation_df.head(20)
        top_20.to_csv('Top20特征相关性.csv', index=False, encoding='utf-8-sig')
        
        logger.info("相关性数据已保存为CSV文件")
    
    def run_analysis(self):
        """运行完整的相关性分析"""
        logger.info("开始特征相关性分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 数据清理
        self.clean_data()
        
        # 3. 计算相关性
        self.calculate_correlations()
        
        # 4. 生成可视化
        self.generate_visualizations()
        
        # 5. 生成报告
        report = self.generate_detailed_report()
        
        # 6. 保存数据
        self.save_correlation_data()
        
        logger.info("特征相关性分析完成！")
        
        return True

def main():
    """主函数"""
    analyzer = FeatureCorrelationAnalyzer('enriched_price_prediction_features.csv')
    success = analyzer.run_analysis()
    
    if success:
        print("\n" + "="*50)
        print("特征相关性分析完成！")
        print("生成的文件：")
        print("- 特征相关性分析.png (可视化图表)")
        print("- 特征相关性分析报告.md (详细报告)")
        print("- 特征相关性分析结果.csv (完整数据)")
        print("- Top20特征相关性.csv (重要特征)")
        print("="*50)
    else:
        print("分析过程中出现错误，请检查日志信息。")

if __name__ == "__main__":
    main() 
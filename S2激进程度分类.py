import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def setup_plotting_styles():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font='SimHei')

warnings.filterwarnings('ignore', category=FutureWarning)

def visualize_classification_results(df, df_transformed, output_prefix='classification_visualization'):

    print("\nGenerating visualizations...")
    
    palette = {'激进': '#FF6B6B', '稳健': '#4ECDC4', '保守': '#45B7D1'}
    
    # 根据可用特征调整图表布局
    available_features = [col for col in ['PeDI', 'R_it_timestamped', "D'_it_timestamped", 'weighted_avg_bid', 'capacity_utilization'] if col in df.columns]
    
    if len(available_features) >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('机组报价策略3分类结果可视化 (增强版)', fontsize=20, fontweight='bold')
        
        # 1. Strategy Distribution Pie Chart
        ax1 = axes[0, 0]
        strategy_counts = df['strategy_label'].value_counts()
        ax1.pie(strategy_counts, labels=strategy_counts.index, autopct='%1.1f%%', 
                colors=[palette.get(key, '#CCCCCC') for key in strategy_counts.index], startangle=90,
                wedgeprops=dict(width=0.4), textprops={'fontsize': 12})
        ax1.set_title('策略分布占比', fontsize=16, fontweight='bold')

        # 2. Core Features Distribution Box Plots
        ax2 = axes[0, 1]
        df_transformed['strategy_label'] = df['strategy_label'].values
        core_features = available_features[:3]  # 使用前3个核心特征
        df_melted = df_transformed[core_features + ['strategy_label']].melt(
            id_vars='strategy_label', var_name='Feature', value_name='Value')
        sns.boxplot(x='Feature', y='Value', hue='strategy_label', data=df_melted, 
                    palette=palette, ax=ax2, hue_order=['激进', '稳健', '保守'])
        ax2.set_title('核心特征分布 (分位数变换后)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('变换后的特征值')
        ax2.tick_params(axis='x', rotation=15)

        # 3. PeDI vs D'_it Scatter Plot
        ax3 = axes[0, 2]
        if 'PeDI' in available_features and "D'_it_timestamped" in available_features:
            df_plot = df_transformed.copy()
            sample_df = df_plot.sample(n=min(10000, len(df_plot)), random_state=42)
            sns.scatterplot(data=sample_df, x='PeDI', y="D'_it_timestamped", hue='strategy_label', 
                            palette=palette, alpha=0.7, s=50, ax=ax3, hue_order=['激进', '稳健', '保守'])
            ax3.set_title("PeDI vs D'_it 特征空间分布", fontsize=16, fontweight='bold')
            ax3.set_xlabel('相对同业报价偏差 (变换后)')
            ax3.set_ylabel("相对出清价偏差 (变换后)")

        # 4. Capacity Utilization vs Price Strategy
        ax4 = axes[1, 0]
        if 'capacity_utilization' in available_features and 'weighted_avg_bid' in available_features:
            sample_df = df.sample(n=min(10000, len(df)), random_state=42)
            sns.scatterplot(data=sample_df, x='capacity_utilization', y='weighted_avg_bid', 
                            hue='strategy_label', palette=palette, alpha=0.7, s=50, ax=ax4, 
                            hue_order=['激进', '稳健', '保守'])
            ax4.set_title('容量利用率 vs 加权平均报价', fontsize=16, fontweight='bold')
            ax4.set_xlabel('容量利用率')
            ax4.set_ylabel('加权平均报价')

        # 5. Time Pattern Analysis
        ax5 = axes[1, 1]
        if 'hour_of_day' in df.columns:
            hourly_strategy = df.groupby(['hour_of_day', 'strategy_label']).size().unstack(fill_value=0)
            hourly_strategy.plot(kind='bar', ax=ax5, color=[palette.get(col, '#CCCCCC') for col in hourly_strategy.columns])
            ax5.set_title('各时段策略分布', fontsize=16, fontweight='bold')
            ax5.set_xlabel('小时')
            ax5.set_ylabel('数据点数量')
            ax5.legend(title='策略类型')
            ax5.tick_params(axis='x', rotation=0)

        # 6. Probability Distribution
        ax6 = axes[1, 2]
        prob_cols = ['prob_激进', 'prob_稳健', 'prob_保守']
        for col in prob_cols:
            if col in df.columns:
                sns.kdeplot(df[col], ax=ax6, label=col.replace('prob_', ''), fill=True, alpha=0.6)
        ax6.set_title('各策略分类概率密度分布', fontsize=16, fontweight='bold')
        ax6.set_xlabel('概率值')
        ax6.set_ylabel('密度')
        ax6.legend()
        
    else:
        # 如果特征较少，使用原来的2x2布局
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle('机组报价策略3分类结果可视化', fontsize=20, fontweight='bold')
        
        # 1. Strategy Distribution Pie Chart
        ax1 = axes[0, 0]
        strategy_counts = df['strategy_label'].value_counts()
        ax1.pie(strategy_counts, labels=strategy_counts.index, autopct='%1.1f%%', 
                colors=[palette.get(key, '#CCCCCC') for key in strategy_counts.index], startangle=90,
                wedgeprops=dict(width=0.4), textprops={'fontsize': 12})
        ax1.set_title('策略分布占比', fontsize=16, fontweight='bold')

        # 2. Feature Distribution Box Plots
        ax2 = axes[0, 1]
        df_transformed['strategy_label'] = df['strategy_label'].values
        df_melted = df_transformed.melt(id_vars='strategy_label', var_name='Feature', value_name='Value')
        sns.boxplot(x='Feature', y='Value', hue='strategy_label', data=df_melted, 
                    palette=palette, ax=ax2, hue_order=['激进', '稳健', '保守'])
        ax2.set_title('特征分布 (分位数变换后)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('变换后的特征值')
        ax2.tick_params(axis='x', rotation=15)

        # 3. Scatter Plot
        ax3 = axes[1, 0]
        if len(available_features) >= 2:
            df_plot = df_transformed.copy()
            sample_df = df_plot.sample(n=min(10000, len(df_plot)), random_state=42)
            sns.scatterplot(data=sample_df, x=available_features[0], y=available_features[1], 
                            hue='strategy_label', palette=palette, alpha=0.7, s=50, ax=ax3, 
                            hue_order=['激进', '稳健', '保守'])
            ax3.set_title(f'{available_features[0]} vs {available_features[1]} 特征空间分布', fontsize=16, fontweight='bold')
            ax3.set_xlabel(f'{available_features[0]} (变换后)')
            ax3.set_ylabel(f'{available_features[1]} (变换后)')

        # 4. Probability Distribution
        ax4 = axes[1, 1]
        prob_cols = ['prob_激进', 'prob_稳健', 'prob_保守']
        for col in prob_cols:
            if col in df.columns:
                sns.kdeplot(df[col], ax=ax4, label=col.replace('prob_', ''), fill=True, alpha=0.6)
        ax4.set_title('各策略分类概率密度分布', fontsize=16, fontweight='bold')
        ax4.set_xlabel('概率值')
        ax4.set_ylabel('密度')
        ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_image_file = f'results/{output_prefix}.png'
    plt.savefig(output_image_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_image_file}")
    plt.show()

def classify_bidding_behavior(feature_file, output_file, n_clusters=3):

    try:
        print(f"Loading features from '{feature_file}'...")
        df = pd.read_csv(feature_file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if df.empty:
            print("Data is empty after cleaning. Aborting.")
            return
    except FileNotFoundError:
        print(f"Error: Input file not found at '{feature_file}'")
        return

    print("Applying Quantile Transformation to handle extreme outliers...")
    
    # 使用更多特征进行分类，提高分类准确性
    features_to_cluster = [
        'PeDI',                    # 核心特征：同类型偏差指数
        'R_it_timestamped',        # 核心特征：高价容量比例
        "D'_it_timestamped",       # 核心特征：相对报价偏差
        'weighted_avg_bid',        # 基础特征：加权平均报价
        'bid_price_std',           # 基础特征：报价标准差
        'capacity_utilization',    # 基础特征：容量利用率
        'clearing_price'           # 市场特征：出清价格
    ]
    
    # 过滤掉不存在的列
    available_features = [col for col in features_to_cluster if col in df.columns]
    if len(available_features) < 3:
        print(f"Warning: Only {len(available_features)} features available. Using minimum required features.")
        available_features = ['PeDI', 'R_it_timestamped', "D'_it_timestamped"]
    
    print(f"Using {len(available_features)} features for classification: {available_features}")
    
    X_original = df[available_features].copy()

    # 确保分位数数量不超过样本数量，且不超过10000
    n_quantiles = min(10000, len(df), max(100, int(len(df) / 1000)))
    transformer = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles, random_state=42)
    X_transformed = transformer.fit_transform(X_original)
    
    df_transformed_plot = pd.DataFrame(X_transformed, columns=available_features)

    print(f"Training Gaussian Mixture Model with {n_clusters} clusters on transformed data...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full', n_init=5)
    cluster_labels = gmm.fit_predict(X_transformed)
    probabilities = gmm.predict_proba(X_transformed)

    df['cluster'] = cluster_labels
    for i in range(n_clusters):
        df[f'prob_cluster_{i}'] = probabilities[:, i]

    print("Interpreting cluster characteristics using original feature values...")
    cluster_characteristics = df.groupby('cluster')[available_features].median().reset_index()
    
    # 基于核心特征进行排序和标签映射
    if "D'_it_timestamped" in available_features:
        cluster_characteristics = cluster_characteristics.sort_values(by="D'_it_timestamped", ascending=False)
    elif 'PeDI' in available_features:
        cluster_characteristics = cluster_characteristics.sort_values(by='PeDI', ascending=False)
    else:
        # 如果没有核心特征，使用第一个特征排序
        cluster_characteristics = cluster_characteristics.sort_values(by=available_features[0], ascending=False)
    
    # Mapping for 3 clusters
    mapping = {
        cluster_characteristics.iloc[0]['cluster']: '激进',
        cluster_characteristics.iloc[1]['cluster']: '稳健',
        cluster_characteristics.iloc[2]['cluster']: '保守'
    }
    
    df['strategy_label'] = df['cluster'].map(mapping)
    for cluster_num, label in mapping.items():
        df.rename(columns={f'prob_cluster_{int(cluster_num)}': f'prob_{label}'}, inplace=True)

    print(f"\nSaving final classified data to '{output_file}'...")
    
    # 保存更多有用的列
    final_cols = ['datetime', '机组名称', 'strategy_label', 'prob_激进', 'prob_稳健', 'prob_保守'] + available_features
    
    # 添加一些有用的基础特征
    additional_cols = ['clearing_price', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour']
    for col in additional_cols:
        if col in df.columns and col not in final_cols:
            final_cols.append(col)
    
    df_final = df[final_cols]
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"SUCCESS: Classification task complete. Output saved to '{output_file}'")
    
    generate_report(df_final, '激进程度分类结果')
    visualize_classification_results(df_final, df_transformed_plot, '激进程度分类结果')

def generate_report(df, output_prefix):
    """Generates and saves a detailed text report of the classification results."""
    print("\n=== 分类结果统计报告 ===")
    strategy_counts = df['strategy_label'].value_counts()
    print(f"总数据点: {len(df)}")
    print("\n策略分布:")
    for strategy in ['激进', '稳健', '保守']:
        if strategy in strategy_counts:
            count = strategy_counts[strategy]
            percentage = (count / len(df)) * 100
            print(f"  {strategy}: {count} 点 ({percentage:.1f}%)")
    
    print("\n各策略特征统计 (原始值):")
    
    # 确定要分析的特征
    core_features = ['PeDI', 'R_it_timestamped', "D'_it_timestamped"]
    additional_features = ['weighted_avg_bid', 'bid_price_std', 'capacity_utilization', 'clearing_price']
    
    # 过滤存在的特征
    available_core = [col for col in core_features if col in df.columns]
    available_additional = [col for col in additional_features if col in df.columns]
    
    if available_core:
        print("\n核心分类特征:")
        strategy_stats_core = df.groupby('strategy_label')[available_core].agg(['mean', 'median', 'std']).round(4)
        print(strategy_stats_core)
    
    if available_additional:
        print("\n辅助分析特征:")
        strategy_stats_additional = df.groupby('strategy_label')[available_additional].agg(['mean', 'median', 'std']).round(4)
        print(strategy_stats_additional)
    
    # 时间模式分析
    if 'hour_of_day' in df.columns:
        print("\n各时段策略分布:")
        hourly_strategy = df.groupby(['hour_of_day', 'strategy_label']).size().unstack(fill_value=0)
        print(hourly_strategy)
    
    if 'is_peak_hour' in df.columns:
        print("\n峰时vs平时策略分布:")
        peak_strategy = df.groupby(['is_peak_hour', 'strategy_label']).size().unstack(fill_value=0)
        print(peak_strategy)
    
    # 保存详细报告
    report_file = f'results/{output_prefix}_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 激进程度分类结果统计报告 ===\n\n")
        f.write(f"总数据点: {len(df)}\n\n")
        f.write("策略分布:\n")
        for strategy in ['激进', '稳健', '保守']:
            if strategy in strategy_counts:
                count = strategy_counts[strategy]
                percentage = (count / len(df)) * 100
                f.write(f"  {strategy}: {count} 点 ({percentage:.1f}%)\n")
        
        if available_core:
            f.write("\n核心分类特征:\n")
            f.write(strategy_stats_core.to_string())
        
        if available_additional:
            f.write("\n\n辅助分析特征:\n")
            f.write(strategy_stats_additional.to_string())
        
        if 'hour_of_day' in df.columns:
            f.write("\n\n各时段策略分布:\n")
            f.write(hourly_strategy.to_string())
        
        if 'is_peak_hour' in df.columns:
            f.write("\n\n峰时vs平时策略分布:\n")
            f.write(peak_strategy.to_string())
    
    print(f"\n统计报告已保存为: {report_file}")

if __name__ == '__main__':
    setup_plotting_styles()
    INPUT_FEATURE_FILE = 'results/unit_metrics_96_points_final.csv'
    OUTPUT_CLASSIFIED_FILE = 'results/unit_behavior_classified_96_points.csv'
    classify_bidding_behavior(INPUT_FEATURE_FILE, OUTPUT_CLASSIFIED_FILE)

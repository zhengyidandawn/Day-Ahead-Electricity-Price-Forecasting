import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
from datetime import datetime, timedelta

# --- 代码设置 ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
def create_results_dirs():
    """创建结果保存目录"""
    base_dir = 'results'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    model_dirs = ['SVR', 'XGBoost', 'GBDT']
    for model_name in model_dirs:
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    return base_dir


# --- 1. 加载和准备数据 ---
try:
    df = pd.read_csv(f'results/enriched_price_prediction_features.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()
    df = df.asfreq('15T').ffill()
    print("数据加载成功。")
except FileNotFoundError:
    print("错误：找不到文件 'enriched_price_prediction_features.csv'。")
    exit()

# --- 2. 严格的日前预测特征工程 ---
print("\n正在为日前预测场景进行特征工程，防止数据泄露...")
PREDICTION_HORIZON = 96
df['target_price'] = df['price'].shift(-PREDICTION_HORIZON)

# 删除可能导致数据泄露的特征
leaky_features = [
    'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_4',
    'daily_mean', 'daily_std', 'price_change', 'price_change_pct'
]
df_safe = df.drop(columns=leaky_features)

# 使用shift(1)确保滚动窗口不包含当前时刻的价格
df_safe['rolling_mean_24h'] = df_safe['price'].shift(1).rolling(window=PREDICTION_HORIZON, min_periods=1).mean()
df_safe['rolling_std_24h'] = df_safe['price'].shift(1).rolling(window=PREDICTION_HORIZON, min_periods=1).std()
df_safe['rolling_min_24h'] = df_safe['price'].shift(1).rolling(window=PREDICTION_HORIZON, min_periods=1).min()
df_safe['rolling_max_24h'] = df_safe['price'].shift(1).rolling(window=PREDICTION_HORIZON, min_periods=1).max()

# 创建更长期的历史统计特征
df_safe['rolling_mean_7d'] = df_safe['price'].shift(1).rolling(window=PREDICTION_HORIZON*7, min_periods=1).mean()
df_safe['rolling_std_7d'] = df_safe['price'].shift(1).rolling(window=PREDICTION_HORIZON*7, min_periods=1).std()

# 删除当前价格列，防止数据泄露
df_safe = df_safe.drop(columns=['price'])
print(f"新增安全特征: 24小时滚动统计(均值、标准差、最小值、最大值), 7天滚动统计(均值、标准差)")

df_safe = df_safe.dropna(subset=['target_price'])
df_safe = df_safe.dropna()
print("特征工程完成，数据已对齐。")


# --- 3. 定义新的特征和目标 ---
TARGET = 'target_price'
features = [col for col in df_safe.columns if col != TARGET]
X = df_safe[features]
y = df_safe[TARGET]

# --- 4. 时间序列分割 ---
split_ratio = 0.9
split_index = int(len(df_safe) * split_ratio)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
print(f"\n数据分割完成，测试集从 {X_test.index.min()} 开始。")

# --- 5. 模型训练与预测 ---
results = {}
models = {}
base_dir = create_results_dirs()

# SVR
print("\n[1/3] 开始训练 SVR 模型...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svr_model = SVR(kernel='rbf', C=100, gamma='auto')
svr_model.fit(X_train_scaled, y_train)
results['SVR'] = svr_model.predict(X_test_scaled)
models['SVR'] = svr_model
print("SVR 模型训练和预测完成。")

# XGBoost
print("\n[2/3] 开始训练 XGBoost 模型...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
    max_depth=5, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
results['XGBoost'] = xgb_model.predict(X_test)
models['XGBoost'] = xgb_model
print("XGBoost 模型训练和预测完成。")

# GBDT
print("\n[3/3] 开始训练 GBDT 模型...")
gbdt_model = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=42
)
gbdt_model.fit(X_train, y_train)
results['GBDT'] = gbdt_model.predict(X_test)
models['GBDT'] = gbdt_model
print("GBDT 模型训练和预测完成。")

# --- 6. 性能评估与对比 ---
print("\n--- 模型性能对比 (无数据泄露) ---")
evaluation_metrics = pd.DataFrame(columns=['MAE', 'RMSE', 'R2'])
for name, pred in results.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    evaluation_metrics.loc[name] = [mae, rmse, r2]
print(evaluation_metrics.sort_values(by='RMSE'))

# 保存评估结果
evaluation_metrics.to_csv(os.path.join(base_dir, 'model_performance_comparison.csv'))
print(f"模型性能对比已保存到 {os.path.join(base_dir, 'model_performance_comparison.csv')}")

# --- 7. 保存预测结果 ---
print("\n正在保存各模型预测结果...")
for model_name, pred_array in results.items():
    # 创建预测结果DataFrame
    pred_df = pd.DataFrame({
        'datetime': y_test.index,
        'actual_price': y_test.values,
        'predicted_price': pred_array,
        'error': y_test.values - pred_array,
        'error_pct': (y_test.values - pred_array) / y_test.values * 100
    })
    
    # 保存到对应模型目录
    model_dir = os.path.join(base_dir, model_name)
    pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)
    
    # 保存模型性能摘要
    performance_summary = {
        'MAE': mean_absolute_error(y_test, pred_array),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_array)),
        'R2': r2_score(y_test, pred_array),
        'MAPE': np.mean(np.abs((y_test.values - pred_array) / y_test.values)) * 100
    }
    
    summary_df = pd.DataFrame([performance_summary])
    summary_df.to_csv(os.path.join(model_dir, 'performance_summary.csv'), index=False)
    
    print(f"{model_name} 预测结果已保存到 {model_dir}/")

# --- 8. 三天预测结果可视化 ---
print("\n正在生成三天预测结果对比图...")

# 将所有预测结果转换为带有正确索引的Pandas Series
for model_name, pred_array in results.items():
    results[model_name] = pd.Series(pred_array, index=y_test.index)

# 选择要可视化的三天（测试集的前三天）
test_start_date = y_test.index.date[0]
dates_to_plot = [test_start_date + timedelta(days=i) for i in range(3)]

fig, axes = plt.subplots(3, 1, figsize=(18, 15))
fig.suptitle('三天预测结果对比', fontsize=16, fontweight='bold')

colors = {'SVR': 'green', 'XGBoost': 'red', 'GBDT': 'purple'}
styles = {'SVR': ':', 'XGBoost': '--', 'GBDT': '-.'}

for i, date in enumerate(dates_to_plot):
    date_str = date.strftime('%Y-%m-%d')
    print(f"绘制日期: {date_str}")
    
    # 筛选出该天的数据
    try:
        y_day = y_test[date_str]
        predictions_day = {name: pred[date_str] for name, pred in results.items()}
        
        # 绘制实际价格
        axes[i].plot(y_day.index, y_day, label='实际价格', color='black', linewidth=3, marker='.')
        
        # 绘制各模型预测
        for model_name, pred_day in predictions_day.items():
            axes[i].plot(pred_day.index, pred_day,
                        label=f'预测 - {model_name}',
                        color=colors.get(model_name),
                        linestyle=styles.get(model_name),
                        linewidth=2)
        
        axes[i].set_title(f'日期: {date_str}', fontsize=14)
        axes[i].set_ylabel('价格', fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # 设置x轴标签
        axes[i].tick_params(axis='x', rotation=45)
        
    except KeyError:
        print(f"警告: 日期 {date_str} 在测试集中不存在")
        axes[i].text(0.5, 0.5, f'日期 {date_str} 无数据', 
                    transform=axes[i].transAxes, ha='center', va='center')
        axes[i].set_title(f'日期: {date_str} (无数据)', fontsize=14)

axes[-1].set_xlabel('时间', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'three_days_prediction_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
print(f"三天预测对比图已保存到 {os.path.join(base_dir, 'three_days_prediction_comparison.png')}")

# --- 9. 特征重要性分析 ---
print("\n正在生成特征重要性分析...")

def plot_feature_importance(model, model_name, features, base_dir):
    """绘制特征重要性图"""
    if hasattr(model, 'feature_importances_'):
        # 树模型特征重要性
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 线性模型系数
        importance = np.abs(model.coef_)
    else:
        print(f"{model_name} 模型不支持特征重要性分析")
        return
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 保存特征重要性数据
    importance_df.to_csv(os.path.join(base_dir, model_name, 'feature_importance.csv'), index=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'{model_name} 模型 - Top 20 特征重要性')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, model_name, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"{model_name} 特征重要性分析已保存到 {os.path.join(base_dir, model_name)}/")
    
    return importance_df

# 为每个模型生成特征重要性分析
feature_importance_results = {}
for model_name, model in models.items():
    if model_name == 'SVR':
        # SVR需要特殊处理，使用基于排列的特征重要性
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
        importance = perm_importance.importances_mean
    else:
        # 树模型直接使用feature_importances_
        importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 保存特征重要性数据
    model_dir = os.path.join(base_dir, model_name)
    importance_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'{model_name} 模型 - Top 20 特征重要性')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    feature_importance_results[model_name] = importance_df
    print(f"{model_name} 特征重要性分析已保存到 {model_dir}/")

# --- 10. 生成综合报告 ---
print("\n正在生成综合分析报告...")

# 创建综合报告
report = []
report.append("# 多模型电价预测分析报告")
report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"数据样本数: {len(df_safe):,}")
report.append(f"训练集样本数: {len(X_train):,}")
report.append(f"测试集样本数: {len(X_test):,}")
report.append(f"特征数量: {len(features)}")
report.append("")

# 模型性能对比
report.append("## 1. 模型性能对比")
report.append("")
report.append("| 模型 | MAE | RMSE | R² |")
report.append("|------|-----|------|----|")
for model_name in evaluation_metrics.index:
    mae, rmse, r2 = evaluation_metrics.loc[model_name]
    report.append(f"| {model_name} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |")
report.append("")

# 最佳模型
best_model = evaluation_metrics['R2'].idxmax()
report.append(f"**最佳模型**: {best_model} (R² = {evaluation_metrics.loc[best_model, 'R2']:.4f})")
report.append("")

# 特征重要性总结
report.append("## 2. 特征重要性分析")
report.append("")

# 获取各模型的前10个重要特征
for model_name, importance_df in feature_importance_results.items():
    report.append(f"### {model_name} 模型 - Top 10 重要特征")
    top_10 = importance_df.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        report.append(f"{i}. **{row['feature']}** (重要性: {row['importance']:.4f})")
    report.append("")

# 共同重要特征分析
report.append("### 共同重要特征分析")
all_important_features = set()
for importance_df in feature_importance_results.values():
    top_10_features = set(importance_df.head(10)['feature'])
    all_important_features.update(top_10_features)

# 统计每个特征在各模型中的重要性排名
feature_rankings = {}
for feature in all_important_features:
    rankings = []
    for model_name, importance_df in feature_importance_results.items():
        if feature in importance_df['feature'].values:
            rank = importance_df[importance_df['feature'] == feature].index[0] + 1
            rankings.append(rank)
        else:
            rankings.append(len(importance_df) + 1)  # 未进入前10的标记为较大值
    feature_rankings[feature] = np.mean(rankings)

# 按平均排名排序
sorted_features = sorted(feature_rankings.items(), key=lambda x: x[1])
report.append("各模型平均重要性排名:")
for i, (feature, avg_rank) in enumerate(sorted_features[:15], 1):
    report.append(f"{i}. **{feature}** (平均排名: {avg_rank:.1f})")
report.append("")

# 保存综合报告
with open(os.path.join(base_dir, 'comprehensive_analysis_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"综合分析报告已保存到 {os.path.join(base_dir, 'comprehensive_analysis_report.md')}")

# --- 11. 输出总结 ---
print("\n" + "="*60)
print("多模型电价预测分析完成！")
print("="*60)
print("生成的文件和目录:")
print(f"- {base_dir}/")
print("  ├── model_performance_comparison.csv (模型性能对比)")
print("  ├── three_days_prediction_comparison.png (三天预测对比图)")
print("  ├── comprehensive_analysis_report.md (综合分析报告)")
print("  ├── SVR/")
print("  │   ├── predictions.csv (预测结果)")
print("  │   ├── performance_summary.csv (性能摘要)")
print("  │   └── feature_importance.png (特征重要性图)")
print("  ├── XGBoost/")
print("  │   ├── predictions.csv (预测结果)")
print("  │   ├── performance_summary.csv (性能摘要)")
print("  │   └── feature_importance.png (特征重要性图)")
print("  └── GBDT/")
print("      ├── predictions.csv (预测结果)")
print("      ├── performance_summary.csv (性能摘要)")
print("      └── feature_importance.png (特征重要性图)")
print("="*60)
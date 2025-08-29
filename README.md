### 环境要求

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- scipy

### 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
```

### 运行顺序

1. **S1生成分类特征.py** - 生成机组分类特征
2. **S2激进程度分类.py** - 进行激进程度分类
3. **S3融合特征.py** - 融合各类特征
4. **S4日前预测.py** - 执行日前价格预测，SVR\GBDT\XGB
5. **特征相关性分析.py** - 分析特征相关性（可选）

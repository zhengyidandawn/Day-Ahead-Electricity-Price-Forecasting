import pandas as pd
import numpy as np

def fuse_classification_features(classified_file, unit_info_file, base_feature_file, output_file):

    try:
        # --- 1. Load Data ---
        print("Loading classified behavior data and unit information...")
        df_classified = pd.read_csv(classified_file)
        
        # 处理Excel文件
        if unit_info_file.endswith('.xlsx'):
            df_units_info = pd.read_excel(unit_info_file)
        else:
            df_units_info = pd.read_csv(unit_info_file)
            
    except FileNotFoundError as e:
        print(f"Error loading input files: {e}")
        return

    # --- 2. Prepare Data for Aggregation ---
    print("Preparing data for aggregation...")
    # Get the latest unit info (Q_max and type)
    df_units_info['日期'] = pd.to_datetime(df_units_info['日期'])
    df_units_latest = df_units_info.sort_values('日期', ascending=False).drop_duplicates('机组名称')
    df_units_latest = df_units_latest[['机组名称', '机组类型', '额定容量']].rename(columns={'额定容量': 'Q_max'})

    # Merge Q_max and unit type into the classified data
    df_classified['datetime'] = pd.to_datetime(df_classified['datetime'])
    df_classified_enriched = pd.merge(df_classified, df_units_latest, on='机组名称', how='left')
    df_classified_enriched.dropna(subset=['Q_max', '机组类型'], inplace=True)

    # --- 3. Calculate Aggregated Macro Features ---
    print("Calculating aggregated macro-level features...")
    
    # --- Scheme 1 & 2: Strategy Capacity Ratios (Overall and Fuel-Specific) ---
    def calculate_ratios(df, group_key=None):
        prefix = f"{group_key.lower()}_" if group_key else ""
        
        # Calculate total capacity for the group at each timestamp
        total_capacity_per_timestamp = df.groupby('datetime')['Q_max'].sum()
        
        # Calculate capacity for each strategy at each timestamp
        strategy_capacity = df.groupby(['datetime', 'strategy_label'])['Q_max'].sum().unstack(fill_value=0)
        
        # Calculate ratios
        strategy_ratios = strategy_capacity.div(total_capacity_per_timestamp, axis=0)
        strategy_ratios.columns = [f"{prefix}{label.lower()}_capacity_ratio" for label in strategy_ratios.columns]
        return strategy_ratios

    # Overall market ratios
    df_overall_ratios = calculate_ratios(df_classified_enriched)

    # Fuel-specific ratios for Coal (燃煤) and Gas (燃气)
    df_coal_ratios = calculate_ratios(df_classified_enriched[df_classified_enriched['机组类型'] == '燃煤'], 'coal')
    df_gas_ratios = calculate_ratios(df_classified_enriched[df_classified_enriched['机组类型'] == '燃气'], 'gas')

    # --- Scheme 3: Continuous Market Sentiment Index ---
    df_classified_enriched['prob_aggressive_mw'] = df_classified_enriched['prob_激进'] * df_classified_enriched['Q_max']
    df_classified_enriched['prob_steady_mw'] = df_classified_enriched['prob_稳健'] * df_classified_enriched['Q_max']
    df_classified_enriched['prob_conservative_mw'] = df_classified_enriched['prob_保守'] * df_classified_enriched['Q_max']

    df_sentiment = df_classified_enriched.groupby('datetime').agg(
        total_capacity=pd.NamedAgg(column='Q_max', aggfunc='sum'),
        prob_aggressive_mw_sum=pd.NamedAgg(column='prob_aggressive_mw', aggfunc='sum'),
        prob_steady_mw_sum=pd.NamedAgg(column='prob_steady_mw', aggfunc='sum'),
        prob_conservative_mw_sum=pd.NamedAgg(column='prob_conservative_mw', aggfunc='sum')
    )
    
    df_sentiment['market_prob_aggressive'] = df_sentiment['prob_aggressive_mw_sum'] / df_sentiment['total_capacity']
    df_sentiment['market_prob_steady'] = df_sentiment['prob_steady_mw_sum'] / df_sentiment['total_capacity']
    df_sentiment['market_prob_conservative'] = df_sentiment['prob_conservative_mw_sum'] / df_sentiment['total_capacity']
    
    # --- 4. Fuse All Features ---
    print("Fusing new features with the base prediction matrix...")
    
    # Combine all new aggregated features into one dataframe
    df_macro_features = pd.concat([
        df_overall_ratios,
        df_coal_ratios,
        df_gas_ratios,
        df_sentiment[['market_prob_aggressive', 'market_prob_steady', 'market_prob_conservative']]
    ], axis=1).reset_index()
    
    # Load the base feature matrix
    if base_feature_file.endswith('.xlsx'):
        df_base = pd.read_excel(base_feature_file)
    else:
        df_base = pd.read_csv(base_feature_file)
    df_base['datetime'] = pd.to_datetime(df_base['datetime'])
    
    # Merge the new macro features
    df_enriched = pd.merge(df_base, df_macro_features, on='datetime', how='left')
    df_enriched.fillna(0, inplace=True) # Fill NaNs for timestamps where a category might be missing

    # --- 5. Save Final Result ---
    df_enriched.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSUCCESS: Enriched feature matrix saved to '{output_file}'")
    print("\n--- Columns added ---")
    print(list(df_macro_features.columns.drop('datetime')))
    print("\n--- Sample of the final data ---")
    print(df_enriched.head())


if __name__ == '__main__':
    # Define file paths - 适配当前项目文件
    CLASSIFIED_BEHAVIOR_FILE = 'results/unit_behavior_classified_96_points.csv'
    UNIT_INFO_FILE = 'data/units_with_cost.csv'
    BASE_FEATURE_FILE = 'tools/日前节点价格 全省_特征矩阵-new.xlsx'
    OUTPUT_ENRICHED_FILE = 'results/enriched_price_prediction_features.csv'
    
    # Run the fusion process
    fuse_classification_features(CLASSIFIED_BEHAVIOR_FILE, UNIT_INFO_FILE, BASE_FEATURE_FILE, OUTPUT_ENRICHED_FILE)

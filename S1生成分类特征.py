import pandas as pd
import numpy as np

def load_and_prepare_data(bid_file, unit_info_file, price_file):

    print("Loading and preparing data...")
    
    # Load CSV files
    df_bids_raw = pd.read_csv(bid_file)
    df_units_info = pd.read_csv(unit_info_file)
    df_prices_raw = pd.read_csv(price_file)
    
    print("All files loaded successfully.")

    # Process Bidding Data (Wide to Long)
    df_bids_raw['日期'] = pd.to_datetime(df_bids_raw['日期'])
    bid_segments = []
    for i in range(1, 12):
        segment_cols = {'start': f'起始出力段{i}', 'end': f'终止出力段{i}', 'price': f'报价{i}'}
        df_segment = df_bids_raw[['日期', '机组名称'] + list(segment_cols.values())].dropna()
        df_segment.rename(columns={v: k for k, v in segment_cols.items()}, inplace=True)
        df_segment['capacity'] = df_segment['end'] - df_segment['start']
        bid_segments.append(df_segment)
    df_bids_long = pd.concat(bid_segments, ignore_index=True)
    df_bids_long = df_bids_long[df_bids_long['capacity'] > 0]

    # Process Unit Info - include variable cost
    df_units_info['日期'] = pd.to_datetime(df_units_info['日期'])
    df_units_latest = df_units_info.sort_values('日期', ascending=False).drop_duplicates('机组名称')
    df_units_latest = df_units_latest[['机组名称', '机组类型', '额定容量', '变动成本电价']].rename(columns={'额定容量': 'Q_max', '变动成本电价': 'variable_cost'})

    # Process 96-Point Clearing Prices
    df_prices_long = df_prices_raw.melt(id_vars=['日期', '节点名称', '数据项'], var_name='time', value_name='clearing_price')
    df_prices_long = df_prices_long[df_prices_long['数据项'] == '电价']
    # Convert date to string before concatenation, then back to datetime
    df_prices_long['日期_str'] = df_prices_long['日期'].astype(str)
    df_prices_long['datetime'] = pd.to_datetime(df_prices_long['日期_str'] + ' ' + df_prices_long['time'])
    df_prices_96 = df_prices_long[['datetime', 'clearing_price']].copy()
    df_prices_96['日期'] = pd.to_datetime(df_prices_96['datetime'].dt.date)
    
    return df_bids_long, df_units_latest, df_prices_96

def calculate_daily_static_features(df_bids_long, df_units_latest):

    print("Calculating daily static features (PeDI)...")
    # Calculate Daily Weighted Average Bid Price
    df_bids_long['price_x_capacity'] = df_bids_long['price'] * df_bids_long['capacity']
    df_unit_daily_bids = df_bids_long.groupby(['日期', '机组名称']).agg(
        total_declared_capacity=pd.NamedAgg(column='capacity', aggfunc='sum'),
        total_price_x_capacity=pd.NamedAgg(column='price_x_capacity', aggfunc='sum')
    ).reset_index()
    df_unit_daily_bids['weighted_avg_bid'] = df_unit_daily_bids['total_price_x_capacity'] / df_unit_daily_bids['total_declared_capacity']

    # Calculate PeDI
    df_daily_metrics = pd.merge(df_unit_daily_bids, df_units_latest, on='机组名称', how='left').dropna(subset=['机组类型', 'Q_max'])
    df_peer_avg = df_daily_metrics.groupby(['日期', '机组类型'])['weighted_avg_bid'].mean().reset_index().rename(columns={'weighted_avg_bid': 'peer_avg_bid'})
    df_daily_metrics = pd.merge(df_daily_metrics, df_peer_avg, on=['日期', '机组类型'], how='left')
    df_daily_metrics['PeDI'] = (df_daily_metrics['weighted_avg_bid'] - df_daily_metrics['peer_avg_bid']) / df_daily_metrics['peer_avg_bid']
    
    return df_daily_metrics

def main():
 
    # --- Define File Paths ---
    BID_DATA_FILE = 'data/bids.csv'
    UNIT_INFO_FILE = 'data/units_with_cost.csv'
    PRICE_DATA_FILE = 'data/prices.csv'
    OUTPUT_FILE = 'results/unit_metrics_96_points_final.csv'

    try:
        # --- Step 1: Load and Prepare Data ---
        df_bids_long, df_units_latest, df_prices_96 = load_and_prepare_data(BID_DATA_FILE, UNIT_INFO_FILE, PRICE_DATA_FILE)

        # --- Step 2: Calculate Daily Static Features ---
        df_daily_metrics = calculate_daily_static_features(df_bids_long, df_units_latest)

        # --- Step 3: Build the 96-Point Feature Matrix ---
        print("Building the 96-point feature matrix...")
        unit_day_combinations = df_daily_metrics[['日期', '机组名称']].drop_duplicates()
  
        df_96_base = pd.merge(df_prices_96, unit_day_combinations, on='日期', how='inner')
        df_96_full = pd.merge(df_96_base, df_daily_metrics, on=['日期', '机组名称'], how='left')
        df_96_full = pd.merge(df_96_full, df_bids_long, on=['日期', '机组名称'], how='left')
        df_96_full = pd.merge(df_96_full, df_units_latest[['机组名称', 'variable_cost']], on='机组名称', how='left')
        
        # 处理列名冲突，重命名重复的列
        if 'variable_cost_x' in df_96_full.columns and 'variable_cost_y' in df_96_full.columns:
            df_96_full['variable_cost'] = df_96_full['variable_cost_y']
            df_96_full.drop(['variable_cost_x', 'variable_cost_y'], axis=1, inplace=True)
        elif 'variable_cost_x' in df_96_full.columns:
            df_96_full['variable_cost'] = df_96_full['variable_cost_x']
            df_96_full.drop('variable_cost_x', axis=1, inplace=True)

        # --- Step 4: Calculate 96-Point Dynamic Metrics ---
        print("Calculating 96-point dynamic metrics...")
        
        # 检查数据完整性
        print(f"df_96_full columns: {df_96_full.columns.tolist()}")
        print(f"df_96_full shape: {df_96_full.shape}")
        
        # R_it (Timestamped) - Modified high price capacity identification
        # High price capacity: bid price > clearing price AND bid price > variable cost
        df_96_full['high_price_capacity'] = np.where(
            (df_96_full['price'] > df_96_full['clearing_price']) & 
            (df_96_full['price'] > df_96_full['variable_cost']), 
            df_96_full['capacity'], 0
        )
        df_r_it_96 = df_96_full.groupby(['datetime', '机组名称', 'Q_max'])['high_price_capacity'].sum().reset_index()
        df_r_it_96['R_it_timestamped'] = df_r_it_96['high_price_capacity'] / df_r_it_96['Q_max']
        
        # D'_it (Timestamped) - Modified calculation
        df_d_it_96 = df_96_full[['datetime', '机组名称', 'weighted_avg_bid', 'clearing_price', 'variable_cost']].drop_duplicates()
        # Use max(variable_cost, clearing_price) instead of clearing_price
        df_d_it_96['max_price'] = np.maximum(df_d_it_96['variable_cost'], df_d_it_96['clearing_price'])
        df_d_it_96["D'_it_timestamped"] = (df_d_it_96['weighted_avg_bid'] - df_d_it_96['max_price']) / df_d_it_96['max_price']

        # --- Step 5: Final Assembly ---
        print("Finalizing and saving results...")
        
        # 计算更多分类特征
        df_static_final = df_daily_metrics[['日期', '机组名称', 'PeDI', 'weighted_avg_bid', 'total_declared_capacity', 'Q_max']]
        
        # 计算容量利用率
        df_static_final['capacity_utilization'] = df_static_final['total_declared_capacity'] / df_static_final['Q_max']
        
        # 计算报价统计特征
        df_bid_stats = df_bids_long.groupby(['日期', '机组名称']).agg({
            'price': ['mean', 'std', 'min', 'max'],
            'capacity': 'sum'
        }).reset_index()
        df_bid_stats.columns = ['日期', '机组名称', 'bid_price_mean', 'bid_price_std', 'bid_price_min', 'bid_price_max', 'total_capacity']
        df_bid_stats['bid_price_range'] = df_bid_stats['bid_price_max'] - df_bid_stats['bid_price_min']
        
 
        
        # 添加时间特征
        df_prices_96['hour_of_day'] = df_prices_96['datetime'].dt.hour
        df_prices_96['day_of_week'] = df_prices_96['datetime'].dt.dayofweek
        df_prices_96['is_weekend'] = df_prices_96['day_of_week'].isin([5, 6]).astype(int)
        df_prices_96['is_peak_hour'] = df_prices_96['hour_of_day'].isin([8, 9, 10, 11, 18, 19, 20, 21]).astype(int)
        
        # 合并所有特征
        final_df = pd.merge(df_prices_96[['datetime', '日期', 'clearing_price', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour']], 
                           unit_day_combinations, on='日期', how='inner')
        final_df = pd.merge(final_df, df_static_final, on=['日期', '机组名称'], how='left')
        final_df = pd.merge(final_df, df_bid_stats, on=['日期', '机组名称'], how='left')
        final_df = pd.merge(final_df, df_r_it_96[['datetime', '机组名称', 'R_it_timestamped']], on=['datetime', '机组名称'], how='left')
        final_df = pd.merge(final_df, df_d_it_96[['datetime', '机组名称', "D'_it_timestamped"]], on=['datetime', '机组名称'], how='left')
        
        # 最终特征列
        final_cols = [
            'datetime', '机组名称', 
            # 核心分类特征
            'PeDI', 'R_it_timestamped', "D'_it_timestamped",
            # 基础报价特征
            'weighted_avg_bid', 'bid_price_mean', 'bid_price_std', 'bid_price_range',
            'capacity_utilization', 'total_declared_capacity', 'Q_max',
            # 市场环境特征
            'clearing_price', 
            # 时间模式特征
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour'
        ]
        
        final_df = final_df[final_cols]
        final_df.fillna(0, inplace=True)
        final_df.sort_values(by=['机组名称', 'datetime'], inplace=True)
        
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nSUCCESS: All metrics calculated and saved to '{OUTPUT_FILE}'")
        print("\n--- Sample of the final output data ---")
        print(final_df.head())

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: File not found -> {e}.")
        print("This confirms a persistent issue with the environment's file access.")
        print("RECOMMENDATION: Please rename your files to simple names (e.g., 'bids.csv', 'units.csv', 'prices.csv'), re-upload them, and update the filenames in this script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    main()

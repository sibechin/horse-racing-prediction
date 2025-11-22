"""
歴史的G1データ(2016-2024)の前処理
2025年データと同じ特徴量エンジニアリングを適用
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def categorize_distance(dist):
    """距離カテゴリ分類"""
    if dist < 1400:
        return 0  # sprint
    elif dist < 1800:
        return 1  # mile
    elif dist < 2200:
        return 2  # middle
    else:
        return 3  # long

def preprocess_historical_g1(raw_data_path: str, train_data_path: str, output_path: str):
    """
    歴史的G1データの前処理

    Args:
        raw_data_path: 生データパス (2016-2024 G1)
        train_data_path: 学習データパス (統計情報取得用)
        output_path: 出力パス
    """
    logging.info("="*60)
    logging.info("歴史的G1データ前処理開始")
    logging.info("="*60)

    # データ読み込み
    df_g1 = pd.read_csv(raw_data_path, encoding='utf-8-sig')
    df_train = pd.read_csv(train_data_path, encoding='utf-8-sig')

    logging.info(f"\nG1データ: {len(df_g1)} records, {df_g1['race_id'].nunique()} races")
    logging.info(f"学習データ: {len(df_train)} records, {df_train['race_id'].nunique()} races")

    # 基本特徴量の抽出
    logging.info("\n基本特徴量の抽出...")

    # Sex and Age
    if 'sex_age' in df_g1.columns:
        df_g1['sex'] = df_g1['sex_age'].str[0]
        df_g1['age'] = df_g1['sex_age'].str[1:].astype(int)

    # Date features
    if 'race_date' in df_g1.columns:
        df_g1['race_date'] = pd.to_datetime(df_g1['race_date'])
        df_g1['year'] = df_g1['race_date'].dt.year
        df_g1['month'] = df_g1['race_date'].dt.month
        df_g1['day_of_week'] = df_g1['race_date'].dt.dayofweek

    # Distance category
    if 'distance' in df_g1.columns:
        df_g1['distance_category'] = df_g1['distance'].apply(categorize_distance)

    # Popularity rank (human_ranking)
    if 'popularity' in df_g1.columns:
        df_g1['popularity_rank'] = df_g1['popularity']

    # Time to seconds
    if 'time' in df_g1.columns:
        def time_to_seconds(time_str):
            if pd.isna(time_str) or time_str == '':
                return np.nan
            try:
                if ':' in str(time_str):
                    parts = str(time_str).split(':')
                    if len(parts) == 2:
                        minutes = float(parts[0])
                        seconds = float(parts[1])
                        return minutes * 60 + seconds
            except:
                pass
            return np.nan

        df_g1['time_seconds'] = df_g1['time'].apply(time_to_seconds)

    # Field size (レース内の出走頭数)
    df_g1['field_size'] = df_g1.groupby('race_id')['horse_name'].transform('count')

    # Odds: convert to numeric (handle corrupted data)
    if 'odds' in df_g1.columns:
        df_g1['odds'] = pd.to_numeric(df_g1['odds'], errors='coerce')

    # Race average odds
    if 'odds' in df_g1.columns:
        df_g1['race_avg_odds'] = df_g1.groupby('race_id')['odds'].transform('mean')

    # エンコーディング
    logging.info("カテゴリカル変数のエンコーディング...")

    # Track type
    track_type_map = {'芝': 0, 'ダート': 1}
    if 'track_type' in df_g1.columns:
        df_g1['track_type_encoded'] = df_g1['track_type'].map(track_type_map).fillna(0).astype(int)

    # Track condition
    track_cond_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    if 'track_condition' in df_g1.columns:
        df_g1['track_condition_encoded'] = df_g1['track_condition'].map(track_cond_map).fillna(0).astype(int)

    # Weather
    weather_map = {'晴': 0, '曇': 1, '雨': 2, '小雨': 2, '雪': 3}
    if 'weather' in df_g1.columns:
        df_g1['weather_encoded'] = df_g1['weather'].map(weather_map).fillna(0).astype(int)

    # Sex
    sex_map = {'牡': 0, '牝': 1, 'セ': 2}
    if 'sex' in df_g1.columns:
        df_g1['sex_encoded'] = df_g1['sex'].map(sex_map).fillna(0).astype(int)

    # Distance category encoding (already numeric, but ensure correct type)
    if 'distance_category' in df_g1.columns:
        df_g1['distance_category_encoded'] = df_g1['distance_category'].astype(int)

    # Season encoding
    if 'month' in df_g1.columns:
        def get_season(month):
            if month in [3, 4, 5]:
                return 0  # spring
            elif month in [6, 7, 8]:
                return 1  # summer
            elif month in [9, 10, 11]:
                return 2  # fall
            else:
                return 3  # winter
        df_g1['season_encoded'] = df_g1['month'].apply(get_season)

    # Track combined (track_type × track_condition)
    if 'track_type_encoded' in df_g1.columns and 'track_condition_encoded' in df_g1.columns:
        df_g1['track_combined_encoded'] = df_g1['track_type_encoded'] * 10 + df_g1['track_condition_encoded']

    # Time weight (exponential decay - 新しいレースほど重要)
    if 'race_date' in df_g1.columns:
        max_date = df_g1['race_date'].max()
        df_g1['days_ago'] = (max_date - df_g1['race_date']).dt.days
        # Half-life = 365 days
        df_g1['time_weight'] = np.exp(-0.693 * df_g1['days_ago'] / 365.0)
    else:
        df_g1['time_weight'] = 1.0

    # 歴史的特徴量 (学習データから統計情報を取得)
    logging.info("\n歴史的特徴量の計算...")

    # Jockey stats (from training data)
    if 'jockey_name' in df_train.columns:
        jockey_stats = df_train.groupby('jockey_name').agg({
            'finish_position': lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0,  # win rate
        }).reset_index()
        jockey_stats.columns = ['jockey_name', 'jockey_win_rate']

        # Top3 rate
        jockey_top3 = df_train.groupby('jockey_name').agg({
            'finish_position': lambda x: (x <= 3).sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        jockey_top3.columns = ['jockey_name', 'jockey_top3_rate']

        jockey_stats = jockey_stats.merge(jockey_top3, on='jockey_name', how='outer')

        # Merge with G1 data
        df_g1 = df_g1.merge(jockey_stats, on='jockey_name', how='left')

    # Jockey track-specific win rate
    if 'jockey_name' in df_train.columns and 'venue_name' in df_train.columns:
        jockey_track_stats = df_train.groupby(['jockey_name', 'venue_name']).agg({
            'finish_position': lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        jockey_track_stats.columns = ['jockey_name', 'venue_name', 'jockey_track_win_rate']

        df_g1 = df_g1.merge(jockey_track_stats, on=['jockey_name', 'venue_name'], how='left')

    # Horse stats
    if 'horse_name' in df_train.columns:
        horse_stats = df_train.groupby('horse_name').agg({
            'finish_position': [
                lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0,  # win rate
                lambda x: (x <= 3).sum() / len(x) if len(x) > 0 else 0,  # top3 rate
                'count'  # race count
            ]
        }).reset_index()
        horse_stats.columns = ['horse_name', 'horse_win_rate', 'horse_top3_rate', 'horse_race_count']

        df_g1 = df_g1.merge(horse_stats, on='horse_name', how='left')

    # Horse distance-specific win rate
    if 'horse_name' in df_train.columns and 'distance' in df_train.columns:
        # Categorize distance for grouping
        df_train['distance_category_temp'] = df_train['distance'].apply(categorize_distance)

        horse_dist_stats = df_train.groupby(['horse_name', 'distance_category_temp']).agg({
            'finish_position': lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        horse_dist_stats.columns = ['horse_name', 'distance_category', 'horse_dist_win_rate']

        # Ensure distance_category is int
        horse_dist_stats['distance_category'] = horse_dist_stats['distance_category'].astype(int)
        df_g1['distance_category'] = df_g1['distance_category'].astype(int)

        df_g1 = df_g1.merge(horse_dist_stats, on=['horse_name', 'distance_category'], how='left')

    # 欠損値処理
    logging.info("\n欠損値の補完...")

    # Numeric features to impute
    numeric_features = [
        'jockey_win_rate', 'jockey_top3_rate', 'jockey_track_win_rate',
        'horse_win_rate', 'horse_top3_rate', 'horse_dist_win_rate',
        'horse_race_count', 'time_seconds', 'odds', 'race_avg_odds'
    ]

    for feat in numeric_features:
        if feat in df_g1.columns:
            median_val = df_g1[feat].median()
            if pd.isna(median_val):
                median_val = 0
            df_g1[feat] = df_g1[feat].fillna(median_val)

    # Integer features
    int_features = [
        'distance', 'weight', 'age', 'horse_number', 'popularity', 'popularity_rank',
        'field_size', 'year', 'month', 'day_of_week',
        'track_type_encoded', 'track_condition_encoded', 'weather_encoded',
        'sex_encoded', 'distance_category_encoded', 'season_encoded',
        'track_combined_encoded', 'venue_code'
    ]

    for feat in int_features:
        if feat in df_g1.columns:
            df_g1[feat] = df_g1[feat].fillna(0).astype(int)

    # 必要な特徴量リスト (29 features)
    required_features = [
        'distance', 'venue_code', 'horse_number', 'weight', 'popularity', 'odds',
        'time_seconds', 'age', 'year', 'jockey_win_rate', 'jockey_top3_rate',
        'jockey_track_win_rate', 'horse_win_rate', 'horse_top3_rate',
        'horse_race_count', 'horse_dist_win_rate', 'field_size', 'race_avg_odds',
        'popularity_rank', 'month', 'day_of_week', 'track_type_encoded',
        'track_condition_encoded', 'weather_encoded', 'sex_encoded',
        'distance_category_encoded', 'season_encoded', 'track_combined_encoded',
        'time_weight'
    ]

    # 特徴量の存在確認
    missing_features = [f for f in required_features if f not in df_g1.columns]
    if missing_features:
        logging.warning(f"⚠️ 不足している特徴量: {missing_features}")
        for feat in missing_features:
            df_g1[feat] = 0

    available_features = [f for f in required_features if f in df_g1.columns]
    logging.info(f"利用可能な特徴量: {len(available_features)}/{len(required_features)}")

    # 保存に必要な追加カラム
    keep_columns = [
        'race_id', 'race_name', 'race_date', 'venue_name',
        'horse_name', 'jockey_name', 'finish_position'
    ] + required_features

    # 実際に存在するカラムのみ選択
    keep_columns = [c for c in keep_columns if c in df_g1.columns]
    df_output = df_g1[keep_columns].copy()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')

    logging.info("\n"+"="*60)
    logging.info("前処理完了!")
    logging.info("="*60)
    logging.info(f"出力レコード数: {len(df_output)}")
    logging.info(f"レース数: {df_output['race_id'].nunique()}")
    logging.info(f"カラム数: {len(df_output.columns)}")
    logging.info(f"特徴量数: {len([f for f in required_features if f in df_output.columns])}")
    logging.info(f"保存先: {output_path}")

    return df_output


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("歴史的G1データ前処理")
    logging.info("="*60)

    # パス設定
    raw_data_path = "data/raw/g1_races_2016_2024_historical.csv"
    train_data_path = "data/processed/features_engineered.csv"
    output_path = "data/processed/g1_races_historical_processed.csv"

    # 前処理実行
    df_processed = preprocess_historical_g1(raw_data_path, train_data_path, output_path)

    # 統計情報
    if df_processed is not None and len(df_processed) > 0:
        logging.info("\n統計情報:")
        logging.info(f"期間: {df_processed['race_date'].min()} - {df_processed['race_date'].max()}")

        if 'year' in df_processed.columns:
            logging.info(f"\n年別レース数:")
            year_counts = df_processed.groupby('year')['race_id'].nunique()
            for year, count in year_counts.items():
                logging.info(f"  {year}: {count}レース")

        if 'venue_name' in df_processed.columns:
            logging.info(f"\n競馬場別:")
            venue_counts = df_processed['venue_name'].value_counts()
            for venue, count in venue_counts.items():
                logging.info(f"  {venue}: {count}頭")

        if 'track_type_encoded' in df_processed.columns:
            track_counts = df_processed['track_type_encoded'].value_counts()
            logging.info(f"\n馬場タイプ別:")
            logging.info(f"  芝: {track_counts.get(0, 0)}頭")
            logging.info(f"  ダート: {track_counts.get(1, 0)}頭")


if __name__ == "__main__":
    main()

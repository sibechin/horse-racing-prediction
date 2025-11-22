"""
2025年G1レースデータの前処理スクリプト

訓練データの統計情報を使用して特徴量エンジニアリングを実行
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print('='*60)
print('2025年G1レースデータ 前処理')
print('='*60)
print()

# ==========================================
# 1. データ読み込み
# ==========================================
logging.info("データ読み込み中...")

# 2025年G1データ
df_2025 = pd.read_csv('data/raw/g1_races_2025_all.csv', encoding='utf-8-sig')
logging.info(f"2025年G1データ: {len(df_2025)}レコード, {len(df_2025.columns)}カラム")

# 訓練データ（特徴量エンジニアリング済み）
df_train = pd.read_csv('data/processed/features_engineered.csv', encoding='utf-8-sig')
logging.info(f"訓練データ: {len(df_train)}レコード, {len(df_train.columns)}カラム")

# ==========================================
# 2. 基本的な特徴量を作成
# ==========================================
logging.info("\n基本特徴量を作成中...")

# 日付変換
df_2025['race_date'] = pd.to_datetime(df_2025['race_date'])

# sex_ageから性別と年齢を分離
df_2025['sex'] = df_2025['sex_age'].str[0]
df_2025['age'] = df_2025['sex_age'].str[1:].astype(int)

# 時間をtime_secondsに変換（'1:23.4' -> 秒）
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return np.nan

df_2025['time_seconds'] = df_2025['time'].apply(time_to_seconds)

# 時間特徴量
df_2025['year'] = df_2025['race_date'].dt.year
df_2025['month'] = df_2025['race_date'].dt.month
df_2025['day_of_week'] = df_2025['race_date'].dt.dayofweek

# 距離カテゴリ
def categorize_distance(distance):
    if distance <= 1400:
        return 0  # 短距離
    elif distance <= 1800:
        return 1  # マイル
    elif distance <= 2200:
        return 2  # 中距離
    else:
        return 3  # 長距離

df_2025['distance_category'] = df_2025['distance'].apply(categorize_distance).astype(int)

# 季節
def get_season(month):
    if month in [3, 4, 5]:
        return 0  # spring
    elif month in [6, 7, 8]:
        return 1  # summer
    elif month in [9, 10, 11]:
        return 2  # fall
    else:
        return 3  # winter

df_2025['season'] = df_2025['month'].apply(get_season)

# 競馬場と馬場タイプの組み合わせ
df_2025['track_combined'] = df_2025['venue_name'].astype(str) + '_' + df_2025['track_type'].astype(str)

logging.info(f"  年範囲: {df_2025['year'].min()} - {df_2025['year'].max()}")
logging.info(f"  月範囲: {df_2025['month'].min()} - {df_2025['month'].max()}")
logging.info(f"  距離範囲: {df_2025['distance'].min()}m - {df_2025['distance'].max()}m")

# ==========================================
# 3. レース内特徴量
# ==========================================
logging.info("\nレース内特徴量を作成中...")

# 出走頭数
field_size = df_2025.groupby('race_id').size().reset_index(name='field_size')
df_2025 = df_2025.merge(field_size, on='race_id', how='left')

# レース平均オッズ
race_avg_odds = df_2025.groupby('race_id')['odds'].mean().reset_index(name='race_avg_odds')
df_2025 = df_2025.merge(race_avg_odds, on='race_id', how='left')

# 人気順位（オッズベース）
df_2025['popularity_rank'] = df_2025.groupby('race_id')['odds'].rank(method='min')

logging.info(f"  出走頭数範囲: {df_2025['field_size'].min():.0f} - {df_2025['field_size'].max():.0f}")
logging.info(f"  人気順位範囲: {df_2025['popularity_rank'].min():.0f} - {df_2025['popularity_rank'].max():.0f}")

# ==========================================
# 4. 騎手・馬の履歴特徴量を訓練データから取得
# ==========================================
logging.info("\n騎手・馬の履歴特徴量を訓練データから取得中...")

# 騎手統計
jockey_stats = df_train.groupby('jockey_name').agg({
    'jockey_win_rate': 'mean',
    'jockey_top3_rate': 'mean'
}).reset_index()

df_2025 = df_2025.merge(
    jockey_stats,
    on='jockey_name',
    how='left'
)

# 騎手x馬場タイプ統計
jockey_track_stats = df_train.groupby(['jockey_name', 'track_type']).agg({
    'jockey_track_win_rate': 'mean'
}).reset_index()

df_2025 = df_2025.merge(
    jockey_track_stats,
    on=['jockey_name', 'track_type'],
    how='left'
)

# 馬統計
horse_stats = df_train.groupby('horse_id').agg({
    'horse_win_rate': 'mean',
    'horse_top3_rate': 'mean',
    'horse_race_count': 'max'
}).reset_index()

df_2025 = df_2025.merge(
    horse_stats,
    on='horse_id',
    how='left'
)

# 馬x距離カテゴリ統計
# 型を一致させる (文字列から数値へ変換)
if 'distance_category' in df_train.columns:
    distance_cat_map = {'sprint': 0, 'mile': 1, 'middle': 2, 'long': 3}
    df_train['distance_category'] = df_train['distance_category'].replace(distance_cat_map).astype(int)

horse_dist_stats = df_train.groupby(['horse_id', 'distance_category']).agg({
    'horse_dist_win_rate': 'mean'
}).reset_index()

df_2025 = df_2025.merge(
    horse_dist_stats,
    on=['horse_id', 'distance_category'],
    how='left'
)

# 新人騎手・馬の場合は訓練データの中央値で補完
median_values = {
    'jockey_win_rate': df_train['jockey_win_rate'].median(),
    'jockey_top3_rate': df_train['jockey_top3_rate'].median(),
    'jockey_track_win_rate': df_train['jockey_track_win_rate'].median(),
    'horse_win_rate': df_train['horse_win_rate'].median(),
    'horse_top3_rate': df_train['horse_top3_rate'].median(),
    'horse_race_count': df_train['horse_race_count'].median(),
    'horse_dist_win_rate': df_train['horse_dist_win_rate'].median()
}

for col, median_val in median_values.items():
    df_2025[col] = df_2025[col].fillna(median_val)
    logging.info(f"  {col}: 欠損 -> {median_val:.4f}で補完")

# ==========================================
# 5. カテゴリカル変数のエンコーディング
# ==========================================
logging.info("\nカテゴリカル変数のエンコーディング中...")

# 訓練データからエンコーディングマッピングを取得
encoding_maps = {}

# track_type
track_type_map = {'芝': 0, 'ダート': 1}
df_2025['track_type_encoded'] = df_2025['track_type'].map(track_type_map)
df_2025['track_type_encoded'] = df_2025['track_type_encoded'].fillna(0)

# track_condition
condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
df_2025['track_condition_encoded'] = df_2025['track_condition'].map(condition_map)
df_2025['track_condition_encoded'] = df_2025['track_condition_encoded'].fillna(0)

# weather
weather_map = {'晴': 0, '曇': 1, '雨': 2, '雪': 3}
df_2025['weather_encoded'] = df_2025['weather'].map(weather_map)
df_2025['weather_encoded'] = df_2025['weather_encoded'].fillna(0)

# sex
sex_map = {'牡': 0, '牝': 1, 'セ': 2}
df_2025['sex_encoded'] = df_2025['sex'].map(sex_map)
df_2025['sex_encoded'] = df_2025['sex_encoded'].fillna(0)

# distance_category (already numeric)
df_2025['distance_category_encoded'] = df_2025['distance_category']

# season (already numeric)
df_2025['season_encoded'] = df_2025['season']

# track_combined - Label Encoding
from sklearn.preprocessing import LabelEncoder
le_track_combined = LabelEncoder()

# 訓練データのtrack_combinedでフィット
le_track_combined.fit(df_train['track_combined'].astype(str))

# 2025年データに適用（未知のカテゴリは0）
df_2025['track_combined_encoded'] = df_2025['track_combined'].astype(str).apply(
    lambda x: le_track_combined.transform([x])[0] if x in le_track_combined.classes_ else 0
)

logging.info(f"  track_type_encoded: {df_2025['track_type_encoded'].unique()}")
logging.info(f"  weather_encoded: {df_2025['weather_encoded'].unique()}")
logging.info(f"  sex_encoded: {df_2025['sex_encoded'].unique()}")

# ==========================================
# 6. 時系列重み
# ==========================================
logging.info("\n時系列重みを計算中...")

current_year = datetime.now().year
focus_start_year = current_year - 4

def calculate_time_weight(year):
    if year >= focus_start_year:
        years_from_focus = year - focus_start_year
        return 1.0 * (1.3 ** years_from_focus)
    else:
        years_from_old = year - (focus_start_year - 6)
        return max(0.3, 0.3 + 0.7 * years_from_old / 6)

df_2025['time_weight'] = df_2025['year'].apply(calculate_time_weight)

logging.info(f"  時系列重み範囲: {df_2025['time_weight'].min():.3f} - {df_2025['time_weight'].max():.3f}")

# ==========================================
# 7. 最終確認とクリーンアップ
# ==========================================
logging.info("\n最終確認中...")

# モデルが必要とする29個の特徴量
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

# 不足している特徴量を確認
missing_features = [f for f in required_features if f not in df_2025.columns]
if missing_features:
    logging.warning(f"⚠️ 不足している特徴量: {missing_features}")
    # 中央値で補完
    for feat in missing_features:
        if feat in df_train.columns:
            median_val = df_train[feat].median()
            df_2025[feat] = median_val
            logging.info(f"  {feat}: {median_val:.4f}で補完")
        else:
            df_2025[feat] = 0
            logging.info(f"  {feat}: 0で補完")

# 欠損値を最終チェック
for feat in required_features:
    if df_2025[feat].isnull().any():
        median_val = df_2025[feat].median()
        if pd.isna(median_val):
            median_val = 0
        df_2025[feat] = df_2025[feat].fillna(median_val)
        logging.info(f"  {feat}: 残存欠損値を{median_val:.4f}で補完")

# ==========================================
# 8. 保存
# ==========================================
print()
print('='*60)
print('前処理完了')
print('='*60)

# 必要な特徴量+メタデータを含めて保存
save_columns = required_features + [
    'race_id', 'race_name', 'horse_name', 'horse_id',
    'jockey_name', 'jockey_id', 'finish_position', 'race_date'
]
save_columns = [c for c in save_columns if c in df_2025.columns]

output_file = 'data/processed/g1_races_2025_processed.csv'
df_2025[save_columns].to_csv(output_file, index=False, encoding='utf-8-sig')

logging.info(f"\n保存先: {output_file}")
logging.info(f"レコード数: {len(df_2025)}")
logging.info(f"カラム数: {len(save_columns)}")
logging.info(f"レース数: {df_2025['race_id'].nunique()}")

print()
print('次のステップ: python test_lightgbm_2025.py')
print()

# 統計情報を表示
print('レース別データ数:')
for race_id in df_2025['race_id'].unique():
    race_data = df_2025[df_2025['race_id'] == race_id]
    race_name = race_data.iloc[0].get('race_name', race_id)
    race_date = race_data.iloc[0]['race_date']
    print(f'  {race_name} ({race_date.strftime("%Y-%m-%d")}): {len(race_data)}頭')

print()
print('特徴量の統計:')
for feat in required_features[:10]:  # 最初の10個だけ表示
    print(f'  {feat}: min={df_2025[feat].min():.2f}, max={df_2025[feat].max():.2f}, mean={df_2025[feat].mean():.2f}')

"""
特徴量エンジニアリングスクリプト
騎手・馬のパフォーマンス、レースコンテキスト、時系列重みを追加
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

print('='*60)
print('特徴量エンジニアリング開始')
print('='*60)
print()

# クリーニング済みデータを読み込み
df = pd.read_csv('data/processed/cleaned_races.csv', encoding='utf-8-sig')
print(f'読み込んだレコード数: {len(df)}')
print(f'レース数: {df["race_id"].nunique()}')
print()

# 日付をdatetime型に変換
df['race_date'] = pd.to_datetime(df['race_date'])

# レースIDでソート（時系列順）
df = df.sort_values('race_date').reset_index(drop=True)

print('特徴量エンジニアリング実行中...')
print()

# ========================================
# 1. 騎手パフォーマンス特徴量
# ========================================
print('1. 騎手パフォーマンス特徴量を作成中...')

# 騎手ごとの勝率・複勝率
jockey_stats = df.groupby('jockey_name').agg({
    'finish_position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
}).reset_index()
jockey_stats.columns = ['jockey_name', 'jockey_race_count', 'jockey_wins', 'jockey_top3']
jockey_stats['jockey_win_rate'] = jockey_stats['jockey_wins'] / jockey_stats['jockey_race_count']
jockey_stats['jockey_top3_rate'] = jockey_stats['jockey_top3'] / jockey_stats['jockey_race_count']

# 馬場タイプ別の騎手勝率
jockey_track_stats = df.groupby(['jockey_name', 'track_type']).agg({
    'finish_position': ['count', lambda x: (x == 1).sum()]
}).reset_index()
jockey_track_stats.columns = ['jockey_name', 'track_type', 'jockey_track_races', 'jockey_track_wins']
jockey_track_stats['jockey_track_win_rate'] = jockey_track_stats['jockey_track_wins'] / jockey_track_stats['jockey_track_races']

# メインデータフレームにマージ
df = df.merge(jockey_stats[['jockey_name', 'jockey_win_rate', 'jockey_top3_rate']],
              on='jockey_name', how='left')
df = df.merge(jockey_track_stats[['jockey_name', 'track_type', 'jockey_track_win_rate']],
              on=['jockey_name', 'track_type'], how='left')

# 欠損値を0で埋める（新人騎手など）
df['jockey_win_rate'] = df['jockey_win_rate'].fillna(0)
df['jockey_top3_rate'] = df['jockey_top3_rate'].fillna(0)
df['jockey_track_win_rate'] = df['jockey_track_win_rate'].fillna(0)

print(f'   騎手勝率範囲: {df["jockey_win_rate"].min():.3f} - {df["jockey_win_rate"].max():.3f}')
print(f'   騎手複勝率範囲: {df["jockey_top3_rate"].min():.3f} - {df["jockey_top3_rate"].max():.3f}')

# ========================================
# 2. 馬のパフォーマンス特徴量
# ========================================
print()
print('2. 馬のパフォーマンス特徴量を作成中...')

# 馬ごとの勝率・複勝率
horse_stats = df.groupby('horse_id').agg({
    'finish_position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
}).reset_index()
horse_stats.columns = ['horse_id', 'horse_race_count', 'horse_wins', 'horse_top3']
horse_stats['horse_win_rate'] = horse_stats['horse_wins'] / horse_stats['horse_race_count']
horse_stats['horse_top3_rate'] = horse_stats['horse_top3'] / horse_stats['horse_race_count']

# 距離カテゴリ別の馬勝率
horse_distance_stats = df.groupby(['horse_id', 'distance_category']).agg({
    'finish_position': ['count', lambda x: (x == 1).sum()]
}).reset_index()
horse_distance_stats.columns = ['horse_id', 'distance_category', 'horse_dist_races', 'horse_dist_wins']
horse_distance_stats['horse_dist_win_rate'] = horse_distance_stats['horse_dist_wins'] / horse_distance_stats['horse_dist_races']

# メインデータフレームにマージ
df = df.merge(horse_stats[['horse_id', 'horse_win_rate', 'horse_top3_rate', 'horse_race_count']],
              on='horse_id', how='left')
df = df.merge(horse_distance_stats[['horse_id', 'distance_category', 'horse_dist_win_rate']],
              on=['horse_id', 'distance_category'], how='left')

df['horse_win_rate'] = df['horse_win_rate'].fillna(0)
df['horse_top3_rate'] = df['horse_top3_rate'].fillna(0)
df['horse_dist_win_rate'] = df['horse_dist_win_rate'].fillna(0)
df['horse_race_count'] = df['horse_race_count'].fillna(0)

print(f'   馬勝率範囲: {df["horse_win_rate"].min():.3f} - {df["horse_win_rate"].max():.3f}')
print(f'   馬複勝率範囲: {df["horse_top3_rate"].min():.3f} - {df["horse_top3_rate"].max():.3f}')
print(f'   馬出走回数範囲: {df["horse_race_count"].min():.0f} - {df["horse_race_count"].max():.0f}')

# ========================================
# 3. レースコンテキスト特徴量
# ========================================
print()
print('3. レースコンテキスト特徴量を作成中...')

# レースごとの出走頭数
race_field_size = df.groupby('race_id').size().reset_index(name='field_size')
df = df.merge(race_field_size, on='race_id', how='left')

# レースごとの平均オッズ（人気度）
race_avg_odds = df.groupby('race_id')['odds'].mean().reset_index(name='race_avg_odds')
df = df.merge(race_avg_odds, on='race_id', how='left')

# 人気順位（オッズが低い順）
df['popularity_rank'] = df.groupby('race_id')['odds'].rank(method='min')

print(f'   出走頭数範囲: {df["field_size"].min():.0f} - {df["field_size"].max():.0f}')
print(f'   人気順位範囲: {df["popularity_rank"].min():.0f} - {df["popularity_rank"].max():.0f}')

# ========================================
# 4. 時間ベース特徴量
# ========================================
print()
print('4. 時間ベース特徴量を作成中...')

# 年、月、曜日
df['year'] = df['race_date'].dt.year
df['month'] = df['race_date'].dt.month
df['day_of_week'] = df['race_date'].dt.dayofweek

# 季節（春夏秋冬）
df['season'] = df['month'].apply(lambda x:
    'spring' if x in [3,4,5] else
    'summer' if x in [6,7,8] else
    'fall' if x in [9,10,11] else
    'winter'
)

print(f'   年範囲: {df["year"].min()} - {df["year"].max()}')
print(f'   季節分布:')
for season, count in df['season'].value_counts().items():
    print(f'      {season}: {count}件')

# ========================================
# 5. カテゴリカル変数のエンコーディング
# ========================================
print()
print('5. カテゴリカル変数のエンコーディング中...')

categorical_cols = ['track_type', 'track_condition', 'weather', 'sex',
                   'distance_category', 'season', 'track_combined']

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f'   {col}: {len(le.classes_)}クラス')

# ========================================
# 6. 時系列重み計算（直近4年を重視）
# ========================================
print()
print('6. 時系列重みを計算中...')

current_year = datetime.now().year
focus_start_year = current_year - 4  # 2021年

def calculate_time_weight(year):
    if year >= focus_start_year:
        # 直近4年: 指数的に重み付け (1.0 → 3.0)
        years_from_focus = year - focus_start_year
        return 1.0 * (1.3 ** years_from_focus)
    else:
        # 過去6年: 線形に重み付け (0.3 → 1.0)
        years_from_old = year - (focus_start_year - 6)
        return max(0.3, 0.3 + 0.7 * years_from_old / 6)

df['time_weight'] = df['year'].apply(calculate_time_weight)

print(f'   時系列重み範囲: {df["time_weight"].min():.3f} - {df["time_weight"].max():.3f}')
print(f'   年別平均重み:')
for year in sorted(df['year'].unique()):
    avg_weight = df[df['year'] == year]['time_weight'].mean()
    count = len(df[df['year'] == year])
    print(f'      {year}年: {avg_weight:.3f} ({count}件)')

# ========================================
# 保存
# ========================================
print()
print('='*60)
print('特徴量エンジニアリング完了')
print('='*60)

output_file = 'data/processed/features_engineered.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f'保存先: {output_file}')
print(f'最終レコード数: {len(df)}')
print(f'最終カラム数: {len(df.columns)}')
print()
print('追加された特徴量:')
print('  - 騎手勝率・複勝率 (jockey_win_rate, jockey_top3_rate)')
print('  - 騎手馬場別勝率 (jockey_track_win_rate)')
print('  - 馬勝率・複勝率 (horse_win_rate, horse_top3_rate)')
print('  - 馬距離別勝率 (horse_dist_win_rate)')
print('  - 出走頭数 (field_size)')
print('  - 人気順位 (popularity_rank)')
print('  - 時間特徴量 (year, month, season)')
print('  - カテゴリエンコーディング (*_encoded)')
print('  - 時系列重み (time_weight)')
print()
print('次のステップ: モデル訓練')

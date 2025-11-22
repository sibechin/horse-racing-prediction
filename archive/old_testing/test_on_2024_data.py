"""
2024年データでモデルをテスト
真の予測精度を検証（訓練: 2016-2023, テスト: 2024）
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys

sys.path.append('.')
from src.data_collection.smart_race_discoverer import SmartRaceDiscoverer
from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete

print("="*60)
print("2024年データでモデルテスト（未知データ）")
print("="*60)
print()

# ステップ1: 2025年のレースID発見
print("ステップ1: 2024年のレースID発見")
print("-"*60)

discoverer = SmartRaceDiscoverer(delay=0.3)

# 2024年1月〜12月の土日レースを発見（テスト用）
print("2024年1-12月の土日レースを探索中...")
test_race_ids = discoverer.discover_races_smart_sampling(
    year=2024,
    start_month=1,
    end_month=12,
    max_races=50  # 50レースでテスト
)

print(f"発見したレース数: {len(test_race_ids)}")
print()

if len(test_race_ids) == 0:
    print("レースが見つかりませんでした")
    sys.exit(1)

# ステップ2: データ収集
print("ステップ2: 2024年データ収集")
print("-"*60)

scraper = NetkeibaScraperComplete(delay=1.5, max_retries=3)

print(f"収集開始: {len(test_race_ids)}レース")
print(f"推定時間: {len(test_race_ids) * 2 / 60:.1f}分")
print()

test_df = scraper.scrape_multiple_races(
    test_race_ids,
    output_dir='data/test'
)

if len(test_df) == 0:
    print("データ収集失敗")
    sys.exit(1)

print(f"収集成功: {test_df['race_id'].nunique()}レース, {len(test_df)}頭")
print()

# ステップ3: 訓練データの統計情報を読み込み
print("ステップ3: 特徴量エンジニアリング")
print("-"*60)

# 訓練データを読み込んで統計情報を取得
train_df = pd.read_csv("data/processed/features_engineered.csv", encoding="utf-8-sig")

# 騎手統計
jockey_stats = train_df.groupby('jockey_name').agg({
    'finish_position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
}).reset_index()
jockey_stats.columns = ['jockey_name', 'jockey_race_count', 'jockey_wins', 'jockey_top3']
jockey_stats['jockey_win_rate'] = jockey_stats['jockey_wins'] / jockey_stats['jockey_race_count']
jockey_stats['jockey_top3_rate'] = jockey_stats['jockey_top3'] / jockey_stats['jockey_race_count']

# 騎手馬場別統計
jockey_track_stats = train_df.groupby(['jockey_name', 'track_type']).agg({
    'finish_position': ['count', lambda x: (x == 1).sum()]
}).reset_index()
jockey_track_stats.columns = ['jockey_name', 'track_type', 'jockey_track_races', 'jockey_track_wins']
jockey_track_stats['jockey_track_win_rate'] = jockey_track_stats['jockey_track_wins'] / jockey_track_stats['jockey_track_races']

# 馬統計
horse_stats = train_df.groupby('horse_id').agg({
    'finish_position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
}).reset_index()
horse_stats.columns = ['horse_id', 'horse_race_count', 'horse_wins', 'horse_top3']
horse_stats['horse_win_rate'] = horse_stats['horse_wins'] / horse_stats['horse_race_count']
horse_stats['horse_top3_rate'] = horse_stats['horse_top3'] / horse_stats['horse_race_count']

print("訓練データ統計取得完了")

# テストデータに特徴量を追加
test_df['race_date'] = pd.to_datetime(test_df['race_date'])
test_df = test_df.sort_values('race_date').reset_index(drop=True)

# 騎手特徴量をマージ
test_df = test_df.merge(
    jockey_stats[['jockey_name', 'jockey_win_rate', 'jockey_top3_rate']],
    on='jockey_name', how='left'
)
test_df = test_df.merge(
    jockey_track_stats[['jockey_name', 'track_type', 'jockey_track_win_rate']],
    on=['jockey_name', 'track_type'], how='left'
)

# 馬特徴量をマージ
test_df = test_df.merge(
    horse_stats[['horse_id', 'horse_win_rate', 'horse_top3_rate', 'horse_race_count']],
    on='horse_id', how='left'
)

# 欠損値を0で埋める（新人騎手・新馬など）
test_df['jockey_win_rate'] = test_df['jockey_win_rate'].fillna(0)
test_df['jockey_top3_rate'] = test_df['jockey_top3_rate'].fillna(0)
test_df['jockey_track_win_rate'] = test_df['jockey_track_win_rate'].fillna(0)
test_df['horse_win_rate'] = test_df['horse_win_rate'].fillna(0)
test_df['horse_top3_rate'] = test_df['horse_top3_rate'].fillna(0)
test_df['horse_race_count'] = test_df['horse_race_count'].fillna(0)

# 距離カテゴリがない場合は追加
if 'distance_category' not in test_df.columns:
    def categorize_distance(distance):
        if distance < 1400:
            return 'short'
        elif distance < 1800:
            return 'mile'
        elif distance < 2200:
            return 'middle'
        else:
            return 'long'
    test_df['distance_category'] = test_df['distance'].apply(categorize_distance)

# レース特徴量
race_field_size = test_df.groupby('race_id').size().reset_index(name='field_size')
test_df = test_df.merge(race_field_size, on='race_id', how='left')

race_avg_odds = test_df.groupby('race_id')['odds'].mean().reset_index(name='race_avg_odds')
test_df = test_df.merge(race_avg_odds, on='race_id', how='left')

test_df['popularity_rank'] = test_df.groupby('race_id')['odds'].rank(method='min')

# 時間特徴量
test_df['year'] = test_df['race_date'].dt.year
test_df['month'] = test_df['race_date'].dt.month
test_df['day_of_week'] = test_df['race_date'].dt.dayofweek

# カテゴリエンコーディング（訓練データのエンコーダを使用）
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['track_type', 'track_condition', 'weather', 'sex', 'distance_category']

for col in categorical_cols:
    if col in test_df.columns:
        le = LabelEncoder()
        # 訓練データの全カテゴリを学習
        train_categories = train_df[col].unique()
        le.fit(train_categories)

        # テストデータをエンコード（未知カテゴリは0）
        test_df[f'{col}_encoded'] = test_df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

# track_combinedエンコーディング
if 'track_combined' in train_df.columns:
    train_combined = train_df['track_combined'].unique()
    le_combined = LabelEncoder()
    le_combined.fit(train_combined)
    test_df['track_combined_encoded'] = test_df.apply(
        lambda row: le_combined.transform([f"{row['track_type']}_{row['track_condition']}"])[0]
        if f"{row['track_type']}_{row['track_condition']}" in le_combined.classes_ else 0,
        axis=1
    )

print(f"特徴量追加完了: {len(test_df)}レコード")
print()

# ステップ4: 予測
print("ステップ4: モデルで予測")
print("-"*60)

# 特徴量準備
feature_cols = [
    "distance", "weight", "age",
    "track_type_encoded", "track_condition_encoded", "weather_encoded",
    "sex_encoded", "distance_category_encoded", "track_combined_encoded",
    "jockey_win_rate", "jockey_top3_rate", "jockey_track_win_rate",
    "horse_win_rate", "horse_top3_rate", "horse_race_count",
    "field_size", "race_avg_odds", "popularity_rank", "odds",
    "year", "month", "day_of_week",
    "time_seconds"
]

# horse_dist_win_rateは訓練時に使ったが、テストデータで計算が難しいので除外
available_features = [col for col in feature_cols if col in test_df.columns]

# 欠損値処理
test_df[available_features] = test_df[available_features].fillna(0)

X_test = test_df[available_features].values

# アンサンブル予測
model_dir = Path("models")
fold_models = []

for i in range(1, 6):
    model_path = model_dir / f"lightgbm_lambdarank_refined_fold{i}.txt"
    if model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        fold_models.append(model)

print(f"読み込んだモデル数: {len(fold_models)}")

# アンサンブル予測
ensemble_predictions = np.zeros(len(X_test))
for model in fold_models:
    pred = model.predict(X_test)
    ensemble_predictions += pred
ensemble_predictions /= len(fold_models)

test_df['ensemble_prediction'] = ensemble_predictions

# レースごとに予測ランキングを作成
test_df['predicted_rank'] = test_df.groupby('race_id')['ensemble_prediction'].rank(
    ascending=True, method='min'
)

print("予測完了")
print()

# ステップ5: 評価
print("="*60)
print("予測精度評価")
print("="*60)
print()

# Top1的中率（1着予測）
def calculate_top1_accuracy(df):
    top1_predicted = df[df['predicted_rank'] == 1]
    if len(top1_predicted) == 0:
        return 0
    correct = (top1_predicted['finish_position'] == 1).sum()
    total = len(df['race_id'].unique())
    return correct / total

# Top3的中率（3着以内予測）
def calculate_top3_accuracy(df):
    top3_predicted = df[df['predicted_rank'] <= 3]
    if len(top3_predicted) == 0:
        return 0
    correct = (top3_predicted['finish_position'] <= 3).sum()
    total = len(top3_predicted)
    return correct / total

top1_acc = calculate_top1_accuracy(test_df)
top3_acc = calculate_top3_accuracy(test_df)

print(f"テストデータ:")
print(f"  レース数: {test_df['race_id'].nunique()}")
print(f"  総データ数: {len(test_df)}頭")
print(f"  期間: {test_df['race_date'].min()} - {test_df['race_date'].max()}")
print()

print("予測精度（2024年未知データ - 真の汎化性能）:")
print(f"  Top1的中率（1着予測）: {top1_acc:.2%}")
print(f"  Top3的中率（3着以内）: {top3_acc:.2%}")
print()

print("参考（訓練データ 2016-2023）:")
print("  Top1的中率: 91.18%")
print("  Top3的中率: 83.44%")
print("  ※訓練精度が高すぎる場合、過学習の可能性")
print()

# オッズとの比較
test_df['odds_rank'] = test_df.groupby('race_id')['odds'].rank(method='min')

def calculate_odds_top1_accuracy(df):
    top1_odds = df[df['odds_rank'] == 1]
    if len(top1_odds) == 0:
        return 0
    correct = (top1_odds['finish_position'] == 1).sum()
    total = len(df['race_id'].unique())
    return correct / total

odds_top1_acc = calculate_odds_top1_accuracy(test_df)
print(f"比較: オッズ1番人気の的中率: {odds_top1_acc:.2%}")
print()

# サンプル表示
print("="*60)
print("サンプル予測結果（最初の3レース）")
print("="*60)

for race_id in test_df['race_id'].unique()[:3]:
    race_data = test_df[test_df['race_id'] == race_id].copy()
    race_data = race_data.sort_values('predicted_rank')

    print(f"\nレースID: {race_id}")
    print(f"日付: {race_data.iloc[0]['race_date']}")
    print()
    print(f"{'予測順位':<8} {'実着順':<8} {'予測スコア':<12} {'オッズ':<8} {'人気':<6}")
    print("-" * 50)

    for _, row in race_data.head(5).iterrows():
        correct_mark = "✓" if int(row['predicted_rank']) == int(row['finish_position']) else " "
        print(f"{int(row['predicted_rank']):<8} "
              f"{int(row['finish_position']):<8} "
              f"{row['ensemble_prediction']:<12.4f} "
              f"{row['odds']:<8.1f} "
              f"{int(row['popularity_rank']):<6} {correct_mark}")

print()
print("="*60)
print("テスト完了")
print("="*60)

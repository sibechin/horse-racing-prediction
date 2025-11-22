"""
JRA公式データでモデルテスト
2025年フェブラリーステークス (G1)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re

print("="*60)
print("JRA公式データでモデルテスト")
print("="*60)
print()

# ステップ1: JRAから2025年フェブラリーステークスのデータを取得
print("ステップ1: JRAデータ取得")
print("-"*60)

url = "https://www.jra.go.jp/datafile/seiseki/g1/feb/result/feb2025.html"
print(f"URL: {url}")

response = requests.get(url)
response.encoding = response.apparent_encoding
soup = BeautifulSoup(response.text, 'html.parser')

# テーブルからデータを抽出
table = soup.find('table')
rows = table.find_all('tr')[1:]  # ヘッダー行をスキップ

race_data = []
for row in rows:
    cols = row.find_all(['td', 'th'])
    if len(cols) < 11:
        continue

    try:
        # データ抽出（空の値に対応）
        finish_text = cols[0].text.strip()
        if not finish_text or not finish_text.isdigit():
            continue

        finish_position = int(finish_text)
        frame_number = int(cols[1].text.strip()) if cols[1].text.strip() else 0
        horse_number = int(cols[2].text.strip()) if cols[2].text.strip() else 0
        horse_name = cols[3].text.strip()
        age_sex = cols[4].text.strip()
        weight_kg = float(cols[5].text.strip()) if cols[5].text.strip() else 55.0
        jockey_name = cols[6].text.strip()
        time_str = cols[7].text.strip()
        margin = cols[8].text.strip()
        corner_positions = cols[9].text.strip()
        odds_str = cols[10].text.strip()
        # タイムを秒に変換
        time_match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
        if time_match:
            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))
            milliseconds = int(time_match.group(3))
            time_seconds = minutes * 60 + seconds + milliseconds / 10
        else:
            time_seconds = 0

        # オッズをfloatに変換
        try:
            odds = float(odds_str)
        except:
            odds = 0.0

        # 年齢・性別を分解
        age_match = re.match(r'(\d+)', age_sex)
        age = int(age_match.group(1)) if age_match else 4
        sex = 'M' if '牡' in age_sex else ('F' if '牝' in age_sex else 'G')

        race_data.append({
            'race_id': '202502_feb',
            'race_date': '2025-02-16',
            'venue_name': '東京',
            'track_type': 'ダート',
            'track_condition': '良',
            'weather': '晴',
            'distance': 1600,
            'finish_position': finish_position,
            'frame_number': frame_number,
            'horse_number': horse_number,
            'horse_name': horse_name,
            'horse_id': f'jra_{horse_name}',
            'age': age,
            'sex': sex,
            'weight': weight_kg,
            'jockey_name': jockey_name,
            'time_seconds': time_seconds,
            'odds': odds
        })
    except (ValueError, IndexError) as e:
        print(f"行のスキップ: {e}")
        continue

test_df = pd.DataFrame(race_data)
print(f"データ取得成功: {len(test_df)}頭")
print(f"レース: 2025年フェブラリーステークス (G1)")
print(f"距離: 1600m (ダート)")
print()

# ステップ2: 訓練データの統計情報を読み込み
print("ステップ2: 特徴量エンジニアリング")
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

# 特徴量を追加
test_df['race_date'] = pd.to_datetime(test_df['race_date'])

# 騎手特徴量をマージ
test_df = test_df.merge(
    jockey_stats[['jockey_name', 'jockey_win_rate', 'jockey_top3_rate']],
    on='jockey_name', how='left'
)
test_df = test_df.merge(
    jockey_track_stats[['jockey_name', 'track_type', 'jockey_track_win_rate']],
    on=['jockey_name', 'track_type'], how='left'
)

# 馬の統計情報は新馬なので0で埋める
test_df['horse_win_rate'] = 0.0
test_df['horse_top3_rate'] = 0.0
test_df['horse_dist_win_rate'] = 0.0
test_df['horse_race_count'] = 0

# 欠損値を0で埋める
test_df['jockey_win_rate'] = test_df['jockey_win_rate'].fillna(0)
test_df['jockey_top3_rate'] = test_df['jockey_top3_rate'].fillna(0)
test_df['jockey_track_win_rate'] = test_df['jockey_track_win_rate'].fillna(0)

# 距離カテゴリ
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
test_df['field_size'] = len(test_df)
test_df['race_avg_odds'] = test_df['odds'].mean()
test_df['popularity_rank'] = test_df['odds'].rank(method='min')

# 時間特徴量
test_df['year'] = test_df['race_date'].dt.year
test_df['month'] = test_df['race_date'].dt.month
test_df['day_of_week'] = test_df['race_date'].dt.dayofweek

# カテゴリエンコーディング
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['track_type', 'track_condition', 'weather', 'sex', 'distance_category']

for col in categorical_cols:
    le = LabelEncoder()
    train_categories = train_df[col].unique()
    le.fit(train_categories)

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

# ステップ3: 予測
print("ステップ3: モデルで予測")
print("-"*60)

# 特徴量準備
feature_cols = [
    "distance", "weight", "age",
    "track_type_encoded", "track_condition_encoded", "weather_encoded",
    "sex_encoded", "distance_category_encoded", "track_combined_encoded",
    "jockey_win_rate", "jockey_top3_rate", "jockey_track_win_rate",
    "horse_win_rate", "horse_top3_rate", "horse_dist_win_rate", "horse_race_count",
    "field_size", "race_avg_odds", "popularity_rank", "odds",
    "year", "month", "day_of_week",
    "time_seconds"
]

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

# 予測ランキングを作成
test_df['predicted_rank'] = test_df['ensemble_prediction'].rank(ascending=True, method='min')

print("予測完了")
print()

# ステップ4: 結果表示
print("="*60)
print("予測結果")
print("="*60)
print()

print("2025年フェブラリーステークス (G1)")
print("東京 1600m (ダート・左)")
print()

# 結果を予測順位でソート
result_df = test_df.sort_values('predicted_rank')

print(f"{'予測順位':<8} {'実着順':<8} {'馬名':<20} {'騎手':<15} {'オッズ':<8} {'人気':<6} {'判定':<4}")
print("-" * 80)

correct_count = 0
for _, row in result_df.iterrows():
    correct_mark = "OK" if int(row['predicted_rank']) == int(row['finish_position']) else "  "
    if int(row['predicted_rank']) == int(row['finish_position']):
        correct_count += 1

    print(f"{int(row['predicted_rank']):<8} "
          f"{int(row['finish_position']):<8} "
          f"{row['horse_name']:<20} "
          f"{row['jockey_name']:<15} "
          f"{row['odds']:<8.1f} "
          f"{int(row['popularity_rank']):<6} "
          f"{correct_mark:<4}")

print()
print("="*60)
print("評価")
print("="*60)
print()

# 1着予測
top1_predicted = result_df.iloc[0]
top1_actual = test_df[test_df['finish_position'] == 1].iloc[0]

print(f"1着予測: {top1_predicted['horse_name']} (実際: {top1_actual['horse_name']})")
if top1_predicted['horse_name'] == top1_actual['horse_name']:
    print("  → 的中！")
else:
    print(f"  → 外れ (予測馬の実着順: {int(top1_predicted['finish_position'])}着)")

print()

# 3着以内予測
top3_predicted = set(result_df.head(3)['horse_name'])
top3_actual = set(test_df[test_df['finish_position'] <= 3]['horse_name'])
top3_matches = len(top3_predicted & top3_actual)

print(f"3着以内予測的中数: {top3_matches}/3")
print(f"予測: {', '.join(list(top3_predicted)[:3])}")
print(f"実際: {', '.join(list(top3_actual)[:3])}")

print()

# オッズとの比較
odds_favorite = test_df.loc[test_df['odds'].idxmin()]
print(f"オッズ1番人気: {odds_favorite['horse_name']} (実着順: {int(odds_favorite['finish_position'])}着)")

print()
print("="*60)
print("完了")
print("="*60)

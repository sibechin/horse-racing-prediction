"""
データクリーニングスクリプト

1. 障害レース除外（名前のみで判定）
2. 異常値除去
3. G1レースの抽出
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("="*60)
print("データクリーニング")
print("="*60)
print()

# ============================================================
# ステップ1: データ読み込み
# ============================================================
print("ステップ1: データ読み込み")
print("-"*60)

df = pd.read_csv("data/processed/merged_data_all.csv", encoding='utf-8-sig')
logging.info(f"読み込み: {len(df)}レコード, {df.get('race_id', df.index).nunique()}レース")
print()

# ============================================================
# ステップ2: 障害レース除外
# ============================================================
print("ステップ2: 障害レース除外")
print("-"*60)

# race_nameが存在するか確認
if 'race_name' in df.columns:
    # 障害レースのキーワード
    obstacle_keywords = ['障害', 'ステープルチェース', 'ハードル']

    # 除外対象をフィルタ
    obstacle_mask = df['race_name'].str.contains('|'.join(obstacle_keywords), na=False, case=False)
    obstacle_count = obstacle_mask.sum()

    if obstacle_count > 0:
        logging.info(f"障害レース検出: {obstacle_count}レコード")
        print(f"除外する障害レース:")
        obstacle_races = df[obstacle_mask]['race_name'].value_counts()
        for race, count in obstacle_races.items():
            print(f"  - {race}: {count}レコード")

        # 除外
        df = df[~obstacle_mask].copy()
        logging.info(f"障害レース除外後: {len(df)}レコード")
    else:
        logging.info("障害レースなし")
else:
    logging.warning("race_nameカラムなし - 障害レース除外スキップ")

print()

# ============================================================
# ステップ3: 異常値除去
# ============================================================
print("ステップ3: 異常値除去")
print("-"*60)

initial_count = len(df)

# 距離の異常値
if 'distance' in df.columns:
    invalid_distance = df['distance'] == 0
    invalid_distance_count = invalid_distance.sum()
    if invalid_distance_count > 0:
        logging.warning(f"距離=0: {invalid_distance_count}件を除外")
        df = df[~invalid_distance].copy()

    # 距離の妥当な範囲（800m〜4000m）
    invalid_range = (df['distance'] < 800) | (df['distance'] > 4000)
    invalid_range_count = invalid_range.sum()
    if invalid_range_count > 0:
        logging.warning(f"距離範囲外: {invalid_range_count}件を除外")
        df = df[~invalid_range].copy()

# 着順の異常値
if 'finish_position' in df.columns:
    invalid_finish = (df['finish_position'] < 1) | (df['finish_position'] > 20)
    invalid_finish_count = invalid_finish.sum()
    if invalid_finish_count > 0:
        logging.warning(f"異常な着順: {invalid_finish_count}件を除外")
        df = df[~invalid_finish].copy()

# オッズの異常値
if 'odds' in df.columns:
    invalid_odds = (df['odds'] < 0) | (df['odds'] > 1000)
    invalid_odds_count = invalid_odds.sum()
    if invalid_odds_count > 0:
        logging.warning(f"異常なオッズ: {invalid_odds_count}件を除外")
        df = df[~invalid_odds].copy()

removed_count = initial_count - len(df)
logging.info(f"異常値除去: {removed_count}件")
logging.info(f"クリーニング後: {len(df)}レコード")
print()

# ============================================================
# ステップ4: G1レースの抽出
# ============================================================
print("ステップ4: G1レースの抽出")
print("-"*60)

# gradeカラムが存在するか確認
if 'grade' in df.columns:
    df_g1 = df[df['grade'] == 'G1'].copy()
    logging.info(f"G1レース: {len(df_g1)}レコード, {df_g1.get('race_id', df_g1.index).nunique()}レース")
else:
    # gradeカラムがない場合、race_nameから推定
    logging.warning("gradeカラムなし - race_nameからG1を推定")

    # G1レース名のキーワード
    g1_keywords = [
        'フェブラリー', '宝塚', '天皇賞', '有馬記念', 'ジャパンカップ',
        'マイルチャンピオンシップ', '安田記念', 'スプリンターズ',
        '秋華賞', '菊花賞', '皐月賞', 'オークス', 'ダービー',
        'ホープフル', 'ジュベナイルフィリーズ', 'フューチュリティ'
    ]

    if 'race_name' in df.columns:
        g1_mask = df['race_name'].str.contains('|'.join(g1_keywords), na=False, case=False)
        df_g1 = df[g1_mask].copy()
        logging.info(f"G1レース（推定）: {len(df_g1)}レコード")
    else:
        logging.error("race_nameカラムもなし - G1抽出不可")
        df_g1 = df.copy()

print()

# ============================================================
# ステップ5: 統計情報
# ============================================================
print("="*60)
print("クリーニング結果")
print("="*60)
print()

print(f"全データ:")
print(f"  総レコード数: {len(df):,}")
print(f"  総レース数: {df.get('race_id', df.index).nunique():,}")
if 'race_date' in df.columns:
    df['race_date'] = pd.to_datetime(df['race_date'])
    print(f"  期間: {df['race_date'].min()} - {df['race_date'].max()}")
print()

print(f"G1データ:")
print(f"  総レコード数: {len(df_g1):,}")
print(f"  総レース数: {df_g1.get('race_id', df_g1.index).nunique():,}")
if 'race_date' in df_g1.columns:
    df_g1['race_date'] = pd.to_datetime(df_g1['race_date'])
    print(f"  期間: {df_g1['race_date'].min()} - {df_g1['race_date'].max()}")
print()

if 'track_type' in df.columns:
    print("馬場タイプ別（全データ）:")
    print(df['track_type'].value_counts())
    print()

    print("馬場タイプ別（G1のみ）:")
    print(df_g1['track_type'].value_counts())
    print()

if 'venue_name' in df.columns:
    print("競馬場別（G1のみ、上位10）:")
    venue_counts = df_g1['venue_name'].value_counts().head(10)
    for venue, count in venue_counts.items():
        print(f"  {venue}: {count}レコード")
    print()

# ============================================================
# ステップ6: 保存
# ============================================================
print("="*60)
print("保存")
print("="*60)
print()

# クリーニング済み全データ
output_all = "data/processed/cleaned_data_all.csv"
Path(output_all).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_all, index=False, encoding='utf-8-sig')
logging.info(f"全データ保存: {output_all}")

# G1データ
output_g1 = "data/processed/cleaned_data_g1_only.csv"
df_g1.to_csv(output_g1, index=False, encoding='utf-8-sig')
logging.info(f"G1データ保存: {output_g1}")

print()
print("="*60)
print("完了!")
print("="*60)
print()

print("次のステップ:")
print("  1. python feature_engineering.py で特徴量エンジニアリングを実行")
print("  2. G1専用モデルを訓練")
print("  3. JRA G2/G3データを追加収集")
print("  4. 混合モデル（G1+G2+G3）を訓練")
print("  5. 両モデルを比較評価")

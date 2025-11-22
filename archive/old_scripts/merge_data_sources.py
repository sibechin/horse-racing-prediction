"""
データソース統合スクリプト

netkeiba.comデータ + JRAデータを統合
重複除去・品質チェックを実施
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
print("データソース統合")
print("="*60)
print()

# ============================================================
# ステップ1: データ読み込み
# ============================================================
print("ステップ1: データ読み込み")
print("-"*60)

# netkeiba.comデータ
netkeiba_path = "data/processed/features_engineered.csv"
if Path(netkeiba_path).exists():
    df_netkeiba = pd.read_csv(netkeiba_path, encoding='utf-8-sig')
    logging.info(f"netkeiba.com: {len(df_netkeiba)}レコード, {df_netkeiba['race_id'].nunique()}レース")
else:
    df_netkeiba = pd.DataFrame()
    logging.warning("netkeiba.comデータが見つかりません")

# JRAデータ
jra_path = "data/jra/g1_races_2010_2024.csv"
if Path(jra_path).exists():
    df_jra = pd.read_csv(jra_path, encoding='utf-8-sig')
    logging.info(f"JRA: {len(df_jra)}レコード, {df_jra['race_id'].nunique()}レース")
else:
    df_jra = pd.DataFrame()
    logging.warning("JRAデータが見つかりません")

print()

# ============================================================
# ステップ2: データ型の統一
# ============================================================
print("ステップ2: データ型の統一")
print("-"*60)

# JRAデータをnetkeibaフォーマットに合わせる
if len(df_jra) > 0:
    # 必要なカラムを追加（netkeibaにあってJRAにないもの）
    jra_columns_to_add = {
        'bracket_number': 0,  # JRAにはない情報
        'post_time': '',
        'venue_code': 0,
        'jockey_id': 'jra_unknown',
        'time': '',
        'margin': '',
        'passing_order': '',
        'last_3f': 0.0,
        'popularity': 0,
    }

    for col, default_value in jra_columns_to_add.items():
        if col not in df_jra.columns:
            df_jra[col] = default_value

    # race_dateを統一
    df_jra['race_date'] = pd.to_datetime(df_jra['race_date'])

    # 既存のnetkeibaカラムと合わせる
    common_columns = list(set(df_netkeiba.columns) & set(df_jra.columns))
    logging.info(f"共通カラム数: {len(common_columns)}")

    # JRAデータを共通カラムのみに制限
    df_jra_aligned = df_jra[common_columns].copy()
else:
    df_jra_aligned = pd.DataFrame()

print()

# ============================================================
# ステップ3: 重複チェックと除去
# ============================================================
print("ステップ3: 重複チェックと除去")
print("-"*60)

# 重複判定: race_date, venue_name, distance, track_typeの組み合わせ
def create_race_key(row):
    """レース識別キーを作成"""
    try:
        return f"{row['race_date']}_{row['venue_name']}_{row['distance']}_{row['track_type']}"
    except:
        return None

if len(df_netkeiba) > 0:
    df_netkeiba['race_key'] = df_netkeiba.apply(create_race_key, axis=1)
    netkeiba_race_keys = set(df_netkeiba['race_key'].dropna())
    logging.info(f"netkeibaユニークレース数: {len(netkeiba_race_keys)}")
else:
    netkeiba_race_keys = set()

if len(df_jra_aligned) > 0:
    df_jra_aligned['race_key'] = df_jra_aligned.apply(create_race_key, axis=1)
    jra_race_keys = set(df_jra_aligned['race_key'].dropna())
    logging.info(f"JRAユニークレース数: {len(jra_race_keys)}")

    # 重複レースを検出
    duplicate_keys = netkeiba_race_keys & jra_race_keys
    logging.info(f"重複レース数: {len(duplicate_keys)}")

    # JRAから重複を除去（netkeibaを優先）
    df_jra_unique = df_jra_aligned[~df_jra_aligned['race_key'].isin(duplicate_keys)].copy()
    logging.info(f"JRA重複除去後: {len(df_jra_unique)}レコード, {df_jra_unique['race_key'].nunique()}レース")
else:
    df_jra_unique = pd.DataFrame()

print()

# ============================================================
# ステップ4: データ統合
# ============================================================
print("ステップ4: データ統合")
print("-"*60)

# データフレームを結合
dfs_to_merge = []
if len(df_netkeiba) > 0:
    dfs_to_merge.append(df_netkeiba)
if len(df_jra_unique) > 0:
    dfs_to_merge.append(df_jra_unique)

if dfs_to_merge:
    df_merged = pd.concat(dfs_to_merge, ignore_index=True)

    # race_keyカラムを削除
    if 'race_key' in df_merged.columns:
        df_merged.drop('race_key', axis=1, inplace=True)

    # 日付でソート
    df_merged['race_date'] = pd.to_datetime(df_merged['race_date'])
    df_merged = df_merged.sort_values('race_date').reset_index(drop=True)

    logging.info(f"統合完了: {len(df_merged)}レコード, {df_merged['race_id'].nunique()}レース")
    logging.info(f"期間: {df_merged['race_date'].min()} - {df_merged['race_date'].max()}")
else:
    df_merged = pd.DataFrame()
    logging.error("統合するデータがありません")

print()

# ============================================================
# ステップ5: 品質チェック
# ============================================================
print("ステップ5: 品質チェック")
print("-"*60)

if len(df_merged) > 0:
    # 欠損値チェック
    essential_columns = ['race_date', 'venue_name', 'distance', 'track_type', 'finish_position']
    missing_counts = df_merged[essential_columns].isnull().sum()

    print("主要カラムの欠損値:")
    for col in essential_columns:
        count = missing_counts[col]
        if count > 0:
            print(f"  {col}: {count}件")

    # 異常値チェック
    print()
    print("異常値チェック:")

    # 距離の範囲
    invalid_distance = df_merged[(df_merged['distance'] < 800) | (df_merged['distance'] > 4000)]
    if len(invalid_distance) > 0:
        logging.warning(f"異常な距離: {len(invalid_distance)}件")

    # 着順の範囲
    invalid_finish = df_merged[(df_merged['finish_position'] < 1) | (df_merged['finish_position'] > 20)]
    if len(invalid_finish) > 0:
        logging.warning(f"異常な着順: {len(invalid_finish)}件")

    # オッズの範囲
    if 'odds' in df_merged.columns:
        invalid_odds = df_merged[(df_merged['odds'] < 0) | (df_merged['odds'] > 1000)]
        if len(invalid_odds) > 0:
            logging.warning(f"異常なオッズ: {len(invalid_odds)}件")

print()

# ============================================================
# ステップ6: 統計情報
# ============================================================
print("="*60)
print("統合データ統計")
print("="*60)
print()

if len(df_merged) > 0:
    print(f"総レコード数: {len(df_merged):,}")
    print(f"総レース数: {df_merged['race_id'].nunique():,}")
    print(f"期間: {df_merged['race_date'].min()} - {df_merged['race_date'].max()}")
    print()

    print("年別レース数:")
    if 'year' not in df_merged.columns and 'race_date' in df_merged.columns:
        df_merged['year'] = pd.to_datetime(df_merged['race_date']).dt.year

    year_counts = df_merged.groupby('year')['race_id'].nunique().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count}レース")
    print()

    print("競馬場別レース数:")
    venue_counts = df_merged.groupby('venue_name')['race_id'].nunique().sort_values(ascending=False)
    for venue, count in venue_counts.head(10).items():
        print(f"  {venue}: {count}レース")
    print()

    print("馬場タイプ別:")
    track_counts = df_merged['track_type'].value_counts()
    for track, count in track_counts.items():
        print(f"  {track}: {count}レコード")
    print()

    print("距離分布:")
    print(df_merged['distance'].describe())
    print()

# ============================================================
# ステップ7: 保存
# ============================================================
print("="*60)
print("保存")
print("="*60)
print()

if len(df_merged) > 0:
    # 統合データを保存
    output_path = "data/processed/merged_data_all.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"統合データ保存: {output_path}")

    # サマリーをテキストファイルに保存
    summary_path = "data/processed/merge_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("データ統合サマリー\n")
        f.write("="*60 + "\n\n")
        f.write(f"netkeibaレコード数: {len(df_netkeiba):,}\n")
        f.write(f"JRAレコード数: {len(df_jra):,}\n")
        f.write(f"重複除去数: {len(df_jra) - len(df_jra_unique):,}\n")
        f.write(f"統合後レコード数: {len(df_merged):,}\n")
        f.write(f"統合後レース数: {df_merged['race_id'].nunique():,}\n")
        f.write(f"\n期間: {df_merged['race_date'].min()} - {df_merged['race_date'].max()}\n")
        f.write(f"\nデータソース:\n")
        f.write(f"  - netkeiba.com (2016-2023)\n")
        f.write(f"  - JRA公式 (2010-2024 G1レース)\n")

    logging.info(f"サマリー保存: {summary_path}")

print()
print("="*60)
print("完了!")
print("="*60)
print()

if len(df_merged) > 0:
    print("次のステップ:")
    print("  1. python feature_engineering.py で特徴量エンジニアリングを実行")
    print("  2. より多くのデータで馬・騎手の勝率を正確に計算")
    print("  3. horse_win_rate, horse_dist_win_rateの精度向上")
    print("  4. popularity_rank依存度の低減")

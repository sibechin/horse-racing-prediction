"""
データ重複チェックと統合スクリプト

1. JRAデータ読み込み
2. netkeibaデータ読み込み
3. 重複を厳格にチェック
4. 重複を除去して統合
5. 最終データセットを保存
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("="*80)
print("データ重複チェックと統合")
print("="*80)
print()

# ============================================================
# ステップ1: データ読み込み
# ============================================================
print("ステップ1: データ読み込み")
print("-"*80)

# JRAデータ
jra_path = "data/jra/g1_races_2010_2024.csv"
if Path(jra_path).exists():
    df_jra = pd.read_csv(jra_path, encoding='utf-8-sig')
    logging.info(f"JRA: {len(df_jra)}レコード, {df_jra['race_id'].nunique()}ユニークレースID")
else:
    df_jra = pd.DataFrame()
    logging.error("JRAデータが見つかりません")

# netkeibaデータ
netkeiba_path = "data/processed/features_engineered.csv"
if Path(netkeiba_path).exists():
    df_netkeiba = pd.read_csv(netkeiba_path, encoding='utf-8-sig')
    logging.info(f"netkeiba: {len(df_netkeiba)}レコード, {df_netkeiba['race_id'].nunique()}ユニークレースID")
else:
    df_netkeiba = pd.DataFrame()
    logging.error("netkeibaデータが見つかりません")

print()

# ============================================================
# ステップ2: 重複チェック用キーの作成
# ============================================================
print("ステップ2: 重複チェック用キーの作成")
print("-"*80)

def create_race_key(row) -> str:
    """
    レース識別キーを作成

    (race_date, venue_name, distance, track_type) の組み合わせ
    """
    try:
        race_date = pd.to_datetime(row['race_date']).strftime('%Y-%m-%d')
        venue = str(row['venue_name']).strip()
        distance = int(row['distance'])
        track = str(row['track_type']).strip()

        return f"{race_date}_{venue}_{distance}_{track}"
    except Exception as e:
        logging.warning(f"キー作成エラー: {e}, row: {row}")
        return None

def create_horse_key(row) -> str:
    """
    馬ごとのユニーク識別キーを作成

    (race_date, venue_name, distance, track_type, horse_name, finish_position)
    """
    try:
        race_key = create_race_key(row)
        if race_key is None:
            return None

        horse = str(row['horse_name']).strip() if 'horse_name' in row else 'unknown'
        finish = int(row['finish_position']) if pd.notna(row['finish_position']) else 0

        return f"{race_key}_{horse}_{finish}"
    except Exception as e:
        logging.warning(f"馬キー作成エラー: {e}")
        return None

# JRAデータのキー作成
if len(df_jra) > 0:
    df_jra['race_key'] = df_jra.apply(create_race_key, axis=1)
    df_jra['horse_key'] = df_jra.apply(create_horse_key, axis=1)

    jra_race_keys = set(df_jra['race_key'].dropna())
    jra_horse_keys = set(df_jra['horse_key'].dropna())

    logging.info(f"JRAユニークレース数: {len(jra_race_keys)}")
    logging.info(f"JRAユニーク馬レコード数: {len(jra_horse_keys)}")

# netkeibaデータのキー作成
if len(df_netkeiba) > 0:
    df_netkeiba['race_key'] = df_netkeiba.apply(create_race_key, axis=1)
    df_netkeiba['horse_key'] = df_netkeiba.apply(create_horse_key, axis=1)

    netkeiba_race_keys = set(df_netkeiba['race_key'].dropna())
    netkeiba_horse_keys = set(df_netkeiba['horse_key'].dropna())

    logging.info(f"netkeibaユニークレース数: {len(netkeiba_race_keys)}")
    logging.info(f"netkeibaユニーク馬レコード数: {len(netkeiba_horse_keys)}")

print()

# ============================================================
# ステップ3: レースレベルでの重複チェック
# ============================================================
print("ステップ3: レースレベルでの重複チェック")
print("-"*80)

if len(df_jra) > 0 and len(df_netkeiba) > 0:
    # レースレベルの重複
    duplicate_race_keys = jra_race_keys & netkeiba_race_keys
    logging.info(f"重複レース数: {len(duplicate_race_keys)}")

    if len(duplicate_race_keys) > 0:
        print()
        print("重複レース詳細:")
        for race_key in sorted(list(duplicate_race_keys))[:20]:  # 最初の20件表示
            # JRAから情報取得
            jra_race = df_jra[df_jra['race_key'] == race_key].iloc[0]
            netkeiba_race = df_netkeiba[df_netkeiba['race_key'] == race_key].iloc[0]

            print(f"  {race_key}")
            print(f"    JRA: {len(df_jra[df_jra['race_key'] == race_key])}頭")
            print(f"    netkeiba: {len(df_netkeiba[df_netkeiba['race_key'] == race_key])}頭")

        if len(duplicate_race_keys) > 20:
            print(f"  ... 他 {len(duplicate_race_keys) - 20}レース")

print()

# ============================================================
# ステップ4: 馬レベルでの重複チェック
# ============================================================
print("ステップ4: 馬レベルでの重複チェック（厳格）")
print("-"*80)

if len(df_jra) > 0 and len(df_netkeiba) > 0:
    # 馬レベルの重複
    duplicate_horse_keys = jra_horse_keys & netkeiba_horse_keys
    logging.info(f"重複馬レコード数: {len(duplicate_horse_keys)}")

    if len(duplicate_horse_keys) > 0:
        print()
        print("重複馬レコード例（最初の10件）:")
        for horse_key in sorted(list(duplicate_horse_keys))[:10]:
            print(f"  {horse_key}")

        if len(duplicate_horse_keys) > 10:
            print(f"  ... 他 {len(duplicate_horse_keys) - 10}件")

print()

# ============================================================
# ステップ5: 重複除去戦略の決定
# ============================================================
print("ステップ5: 重複除去")
print("-"*80)

if len(df_jra) > 0 and len(df_netkeiba) > 0:
    # 戦略: netkeibaデータを優先（より詳細な特徴量を持つため）
    # JRAから重複を除去

    df_jra_unique = df_jra[~df_jra['horse_key'].isin(duplicate_horse_keys)].copy()

    logging.info(f"JRA重複除去前: {len(df_jra)}レコード")
    logging.info(f"JRA重複除去後: {len(df_jra_unique)}レコード")
    logging.info(f"除去されたレコード: {len(df_jra) - len(df_jra_unique)}件")

    # JRAの重複しないレース数を確認
    jra_unique_races = df_jra_unique['race_key'].nunique()
    logging.info(f"JRAユニークレース（重複除去後）: {jra_unique_races}")
else:
    df_jra_unique = df_jra.copy() if len(df_jra) > 0 else pd.DataFrame()

print()

# ============================================================
# ステップ6: データ型の統一
# ============================================================
print("ステップ6: データ型の統一")
print("-"*80)

# JRAデータをnetkeibaフォーマットに合わせる
if len(df_jra_unique) > 0 and len(df_netkeiba) > 0:
    # 共通カラムを特定
    common_cols = list(set(df_jra_unique.columns) & set(df_netkeiba.columns))
    # 'race_key', 'horse_key'を除外
    common_cols = [col for col in common_cols if col not in ['race_key', 'horse_key']]

    logging.info(f"共通カラム数: {len(common_cols)}")

    # JRAデータを共通カラムのみに制限
    df_jra_aligned = df_jra_unique[common_cols].copy()
    df_netkeiba_aligned = df_netkeiba[common_cols].copy()
else:
    df_jra_aligned = df_jra_unique.copy()
    df_netkeiba_aligned = df_netkeiba.copy()

print()

# ============================================================
# ステップ7: データ統合
# ============================================================
print("ステップ7: データ統合")
print("-"*80)

dfs_to_merge = []
if len(df_netkeiba_aligned) > 0:
    dfs_to_merge.append(df_netkeiba_aligned)
    logging.info(f"netkeiba: {len(df_netkeiba_aligned)}レコード")

if len(df_jra_aligned) > 0:
    dfs_to_merge.append(df_jra_aligned)
    logging.info(f"JRA（重複除去後）: {len(df_jra_aligned)}レコード")

if dfs_to_merge:
    df_merged = pd.concat(dfs_to_merge, ignore_index=True)

    # 日付でソート
    df_merged['race_date'] = pd.to_datetime(df_merged['race_date'])
    df_merged = df_merged.sort_values('race_date').reset_index(drop=True)

    logging.info(f"統合完了: {len(df_merged)}レコード")
    logging.info(f"期間: {df_merged['race_date'].min()} - {df_merged['race_date'].max()}")
else:
    df_merged = pd.DataFrame()
    logging.error("統合するデータがありません")

print()

# ============================================================
# ステップ8: 最終重複チェック
# ============================================================
print("ステップ8: 最終重複チェック")
print("-"*80)

if len(df_merged) > 0:
    # 最終データセット内での重複チェック
    df_merged_temp = df_merged.copy()
    df_merged_temp['final_race_key'] = df_merged_temp.apply(create_race_key, axis=1)
    df_merged_temp['final_horse_key'] = df_merged_temp.apply(create_horse_key, axis=1)

    # レースレベルの重複
    final_race_count = df_merged_temp['final_race_key'].nunique()
    logging.info(f"最終ユニークレース数: {final_race_count}")

    # 馬レコードレベルの重複
    duplicated_horses = df_merged_temp[df_merged_temp.duplicated(subset=['final_horse_key'], keep=False)]

    if len(duplicated_horses) > 0:
        logging.warning(f"⚠️ 最終データセットに{len(duplicated_horses)}件の重複馬レコードが残っています!")

        print()
        print("重複馬レコード詳細（最初の10件）:")
        for horse_key in duplicated_horses['final_horse_key'].unique()[:10]:
            dups = df_merged_temp[df_merged_temp['final_horse_key'] == horse_key]
            print(f"  {horse_key}: {len(dups)}件")
            for idx, row in dups.iterrows():
                print(f"    - レースID: {row['race_id']}, 馬名: {row['horse_name']}, 着順: {row['finish_position']}")

        # 重複を除去（最初のレコードを保持）
        logging.info("重複馬レコードを除去します（最初のレコードを保持）")
        df_merged = df_merged_temp.drop_duplicates(subset=['final_horse_key'], keep='first')
        df_merged = df_merged.drop(columns=['final_race_key', 'final_horse_key'])

        logging.info(f"重複除去後: {len(df_merged)}レコード")
    else:
        logging.info("✓ 重複なし - データセットは正常です")

print()

# ============================================================
# ステップ9: 統計情報
# ============================================================
print("="*80)
print("最終データセット統計")
print("="*80)
print()

if len(df_merged) > 0:
    print(f"総レコード数: {len(df_merged):,}")

    # race_idでユニークレース数をカウント
    unique_races = df_merged['race_id'].nunique()
    print(f"総レース数: {unique_races:,}")
    print(f"期間: {df_merged['race_date'].min()} - {df_merged['race_date'].max()}")
    print()

    # データソース別集計
    if 'year' in df_merged.columns or 'race_date' in df_merged.columns:
        if 'year' not in df_merged.columns:
            df_merged['year'] = pd.to_datetime(df_merged['race_date']).dt.year

        print("年別レース数:")
        year_counts = df_merged.groupby('year')['race_id'].nunique().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count}レース")
        print()

    # 競馬場別
    if 'venue_name' in df_merged.columns:
        print("競馬場別レース数（上位10）:")
        venue_counts = df_merged.groupby('venue_name')['race_id'].nunique().sort_values(ascending=False)
        for venue, count in venue_counts.head(10).items():
            print(f"  {venue}: {count}レース")
        print()

    # 馬場タイプ別
    if 'track_type' in df_merged.columns:
        print("馬場タイプ別:")
        track_counts = df_merged['track_type'].value_counts()
        for track, count in track_counts.items():
            print(f"  {track}: {count}レコード")
        print()

# ============================================================
# ステップ10: 保存
# ============================================================
print("="*80)
print("保存")
print("="*80)
print()

if len(df_merged) > 0:
    # 最終統合データを保存
    output_path = "data/processed/final_merged_deduplicated.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"最終データ保存: {output_path}")

    # サマリーをテキストファイルに保存
    summary_path = "data/processed/deduplication_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("データ重複チェックと統合サマリー\n")
        f.write("="*80 + "\n\n")

        f.write(f"【入力データ】\n")
        f.write(f"  netkeiba: {len(df_netkeiba):,}レコード, {df_netkeiba['race_id'].nunique() if len(df_netkeiba) > 0 else 0}レース\n")
        f.write(f"  JRA: {len(df_jra):,}レコード, {df_jra['race_id'].nunique() if len(df_jra) > 0 else 0}レース\n\n")

        f.write(f"【重複検出】\n")
        if len(df_jra) > 0 and len(df_netkeiba) > 0:
            f.write(f"  重複レース数: {len(duplicate_race_keys)}\n")
            f.write(f"  重複馬レコード数: {len(duplicate_horse_keys)}\n\n")

        f.write(f"【重複除去】\n")
        f.write(f"  JRA重複除去: {len(df_jra) - len(df_jra_unique):,}レコード\n")
        f.write(f"  JRA残存: {len(df_jra_unique):,}レコード\n\n")

        f.write(f"【最終データセット】\n")
        f.write(f"  総レコード数: {len(df_merged):,}\n")
        f.write(f"  総レース数: {df_merged['race_id'].nunique():,}\n")
        f.write(f"  期間: {df_merged['race_date'].min()} - {df_merged['race_date'].max()}\n\n")

        f.write(f"【データソース】\n")
        f.write(f"  - netkeiba.com (2016-2023)\n")
        f.write(f"  - JRA公式 (2010-2024 G1レース、重複除去後)\n")

    logging.info(f"サマリー保存: {summary_path}")

print()
print("="*80)
print("完了!")
print("="*80)
print()

if len(df_merged) > 0:
    print("次のステップ:")
    print("  1. python clean_data.py で障害レース除外・異常値除去")
    print("  2. python feature_engineering.py で特徴量エンジニアリング")
    print("  3. モデルトレーニング")

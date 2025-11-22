"""
重複データのサマリー
"""
import pandas as pd

def summarize_duplicates(file_path, dataset_name):
    """重複データのサマリー"""
    print(f"\n{'='*70}")
    print(f"{dataset_name}")
    print(f"{'='*70}")

    df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(f"\n基本統計:")
    print(f"  総レコード数: {len(df):,}")
    print(f"  ユニークrace_id数: {df['race_id'].nunique():,}")

    # 同日同名レースの確認
    df['date_race'] = df['expected_date'] + '_' + df['expected_race_name']

    # 各date_raceごとのrace_id数
    date_race_counts = df.groupby('date_race')['race_id'].nunique()

    total_date_races = len(date_race_counts)
    duplicated_date_races = (date_race_counts > 1).sum()

    print(f"\n重複分析:")
    print(f"  ユニークな「日付+レース名」組: {total_date_races:,}")
    print(f"  そのうち重複あり: {duplicated_date_races:,} ({duplicated_date_races/total_date_races*100:.1f}%)")
    print(f"  重複なし（正常）: {total_date_races - duplicated_date_races:,}")

    # 年ごとのレース数
    df['year'] = pd.to_datetime(df['expected_date']).dt.year
    year_race_counts = df.groupby('year')['race_id'].nunique().sort_index()

    print(f"\n年ごとのG1レース数:")
    for year, count in year_race_counts.items():
        expected_g1 = 25
        ratio = count / expected_g1
        status = "OK" if ratio <= 1.2 else f"WARNING: x{ratio:.1f}"
        print(f"  {year}: {count:3d} races {status}")

    # クリーニング後の予測
    unique_date_races = total_date_races
    current_races = df['race_id'].nunique()
    reduction = current_races - unique_date_races

    print(f"\nクリーニング予測:")
    print(f"  現在のrace_id数: {current_races:,}")
    print(f"  クリーニング後: {unique_date_races:,} (推定)")
    print(f"  削減数: {reduction:,} ({reduction/current_races*100:.1f}%減)")


# 訓練データ
summarize_duplicates(
    'data/processed/keibalab_g1_2000_2024_processed.csv',
    '訓練データ (2000-2024)'
)

# テストデータ
summarize_duplicates(
    'data/processed/keibalab_g1_2025_processed.csv',
    'テストデータ (2025)'
)

print(f"\n{'='*70}")
print("結論")
print(f"{'='*70}")
print("\n⚠️ 両データセットに深刻な重複問題あり")
print("\n推奨アクション:")
print("1. データクリーニング実施（重複race_id除去）")
print("2. クリーニング済みデータで再訓練")
print("3. 正確な評価を実施")

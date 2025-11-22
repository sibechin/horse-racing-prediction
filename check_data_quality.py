"""
データ品質チェック: 重複レースの確認
"""
import pandas as pd
from collections import Counter

def check_data_quality(file_path, dataset_name):
    """データ品質チェック"""
    print(f"\n{'='*70}")
    print(f"データ品質チェック: {dataset_name}")
    print(f"ファイル: {file_path}")
    print(f"{'='*70}")

    df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(f"\n基本情報:")
    print(f"  総レコード数: {len(df):,}")
    print(f"  ユニークrace_id数: {df['race_id'].nunique():,}")

    # レース情報の集計
    race_info = df.groupby('race_id').agg({
        'expected_race_name': 'first',
        'expected_date': 'first',
        'horse_name': 'count'
    }).rename(columns={'horse_name': 'num_horses'})

    race_info = race_info.sort_values('expected_date')

    # レース名ごとの出現回数
    race_name_counts = race_info['expected_race_name'].value_counts()

    print(f"\n重複の可能性があるレース:")
    print(f"  (同じレース名が複数回出現)")

    duplicates_found = False
    for race_name, count in race_name_counts.items():
        if count > 1:
            duplicates_found = True
            print(f"\n  【{race_name}】: {count}回出現")

            # そのレース名の全race_idを表示
            dup_races = race_info[race_info['expected_race_name'] == race_name]
            for race_id, row in dup_races.iterrows():
                print(f"    - {race_id} | {row['expected_date']} | {row['num_horses']}頭")

    if not duplicates_found:
        print("  重複なし")

    # 年ごとのレース数
    print(f"\n年ごとのG1レース数:")
    df['year'] = pd.to_datetime(df['expected_date']).dt.year
    year_race_counts = df.groupby('year')['race_id'].nunique().sort_index()

    for year, count in year_race_counts.items():
        expected_g1 = 25  # 日本の年間G1レース数の目安
        status = "✓" if count <= expected_g1 else "⚠️ 多すぎる"
        print(f"  {year}: {count:2d}レース {status}")

    # 同日同名レースの詳細確認
    print(f"\n同日同名レースの確認:")
    df['date_race'] = df['expected_date'] + '_' + df['expected_race_name']
    same_day_races = df.groupby('date_race')['race_id'].nunique()
    duplicates = same_day_races[same_day_races > 1]

    if len(duplicates) > 0:
        print(f"  ⚠️ 同日同名で複数race_idが存在: {len(duplicates)}件")
        for date_race, count in duplicates.head(10).items():
            date, race_name = date_race.rsplit('_', 1)
            print(f"    - {date} {race_name}: {count}個のrace_id")
    else:
        print("  ✓ 同日同名の重複なし")

    return df, race_info, duplicates_found


def main():
    # 訓練データチェック
    train_df, train_race_info, train_dup = check_data_quality(
        'data/processed/keibalab_g1_2000_2024_processed.csv',
        '訓練データ (2000-2024)'
    )

    # テストデータチェック
    test_df, test_race_info, test_dup = check_data_quality(
        'data/processed/keibalab_g1_2025_processed.csv',
        'テストデータ (2025)'
    )

    # 総括
    print(f"\n{'='*70}")
    print("総括")
    print(f"{'='*70}")

    if train_dup or test_dup:
        print("\n⚠️ 重複データが検出されました")
        print("\n推奨アクション:")
        print("1. 重複レースを除去（各レース名で最新のrace_idのみ保持）")
        print("2. クリーニング済みデータで再訓練")
        print("3. 正確な評価を実施")
    else:
        print("\n✓ データに問題は検出されませんでした")


if __name__ == "__main__":
    main()

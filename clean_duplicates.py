"""
重複レースの除去

各「日付+レース名」組で最大のrace_idのみを保持
（最大のrace_idが最新のデータと仮定）
"""
import pandas as pd
from pathlib import Path


def clean_duplicates(input_file, output_file, dataset_name):
    """重複レースの除去"""
    print(f"\n{'='*70}")
    print(f"クリーニング: {dataset_name}")
    print(f"{'='*70}")

    # データ読み込み
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    original_count = len(df)
    original_races = df['race_id'].nunique()

    print(f"\n元データ:")
    print(f"  レコード数: {original_count:,}")
    print(f"  race_id数: {original_races:,}")

    # 日付+レース名で重複を特定
    df['date_race'] = df['expected_date'] + '_' + df['expected_race_name']

    # 各date_raceごとに最大のrace_idを選択
    max_race_ids = df.groupby('date_race')['race_id'].max().reset_index()
    max_race_ids = max_race_ids['race_id'].unique()

    # 最大race_idのみ残す
    df_cleaned = df[df['race_id'].isin(max_race_ids)].copy()
    df_cleaned = df_cleaned.drop(columns=['date_race'])

    cleaned_count = len(df_cleaned)
    cleaned_races = df_cleaned['race_id'].nunique()

    print(f"\nクリーニング後:")
    print(f"  レコード数: {cleaned_count:,}")
    print(f"  race_id数: {cleaned_races:,}")
    print(f"\n削減:")
    print(f"  レコード: {original_count - cleaned_count:,} ({(original_count - cleaned_count)/original_count*100:.1f}%)")
    print(f"  race_id: {original_races - cleaned_races:,} ({(original_races - cleaned_races)/original_races*100:.1f}%)")

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n保存: {output_file}")

    return df_cleaned


def main():
    """メイン処理"""
    print("\n" + "="*70)
    print("データクリーニング開始")
    print("="*70)

    # 訓練データクリーニング
    train_cleaned = clean_duplicates(
        input_file='data/processed/keibalab_g1_2000_2024_processed.csv',
        output_file='data/processed/keibalab_g1_2000_2024_cleaned.csv',
        dataset_name='訓練データ (2000-2024)'
    )

    # テストデータクリーニング
    test_cleaned = clean_duplicates(
        input_file='data/processed/keibalab_g1_2025_processed.csv',
        output_file='data/processed/keibalab_g1_2025_cleaned.csv',
        dataset_name='テストデータ (2025)'
    )

    # サマリー
    print(f"\n{'='*70}")
    print("クリーニング完了")
    print(f"{'='*70}")

    print(f"\n次のステップ:")
    print(f"1. クリーニング済みデータで再訓練:")
    print(f"   - train_top3_no_odds.py を修正して *_cleaned.csv を使用")
    print(f"2. 正確な評価を実施")
    print(f"\nクリーニング済みファイル:")
    print(f"  - data/processed/keibalab_g1_2000_2024_cleaned.csv")
    print(f"  - data/processed/keibalab_g1_2025_cleaned.csv")


if __name__ == "__main__":
    main()

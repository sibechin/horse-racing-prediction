"""
keibalab G1データ(2000-2024)の前処理
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_race_features(race_id):
    """
    race_idから特徴量を抽出
    Format: YYYYMMDDVVRR
    """
    race_id_str = str(race_id)
    return {
        'venue_code': int(race_id_str[8:10]),  # 05=東京, 06=中山, 07=中京, 08=京都, 09=阪神
        'race_number': int(race_id_str[10:12])
    }


def parse_horse_weight(hw_str):
    """
    馬体重文字列をパース
    例: '500(+10)' → (500, 10), '480(-5)' → (480, -5)
    """
    if pd.isna(hw_str) or hw_str == '':
        return np.nan, np.nan

    try:
        # '500(+10)' or '480(-5)' 形式
        if '(' in str(hw_str):
            weight_part = str(hw_str).split('(')[0]
            change_part = str(hw_str).split('(')[1].replace(')', '')

            weight = int(weight_part)

            # 変化量
            if '+' in change_part or '-' in change_part or change_part.isdigit():
                change = int(change_part.replace('+', '').replace('＋', ''))
            else:
                change = 0

            return weight, change
        else:
            # 数字のみの場合
            return int(hw_str), 0
    except:
        return np.nan, np.nan


def parse_time(time_str):
    """
    タイム文字列を秒数に変換
    例: '3:30.2' → 210.2
    """
    if pd.isna(time_str) or time_str == '':
        return np.nan

    try:
        if ':' in str(time_str):
            parts = str(time_str).split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
    except:
        pass
    return np.nan


def preprocess_keibalab_g1(input_file: str, output_file: str):
    """
    keibalab G1データの前処理

    Args:
        input_file: 入力CSVファイル
        output_file: 出力CSVファイル
    """
    logging.info("="*60)
    logging.info("keibalab G1データ前処理開始")
    logging.info("="*60)

    # データ読み込み
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    logging.info(f"\n元データ: {len(df)} records, {df['race_id'].nunique()} races")
    logging.info(f"期間: {df['expected_year'].min()}-{df['expected_year'].max()}年")

    # 1. 基本特徴量の抽出
    logging.info("\n特徴量エンジニアリング...")

    # Sex and Age
    if 'sex_age' in df.columns:
        df['sex'] = df['sex_age'].str[0]
        df['age'] = df['sex_age'].str[1:].astype(int, errors='ignore')

    # Date features
    if 'expected_date' in df.columns:
        df['race_date'] = pd.to_datetime(df['expected_date'])
        df['year'] = df['race_date'].dt.year
        df['month'] = df['race_date'].dt.month
        df['day_of_week'] = df['race_date'].dt.dayofweek

    # Race ID features (venue, race number)
    race_features = df['race_id'].apply(extract_race_features)
    df['venue_code'] = race_features.apply(lambda x: x['venue_code'])
    df['race_number'] = race_features.apply(lambda x: x['race_number'])

    # Time to seconds
    if 'time' in df.columns:
        df['time_seconds'] = df['time'].apply(parse_time)

    # Horse weight and weight change
    if 'horse_weight' in df.columns:
        weight_data = df['horse_weight'].apply(parse_horse_weight)
        df['horse_weight_kg'] = weight_data.apply(lambda x: x[0])
        df['horse_weight_change'] = weight_data.apply(lambda x: x[1])

    # last_3f (上がり3F) - already numeric

    # Odds: convert to numeric
    if 'odds' in df.columns:
        df['odds_numeric'] = pd.to_numeric(df['odds'], errors='coerce')

    # Popularity (already numeric)

    # 2. レース単位の統計量
    logging.info("レース単位の統計量を計算...")

    # Field size (出走頭数)
    df['field_size'] = df.groupby('race_id')['horse_name'].transform('count')

    # Race average odds
    if 'odds_numeric' in df.columns:
        df['race_avg_odds'] = df.groupby('race_id')['odds_numeric'].transform('mean')

    # Race average weight
    if 'weight' in df.columns:
        df['race_avg_weight'] = df.groupby('race_id')['weight'].transform('mean')

    # 3. Target変数の準備
    logging.info("Target変数の準備...")

    # finish_position to numeric
    if 'finish_position' in df.columns:
        df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

        # 1-3着をTop3として分類
        df['is_top3'] = (df['finish_position_numeric'] <= 3).astype(int)
        df['is_winner'] = (df['finish_position_numeric'] == 1).astype(int)

    # 4. カテゴリカル変数のエンコーディング
    logging.info("カテゴリカル変数のエンコーディング...")

    # Sex encoding
    sex_map = {'牡': 0, '牝': 1, 'セ': 2}
    if 'sex' in df.columns:
        df['sex_encoded'] = df['sex'].map(sex_map).fillna(-1).astype(int)

    # Trainer encoding (頻度エンコーディング)
    if 'trainer' in df.columns:
        trainer_counts = df['trainer'].value_counts()
        df['trainer_frequency'] = df['trainer'].map(trainer_counts).fillna(0)

        # Top N trainers only
        top_trainers = trainer_counts.head(50).index
        df['trainer_top50'] = df['trainer'].apply(lambda x: x if x in top_trainers else 'Other')

        # Label encoding
        le_trainer = LabelEncoder()
        df['trainer_encoded'] = le_trainer.fit_transform(df['trainer_top50'])

    # Jockey encoding (頻度エンコーディング)
    if 'jockey' in df.columns:
        jockey_counts = df['jockey'].value_counts()
        df['jockey_frequency'] = df['jockey'].map(jockey_counts).fillna(0)

        # Top N jockeys only
        top_jockeys = jockey_counts.head(50).index
        df['jockey_top50'] = df['jockey'].apply(lambda x: x if x in top_jockeys else 'Other')

        # Label encoding
        le_jockey = LabelEncoder()
        df['jockey_encoded'] = le_jockey.fit_transform(df['jockey_top50'])

    # Expected race name encoding (G1レース種別)
    if 'expected_race_name' in df.columns:
        le_race = LabelEncoder()
        df['race_type_encoded'] = le_race.fit_transform(df['expected_race_name'].fillna('Unknown'))

    # 5. 欠損値の処理
    logging.info("欠損値の処理...")

    # 数値特徴量の欠損値を中央値で埋める
    numeric_features = ['weight', 'popularity', 'odds_numeric', 'last_3f',
                       'horse_weight_kg', 'horse_weight_change', 'time_seconds']

    for col in numeric_features:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logging.info(f"  {col}: {df[col].isnull().sum()} missing values filled with {median_val:.2f}")

    # 6. 特徴量選択
    logging.info("\n最終特徴量の選択...")

    feature_columns = [
        # 基本特徴量
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight', 'popularity', 'odds_numeric',
        'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size', 'race_avg_odds', 'race_avg_weight',

        # エンコード済みカテゴリカル
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded',

        # Target
        'finish_position_numeric', 'is_top3', 'is_winner',

        # メタデータ
        'race_id', 'horse_name', 'expected_race_name', 'expected_date'
    ]

    # 存在するカラムのみ選択
    available_columns = [col for col in feature_columns if col in df.columns]
    df_processed = df[available_columns].copy()

    # 7. データ品質チェック
    logging.info("\nデータ品質チェック...")

    # 無効なレコードを除外
    before_count = len(df_processed)

    # finish_positionが欠損しているレコードを除外
    df_processed = df_processed[df_processed['finish_position_numeric'].notna()]

    # 主要特徴量が全て欠損しているレコードを除外
    key_features = ['weight', 'popularity', 'odds_numeric']
    df_processed = df_processed[df_processed[key_features].notna().all(axis=1)]

    after_count = len(df_processed)
    logging.info(f"  除外されたレコード: {before_count - after_count} ({(before_count - after_count) / before_count * 100:.2f}%)")

    # 8. 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 9. サマリー
    logging.info("\n" + "="*60)
    logging.info("前処理完了!")
    logging.info("="*60)
    logging.info(f"最終データ数: {len(df_processed)} records")
    logging.info(f"レース数: {df_processed['race_id'].nunique()}")
    logging.info(f"期間: {df_processed['year'].min()}-{df_processed['year'].max()}年")
    logging.info(f"G1レース種類: {df_processed['expected_race_name'].nunique()}")
    logging.info(f"特徴量数: {len(df_processed.columns)}")
    logging.info(f"保存先: {output_path}")

    # 統計情報
    logging.info("\nTarget分布:")
    logging.info(f"  1着: {df_processed['is_winner'].sum()} ({df_processed['is_winner'].mean() * 100:.1f}%)")
    logging.info(f"  Top3: {df_processed['is_top3'].sum()} ({df_processed['is_top3'].mean() * 100:.1f}%)")

    logging.info("\n年別データ数:")
    print(df_processed.groupby('year').size())

    logging.info("\nG1レース別データ数 (Top 10):")
    print(df_processed['expected_race_name'].value_counts().head(10))

    logging.info("\n競馬場別データ数:")
    venue_map = {5: '東京', 6: '中山', 7: '中京', 8: '京都', 9: '阪神'}
    venue_counts = df_processed['venue_code'].value_counts()
    for code, count in venue_counts.items():
        logging.info(f"  {venue_map.get(code, f'Code {code}')}: {count}")

    return df_processed


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("keibalab G1データ前処理スクリプト")
    logging.info("="*60)

    # ファイルパス
    input_file = 'data/raw/keibalab/g1_improved_2024_2000_20251120_152632.csv'
    output_file = 'data/processed/keibalab_g1_2000_2024_processed.csv'

    # 前処理実行
    df_processed = preprocess_keibalab_g1(input_file, output_file)

    logging.info("\n" + "="*60)
    logging.info("完了!")
    logging.info("="*60)


if __name__ == "__main__":
    main()

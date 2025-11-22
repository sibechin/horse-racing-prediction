"""
マイルチャンピオンシップのデータを予想システム用に変換
"""
import pandas as pd
import numpy as np
from datetime import datetime

def convert_mile_cs_data(input_file, output_file):
    """
    mile_cs_2025.csvを予想システム用の形式に変換
    """
    # データ読み込み
    df = pd.read_csv(input_file)

    print(f"入力データ: {len(df)}頭")
    print(f"カラム: {df.columns.tolist()}")

    # 新しいデータフレームを作成
    converted = pd.DataFrame()

    # 基本情報（全行に同じ値を設定）
    num_horses = len(df)
    converted['race_id'] = ['202511230811'] * num_horses  # 2025年11月23日京都11R
    converted['race_name'] = ['マイルチャンピオンシップ'] * num_horses
    converted['horse_number'] = df['horse_number'].values
    converted['horse_name'] = df['horse_name'].values

    # 日付情報（2025年11月23日＝土曜日）
    converted['year'] = 2025
    converted['month'] = 11
    converted['day_of_week'] = 6  # 0=月曜, 6=日曜

    # 競馬場情報
    converted['venue_code'] = 8  # 京都競馬場
    converted['race_number'] = 11  # 11R

    # 馬情報
    converted['age'] = df['age']
    converted['weight'] = df['jockey_weight']  # 斤量

    # 性別エンコーディング（データに含まれていないのでageから推測）
    # 3歳牝馬は斤量が軽い傾向があるので、斤量57以下を牝馬と仮定
    # ただし、より正確にはデータに含める必要がある
    converted['sex_encoded'] = df.apply(
        lambda row: 1 if row['jockey_weight'] <= 56.5 and row['age'] <= 4 else 0,
        axis=1
    )

    # 人気とオッズ（勝率から推定）
    # jockey_win_rate * trainer_win_rateを総合評価として使用
    df['combined_rating'] = df['jockey_win_rate'] * df['trainer_win_rate']
    df['popularity_rank'] = df['combined_rating'].rank(ascending=False, method='dense').astype(int)

    converted['popularity'] = df['popularity_rank']

    # オッズは人気から推定（1番人気=2.5倍、2番人気=4.5倍...）
    odds_map = {
        1: 2.5, 2: 4.5, 3: 7.0, 4: 10.0, 5: 15.0,
        6: 20.0, 7: 30.0, 8: 45.0, 9: 65.0, 10: 90.0,
        11: 120.0, 12: 160.0, 13: 210.0, 14: 280.0, 15: 360.0,
        16: 450.0, 17: 550.0, 18: 650.0
    }
    converted['odds_numeric'] = df['popularity_rank'].map(odds_map)

    # レースデータ
    converted['last_3f'] = df['last_3f']
    converted['horse_weight_kg'] = df['horse_weight_kg']
    converted['horse_weight_change'] = df['weight_change']
    converted['field_size'] = df['field_size']

    # レース全体の平均値
    converted['race_avg_odds'] = converted['odds_numeric'].mean()
    converted['race_avg_weight'] = converted['weight'].mean()

    # 調教師とジョッキーのエンコーディング
    # 名前をハッシュ化してエンコード
    trainer_map = {name: idx for idx, name in enumerate(df['trainer_name'].unique(), 1)}
    jockey_map = {name: idx for idx, name in enumerate(df['jockey_name'].unique(), 1)}

    converted['trainer_name'] = df['trainer_name']
    converted['trainer_encoded'] = df['trainer_name'].map(trainer_map)
    converted['trainer_frequency'] = df['trainer_frequency']

    converted['jockey_name'] = df['jockey_name']
    converted['jockey_encoded'] = df['jockey_name'].map(jockey_map)
    converted['jockey_frequency'] = df['jockey_frequency']

    # レースタイプ（芝1600m G1）
    converted['race_type_encoded'] = 1  # 芝

    # 追加情報（参考用）
    if 'corner_rank_4' in df.columns:
        converted['corner_rank_4'] = df['corner_rank_4']
    if 'horse_win_rate' in df.columns:
        converted['horse_win_rate'] = df['horse_win_rate']
    if 'jockey_win_rate' in df.columns:
        converted['jockey_win_rate'] = df['jockey_win_rate']
    if 'trainer_win_rate' in df.columns:
        converted['trainer_win_rate'] = df['trainer_win_rate']

    # 保存
    converted.to_csv(output_file, index=False)
    print(f"\n変換完了: {output_file}")
    print(f"出力データ: {len(converted)}頭")
    print(f"\n変換後のカラム:")
    for col in converted.columns:
        print(f"  - {col}")

    # サマリー表示
    print("\n" + "="*60)
    print("マイルチャンピオンシップ 2025 出走馬一覧")
    print("="*60)
    for idx, row in converted.iterrows():
        print(f"{int(row['horse_number']):2d}. {row['horse_name']:20s} "
              f"({int(row['age'])}歳 {row['weight']:.1f}kg) "
              f"J:{row['jockey_name']:10s} "
              f"人気:{int(row['popularity'])}番 "
              f"オッズ:{row['odds_numeric']:.1f}倍")

    return converted


if __name__ == '__main__':
    input_file = 'data/mile_cs_2025.csv'
    output_file = 'data/mile_cs_2025_converted.csv'

    df_converted = convert_mile_cs_data(input_file, output_file)

    print("\n" + "="*60)
    print("予想実行コマンド:")
    print("="*60)
    print(f"python predict_races.py --input {output_file} --output predictions/mile_cs_2025_predictions.csv")

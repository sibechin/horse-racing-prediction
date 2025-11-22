"""
11月23日のレースデータを作成（サンプル）
実際のデータは公式サイトから取得してください
"""
import pandas as pd
from pathlib import Path

def create_sample_races_nov23():
    """
    11月23日のサンプルレースデータを作成

    注意: これはデモンストレーション用のサンプルデータです。
    実際の予測には、公式サイトから最新の出馬表を取得してください。
    """

    # レース1: 東京11R - ジャパンカップ（G1）
    # 実際の出走馬データが必要です
    race1_data = []

    # サンプル馬データ（実際のデータに置き換えてください）
    sample_horses = [
        # 馬名, 性別(0=牡,1=牝), 年齢, 斤量, 人気, オッズ, 上がり3F予想, 馬体重, 体重変化
        ("イクイノックス", 0, 5, 58.0, 1, 2.5, 34.5, 520, 0),
        ("ドウデュース", 0, 4, 58.0, 2, 4.2, 34.8, 496, 2),
        ("ジャスティンパレス", 0, 5, 58.0, 3, 6.8, 35.0, 480, -3),
        ("タイトルホルダー", 0, 5, 58.0, 4, 8.5, 35.2, 508, 5),
        ("パンサラッサ", 0, 6, 58.0, 5, 12.3, 35.5, 472, -2),
        ("アスクビクターモア", 0, 7, 58.0, 6, 15.8, 35.8, 485, 1),
        ("ヴェラアズール", 1, 4, 56.0, 7, 20.4, 35.3, 458, 3),
        ("リバティアイランド", 1, 4, 56.0, 8, 25.6, 35.6, 462, -1),
        ("スターズオンアース", 0, 4, 58.0, 9, 32.8, 36.0, 490, 2),
        ("ディープボンド", 0, 6, 58.0, 10, 45.2, 36.2, 502, -4),
    ]

    for i, (horse_name, sex, age, weight, pop, odds, last3f, hw, hwc) in enumerate(sample_horses):
        race1_data.append({
            'race_id': '20251123_tokyo_11',
            'race_name': 'ジャパンカップ',
            'race_date': '2025-11-23',
            'venue_code': 5,  # 東京
            'race_number': 11,
            'horse_name': horse_name,
            'sex_encoded': sex,
            'age': age,
            'weight': weight,
            'popularity': pop,
            'odds_numeric': odds,
            'last_3f': last3f,
            'horse_weight_kg': hw,
            'horse_weight_change': hwc,
            'trainer_encoded': i + 1,  # 簡略化
            'trainer_frequency': 50 - i * 2,
            'jockey_encoded': i + 1,  # 簡略化
            'jockey_frequency': 100 - i * 5,
            'year': 2025,
            'month': 11,
            'day_of_week': 5,  # 土曜日
            'field_size': len(sample_horses),
            'race_avg_odds': sum(h[5] for h in sample_horses) / len(sample_horses),
            'race_avg_weight': 57.6,
            'race_type_encoded': 0
        })

    # レース2: 阪神11R - 阪神ジュベナイルフィリーズ（G1）
    race2_horses = [
        ("ブレイディヴェーグ", 1, 2, 54.0, 1, 3.2, 34.0, 456, 2),
        ("ステレンボッシュ", 1, 2, 54.0, 2, 5.8, 34.3, 448, 1),
        ("オルフェクレール", 1, 2, 54.0, 3, 8.4, 34.5, 462, -2),
        ("フォーエバーヤング", 1, 2, 54.0, 4, 12.6, 34.8, 440, 3),
        ("リバティファイター", 1, 2, 54.0, 5, 18.9, 35.0, 455, 0),
        ("トゥザレジェンド", 1, 2, 54.0, 6, 25.4, 35.2, 450, -1),
        ("スキルヴィング", 1, 2, 54.0, 7, 35.6, 35.5, 458, 2),
        ("ペプチドバンブー", 1, 2, 54.0, 8, 48.2, 35.8, 445, -3),
    ]

    race2_data = []
    for i, (horse_name, sex, age, weight, pop, odds, last3f, hw, hwc) in enumerate(race2_horses):
        race2_data.append({
            'race_id': '20251123_hanshin_11',
            'race_name': '阪神ジュベナイルフィリーズ',
            'race_date': '2025-11-23',
            'venue_code': 9,  # 阪神
            'race_number': 11,
            'horse_name': horse_name,
            'sex_encoded': sex,
            'age': age,
            'weight': weight,
            'popularity': pop,
            'odds_numeric': odds,
            'last_3f': last3f,
            'horse_weight_kg': hw,
            'horse_weight_change': hwc,
            'trainer_encoded': i + 1,
            'trainer_frequency': 45 - i * 2,
            'jockey_encoded': i + 1,
            'jockey_frequency': 95 - i * 5,
            'year': 2025,
            'month': 11,
            'day_of_week': 5,
            'field_size': len(race2_horses),
            'race_avg_odds': sum(h[5] for h in race2_horses) / len(race2_horses),
            'race_avg_weight': 54.0,
            'race_type_encoded': 0
        })

    # 全データを結合
    all_races = race1_data + race2_data

    return pd.DataFrame(all_races)


def main():
    """メイン実行"""
    print("="*60)
    print("11月23日のサンプルレースデータ作成")
    print("="*60)
    print("\n注意: これはデモンストレーション用のサンプルデータです。")
    print("実際の予測には、公式サイトから最新の出馬表を取得してください。\n")

    # データ作成
    df = create_sample_races_nov23()

    # 保存
    output_file = 'race_20251123_sample.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"✓ サンプルデータ作成完了: {output_file}")
    print(f"  - レース数: {df['race_id'].nunique()}")
    print(f"  - 出走馬数: {len(df)}")

    print("\nレース内訳:")
    for race_id in df['race_id'].unique():
        race_df = df[df['race_id'] == race_id]
        race_name = race_df.iloc[0]['race_name']
        print(f"  - {race_name}: {len(race_df)}頭")

    print(f"\n次のステップ:")
    print(f"  1. このファイルを確認: {output_file}")
    print(f"  2. 実際のデータに置き換える（オプション）")
    print(f"  3. 予測を実行: python predict_races.py --input {output_file}")


if __name__ == "__main__":
    main()

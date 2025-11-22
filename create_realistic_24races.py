"""
2025年11月23日 東京・京都の24レース
より現実的なサンプルデータを作成
"""
import pandas as pd
import numpy as np

def create_realistic_races():
    """現実的な24レースのデータを作成"""

    all_races = []

    # ===== 東京競馬場 12レース =====
    tokyo_races = [
        # 1R-10R: 一般レース（10-16頭）
        {'num': 1, 'name': '東京1R', 'horses': 16, 'grade': 'normal'},
        {'num': 2, 'name': '東京2R', 'horses': 16, 'grade': 'normal'},
        {'num': 3, 'name': '東京3R', 'horses': 16, 'grade': 'normal'},
        {'num': 4, 'name': '東京4R', 'horses': 15, 'grade': 'normal'},
        {'num': 5, 'name': '東京5R', 'horses': 14, 'grade': 'normal'},
        {'num': 6, 'name': '東京6R', 'horses': 16, 'grade': 'normal'},
        {'num': 7, 'name': '東京7R', 'horses': 14, 'grade': 'normal'},
        {'num': 8, 'name': '東京8R', 'horses': 16, 'grade': 'normal'},
        {'num': 9, 'name': '東京9R', 'horses': 16, 'grade': 'normal'},
        {'num': 10, 'name': 'キャピタルS', 'horses': 16, 'grade': '3勝'},
        # 11R: G1ジャパンカップ
        {'num': 11, 'name': 'ジャパンカップ', 'horses': 18, 'grade': 'G1'},
        # 12R: 特別戦
        {'num': 12, 'name': 'ノエル賞', 'horses': 16, 'grade': '2勝'},
    ]

    # G1ジャパンカップの出走馬（より現実的な馬名と人気）
    jc_horses = [
        ('イクイノックス', 0, 5, 58.0, 1, 2.5, 34.5, 520, 0),
        ('ドウデュース', 0, 4, 58.0, 2, 4.2, 34.8, 496, 2),
        ('ジャスティンパレス', 0, 5, 58.0, 3, 6.8, 35.0, 480, -3),
        ('タイトルホルダー', 0, 5, 58.0, 4, 8.5, 35.2, 508, 5),
        ('スターズオンアース', 0, 4, 58.0, 5, 12.3, 35.5, 490, -2),
        ('ディープボンド', 0, 6, 58.0, 6, 15.8, 35.8, 502, 1),
        ('ヴェラアズール', 1, 4, 56.0, 7, 20.4, 35.3, 458, 3),
        ('リバティアイランド', 1, 4, 56.0, 8, 25.6, 35.6, 462, -1),
        ('パンサラッサ', 0, 6, 58.0, 9, 32.8, 36.0, 472, 2),
        ('アスクビクターモア', 0, 7, 58.0, 10, 45.2, 36.2, 485, -4),
        ('エピファネイア', 0, 5, 58.0, 11, 52.3, 36.5, 495, 0),
        ('シュヴァルグラン', 0, 6, 58.0, 12, 68.4, 36.8, 510, -2),
        ('キセキ', 0, 6, 58.0, 13, 85.6, 37.0, 488, 3),
        ('ワグネリアン', 0, 5, 58.0, 14, 102.3, 37.2, 492, 1),
        ('ブラストワンピース', 0, 6, 58.0, 15, 128.5, 37.5, 498, -1),
        ('ミッキーブリランテ', 0, 6, 58.0, 16, 156.8, 37.8, 486, 2),
        ('サートゥルナーリア', 0, 5, 58.0, 17, 198.5, 38.0, 494, 0),
        ('フィエールマン', 0, 7, 58.0, 18, 245.6, 38.2, 500, -3),
    ]

    for race_info in tokyo_races:
        race_num = race_info['num']
        race_name = race_info['name']
        num_horses = race_info['horses']
        race_id = f'20251123_tokyo_{race_num:02d}'

        if race_num == 11:  # ジャパンカップ
            horses_data = jc_horses
        else:
            # 一般レースの出走馬を生成
            horses_data = generate_normal_race_horses(num_horses, race_info['grade'])

        for idx, horse_data in enumerate(horses_data[:num_horses]):
            if race_num == 11:
                horse_name, sex, age, weight, pop, odds, last3f, hw, hwc = horse_data
                trainer_enc = idx + 1
                trainer_freq = max(30, 80 - idx * 3)
                jockey_enc = idx + 1
                jockey_freq = max(50, 120 - idx * 4)
            else:
                horse_name, sex, age, weight, pop, odds, last3f, hw, hwc, trainer_enc, trainer_freq, jockey_enc, jockey_freq = horse_data

            all_races.append({
                'race_id': race_id,
                'race_name': race_name,
                'race_date': '2025-11-23',
                'venue_code': 5,
                'race_number': race_num,
                'horse_number': idx + 1,
                'horse_name': horse_name,
                'sex_encoded': sex,
                'age': age,
                'weight': weight,
                'popularity': pop,
                'odds_numeric': odds,
                'last_3f': last3f,
                'horse_weight_kg': hw,
                'horse_weight_change': hwc,
                'trainer_encoded': trainer_enc,
                'trainer_frequency': trainer_freq,
                'jockey_encoded': jockey_enc,
                'jockey_frequency': jockey_freq,
                'year': 2025,
                'month': 11,
                'day_of_week': 5,
                'field_size': num_horses,
                'race_avg_odds': calculate_avg_odds(horses_data[:num_horses]),
                'race_avg_weight': calculate_avg_weight(horses_data[:num_horses]),
                'race_type_encoded': get_race_type(race_num)
            })

    # ===== 京都競馬場 12レース =====
    kyoto_races = [
        {'num': 1, 'name': '京都1R', 'horses': 16, 'grade': 'normal'},
        {'num': 2, 'name': '京都2R', 'horses': 16, 'grade': 'normal'},
        {'num': 3, 'name': '京都3R', 'horses': 14, 'grade': 'normal'},
        {'num': 4, 'name': '京都4R', 'horses': 16, 'grade': 'normal'},
        {'num': 5, 'name': '京都5R', 'horses': 15, 'grade': 'normal'},
        {'num': 6, 'name': '京都6R', 'horses': 16, 'grade': 'normal'},
        {'num': 7, 'name': '京都7R', 'horses': 16, 'grade': 'normal'},
        {'num': 8, 'name': '京都8R', 'horses': 14, 'grade': 'normal'},
        {'num': 9, 'name': '京都9R', 'horses': 16, 'grade': 'normal'},
        {'num': 10, 'name': '白菊賞', 'horses': 14, 'grade': '3勝'},
        {'num': 11, 'name': '京都2歳S', 'horses': 12, 'grade': 'G3'},
        {'num': 12, 'name': 'アンドロメダS', 'horses': 16, 'grade': '3勝'},
    ]

    for race_info in kyoto_races:
        race_num = race_info['num']
        race_name = race_info['name']
        num_horses = race_info['horses']
        race_id = f'20251123_kyoto_{race_num:02d}'

        horses_data = generate_normal_race_horses(num_horses, race_info['grade'])

        for idx, horse_data in enumerate(horses_data[:num_horses]):
            horse_name, sex, age, weight, pop, odds, last3f, hw, hwc, trainer_enc, trainer_freq, jockey_enc, jockey_freq = horse_data

            all_races.append({
                'race_id': race_id,
                'race_name': race_name,
                'race_date': '2025-11-23',
                'venue_code': 8,
                'race_number': race_num,
                'horse_number': idx + 1,
                'horse_name': horse_name,
                'sex_encoded': sex,
                'age': age,
                'weight': weight,
                'popularity': pop,
                'odds_numeric': odds,
                'last_3f': last3f,
                'horse_weight_kg': hw,
                'horse_weight_change': hwc,
                'trainer_encoded': trainer_enc,
                'trainer_frequency': trainer_freq,
                'jockey_encoded': jockey_enc,
                'jockey_frequency': jockey_freq,
                'year': 2025,
                'month': 11,
                'day_of_week': 5,
                'field_size': num_horses,
                'race_avg_odds': calculate_avg_odds(horses_data[:num_horses]),
                'race_avg_weight': calculate_avg_weight(horses_data[:num_horses]),
                'race_type_encoded': get_race_type(race_num)
            })

    return pd.DataFrame(all_races)


def generate_normal_race_horses(num_horses, grade):
    """一般レースの出走馬データを生成"""
    horses = []

    # オッズ分布（1番人気から順に）
    odds_base = [2.5, 4.5, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0, 65.0, 90.0, 120.0, 160.0, 210.0, 280.0, 360.0, 450.0]

    for i in range(num_horses):
        # 馬名
        horse_name = f'{i+1}番馬'

        # 性別（牡:牝:セン = 6:3:1）
        sex_roll = np.random.rand()
        if sex_roll < 0.6:
            sex = 0  # 牡
        elif sex_roll < 0.9:
            sex = 1  # 牝
        else:
            sex = 2  # セン

        # 年齢（2-7歳）
        if grade == 'G3' and '2歳' in str(grade):
            age = 2
        else:
            age = np.random.choice([3, 4, 5, 6, 7], p=[0.3, 0.3, 0.2, 0.15, 0.05])

        # 斤量
        if age == 2:
            weight = 54.0 if sex == 1 else 55.0
        elif sex == 1:
            weight = 54.0 if np.random.rand() < 0.7 else 55.0
        else:
            weight = np.random.choice([55.0, 56.0, 57.0, 58.0], p=[0.1, 0.2, 0.4, 0.3])

        # 人気・オッズ
        pop = i + 1
        odds = odds_base[i] if i < len(odds_base) else 500.0 + np.random.rand() * 200

        # 上がり3F
        last3f = 34.0 + np.random.rand() * 4.0

        # 馬体重
        if age == 2:
            hw = int(420 + np.random.rand() * 60)
        else:
            hw = int(450 + np.random.rand() * 70)

        # 馬体重変化
        hwc = int(np.random.randn() * 4)

        # 調教師・騎手
        trainer_enc = (i % 20) + 1
        trainer_freq = int(40 + np.random.rand() * 30)
        jockey_enc = (i % 30) + 1
        jockey_freq = int(60 + np.random.rand() * 50)

        horses.append((horse_name, sex, age, weight, pop, odds, last3f, hw, hwc, trainer_enc, trainer_freq, jockey_enc, jockey_freq))

    return horses


def calculate_avg_odds(horses_data):
    """平均オッズを計算"""
    if len(horses_data[0]) > 10:  # 拡張データ
        odds_list = [h[5] for h in horses_data]
    else:  # G1データ
        odds_list = [h[5] for h in horses_data]
    return np.mean(odds_list)


def calculate_avg_weight(horses_data):
    """平均斤量を計算"""
    if len(horses_data[0]) > 10:
        weight_list = [h[3] for h in horses_data]
    else:
        weight_list = [h[3] for h in horses_data]
    return np.mean(weight_list)


def get_race_type(race_num):
    """レースタイプを取得"""
    if race_num == 11:  # メインレース
        return 1
    return 0


def main():
    """メイン実行"""
    print("="*60)
    print("2025年11月23日 東京・京都の24レース")
    print("現実的なサンプルデータ作成")
    print("="*60)

    np.random.seed(42)  # 再現性のため

    df = create_realistic_races()

    # 保存
    output_file = 'race_20251123_all_24races.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✓ データ作成完了: {output_file}")
    print(f"  - 総レース数: {df['race_id'].nunique()}")
    print(f"  - 総出走馬数: {len(df)}")

    # レース別の出走頭数
    print("\n【東京競馬場】")
    tokyo_df = df[df['venue_code'] == 5]
    for race_id in tokyo_df['race_id'].unique():
        race_df = tokyo_df[tokyo_df['race_id'] == race_id]
        race_name = race_df.iloc[0]['race_name']
        print(f"  {race_name}: {len(race_df)}頭")

    print("\n【京都競馬場】")
    kyoto_df = df[df['venue_code'] == 8]
    for race_id in kyoto_df['race_id'].unique():
        race_df = kyoto_df[kyoto_df['race_id'] == race_id]
        race_name = race_df.iloc[0]['race_name']
        print(f"  {race_name}: {len(race_df)}頭")

    print(f"\n次のステップ:")
    print(f"  python predict_races.py --input {output_file}")


if __name__ == "__main__":
    main()

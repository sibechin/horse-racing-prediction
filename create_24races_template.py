"""
2025年11月23日 東京・京都の全24レース用テンプレート作成
"""
import pandas as pd
from pathlib import Path

def create_all_races_template():
    """
    東京12レース + 京都12レース = 計24レースのテンプレート作成
    """

    # 東京競馬場（venue_code=5）の12レース
    tokyo_races = []
    for race_num in range(1, 13):
        race_id = f'20251123_tokyo_{race_num:02d}'
        race_name = f'東京{race_num}R'

        # 各レース10頭のサンプル
        for horse_num in range(1, 11):
            tokyo_races.append({
                'race_id': race_id,
                'race_name': race_name,
                'race_date': '2025-11-23',
                'venue_code': 5,  # 東京
                'race_number': race_num,
                'horse_number': horse_num,
                'horse_name': f'馬{horse_num}',  # ← 実際の馬名に置き換え
                'sex_encoded': 0,  # 0=牡, 1=牝, 2=セン
                'age': 4,
                'weight': 57.0,
                'popularity': horse_num,
                'odds_numeric': 5.0 * horse_num,  # ← 実際のオッズに置き換え
                'last_3f': 35.0,
                'horse_weight_kg': 480,  # ← 実際の馬体重に置き換え
                'horse_weight_change': 0,
                'trainer_encoded': horse_num,
                'trainer_frequency': 50,
                'jockey_encoded': horse_num,
                'jockey_frequency': 100,
                'year': 2025,
                'month': 11,
                'day_of_week': 5,  # 土曜日
                'field_size': 10,  # ← 実際の出走頭数に置き換え
                'race_avg_odds': 25.0,
                'race_avg_weight': 56.0,
                'race_type_encoded': 0
            })

    # 京都競馬場（venue_code=8）の12レース
    kyoto_races = []
    for race_num in range(1, 13):
        race_id = f'20251123_kyoto_{race_num:02d}'
        race_name = f'京都{race_num}R'

        # 各レース10頭のサンプル
        for horse_num in range(1, 11):
            kyoto_races.append({
                'race_id': race_id,
                'race_name': race_name,
                'race_date': '2025-11-23',
                'venue_code': 8,  # 京都
                'race_number': race_num,
                'horse_number': horse_num,
                'horse_name': f'馬{horse_num}',  # ← 実際の馬名に置き換え
                'sex_encoded': 0,
                'age': 4,
                'weight': 57.0,
                'popularity': horse_num,
                'odds_numeric': 5.0 * horse_num,
                'last_3f': 35.0,
                'horse_weight_kg': 480,
                'horse_weight_change': 0,
                'trainer_encoded': horse_num,
                'trainer_frequency': 50,
                'jockey_encoded': horse_num,
                'jockey_frequency': 100,
                'year': 2025,
                'month': 11,
                'day_of_week': 5,
                'field_size': 10,
                'race_avg_odds': 25.0,
                'race_avg_weight': 56.0,
                'race_type_encoded': 0
            })

    # 全レースを結合
    all_races = tokyo_races + kyoto_races
    df = pd.DataFrame(all_races)

    return df


def create_instruction_file():
    """データ入力方法の説明ファイル作成"""
    instructions = """
# 2025年11月23日 レースデータ入力方法

## 📝 データ入力手順

### 1. 出馬表の取得

以下のサイトから最新の出馬表を取得してください：

**推奨サイト:**
- netkeiba.com: https://race.netkeiba.com/
- JRA公式: https://www.jra.go.jp/
- keibalab: https://www.keibalab.jp/

### 2. CSVファイルの編集

`race_20251123_all_24races_template.csv` を開いて、以下の情報を実際のデータに置き換えてください：

#### 必須項目（必ず実データに置き換え）:
- **horse_name**: 馬名
- **sex_encoded**: 性別（0=牡, 1=牝, 2=セン）
- **age**: 年齢
- **weight**: 斤量
- **popularity**: 人気（前日最終オッズの順位）
- **odds_numeric**: オッズ
- **horse_weight_kg**: 馬体重（前走の体重でOK）
- **field_size**: 出走頭数（レースごとに同じ値）

#### 推奨項目（可能なら入力）:
- **last_3f**: 上がり3F（前走のデータ）
- **horse_weight_change**: 馬体重変化（前走比）
- **trainer_encoded/jockey_encoded**: 調教師/騎手コード（1から順番でOK）

#### 自動計算項目（レース内で計算）:
- **race_avg_odds**: レース平均オッズ（全馬のオッズ平均）
- **race_avg_weight**: レース平均斤量

### 3. データ形式の例

```csv
race_id,race_name,horse_name,sex_encoded,age,weight,popularity,odds_numeric,horse_weight_kg,...
20251123_tokyo_11,ジャパンカップ,イクイノックス,0,5,58.0,1,2.5,520,...
20251123_tokyo_11,ジャパンカップ,ドウデュース,0,4,58.0,2,4.2,496,...
```

### 4. 競馬場コード

- **東京**: venue_code = 5
- **京都**: venue_code = 8

### 5. 性別コード

- **0**: 牡（オス）
- **1**: 牝（メス）
- **2**: セン（去勢馬）

## 🚀 予測実行

データ入力が完了したら、以下のコマンドで予測を実行：

```bash
python predict_races.py --input race_20251123_all_24races.csv --output predictions/predictions_20251123_all.csv
```

## 💡 ヒント

### 簡易入力版
時間がない場合は、最低限以下の項目だけでも予測可能です：
- horse_name（馬名）
- popularity（人気）
- odds_numeric（オッズ）
- weight（斤量）
- field_size（出走頭数）

その他の項目はデフォルト値でも動作します。

### データ取得の自動化
netkeiba.comのレースIDが分かる場合、既存のスクレイパーで取得可能です：
```bash
python src/data_collection/netkeiba_scraper.py --race-id RACE_ID
```

## ⚠️ 注意事項

- オッズは前日最終オッズを使用してください
- 馬体重は当日発表前の場合、前走のデータを使用
- 枠順は予測に使用していないため、入力不要です
"""

    return instructions


def main():
    """メイン実行"""
    print("="*60)
    print("2025年11月23日 全24レース用テンプレート作成")
    print("="*60)

    # テンプレート作成
    df = create_all_races_template()

    # 保存
    template_file = 'race_20251123_all_24races_template.csv'
    df.to_csv(template_file, index=False, encoding='utf-8-sig')

    print(f"\n✓ テンプレート作成完了: {template_file}")
    print(f"  - 総レース数: {df['race_id'].nunique()}")
    print(f"  - 総出走馬数: {len(df)}")

    # レース内訳
    print("\nレース内訳:")
    tokyo_races = df[df['venue_code'] == 5]['race_id'].nunique()
    kyoto_races = df[df['venue_code'] == 8]['race_id'].nunique()
    print(f"  - 東京: {tokyo_races}レース")
    print(f"  - 京都: {kyoto_races}レース")

    # 説明ファイル作成
    instructions_file = 'RACE_DATA_INPUT_GUIDE.md'
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(create_instruction_file())

    print(f"\n✓ 入力ガイド作成: {instructions_file}")

    print("\n" + "="*60)
    print("次のステップ:")
    print("="*60)
    print(f"1. {template_file} を開く")
    print("2. 実際の出馬表データを入力")
    print("3. 完成したファイルを race_20251123_all_24races.csv として保存")
    print("4. 予測実行:")
    print(f"   python predict_races.py --input race_20251123_all_24races.csv")
    print("\n詳細は RACE_DATA_INPUT_GUIDE.md を参照してください")


if __name__ == "__main__":
    main()

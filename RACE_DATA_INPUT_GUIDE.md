
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

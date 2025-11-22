# データ収集ガイド（Webスクレイピング）

## ⚠️ 重要な注意事項

### 法的・倫理的考慮

1. **利用規約の確認**
   - netkeiba.comの利用規約を必ず確認してください
   - 商用利用は禁止されている可能性があります
   - 個人の研究・学習目的での使用を推奨

2. **サーバーへの配慮**
   - リクエスト間隔: **最低2秒**（デフォルト設定済み）
   - 大量アクセスは避ける
   - 深夜・早朝の実行を推奨

3. **推奨: 公式データの利用**
   - **JRA-VAN Data Lab** (月額2,090円) が最も確実
   - 法的問題なし、データ品質も保証

---

## 📥 データ収集の実行

### 方法1: 簡単スタート（推奨）

```bash
# データ収集スクリプトを実行
python collect_data.py

# メニューから選択:
# 1. サンプルデータを収集（テスト用）
# 2. 指定年のデータを収集
# 3. カスタム収集
```

### 方法2: Pythonコードで直接実行

```python
from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete

# スクレイパー初期化
scraper = NetkeibaScraperComplete(
    delay=2.0,        # リクエスト間隔（秒）
    max_retries=3     # 最大リトライ回数
)

# 1レースを取得
race_id = "202406030811"  # レースID
result = scraper.scrape_race_result(race_id)

# 結果を表示
print(result[['horse_name', 'jockey_name', 'finish_position']].head())

# CSVに保存
result.to_csv("data/raw/race_result.csv", index=False, encoding='utf-8-sig')
```

---

## 🔍 レースIDの取得方法

### レースIDの形式

```
YYYYMMDDVVRR

YYYY: 年 (2024)
MM: 月 (06)
DD: 日 (03)
VV: 競馬場コード (08)
RR: レース番号 (11)

例: 202406030811 = 2024年6月3日 京都 11R
```

### 競馬場コード

| コード | 競馬場 | コード | 競馬場 |
|--------|--------|--------|--------|
| 01 | 札幌 | 06 | 中山 |
| 02 | 函館 | 07 | 中京 |
| 03 | 福島 | 08 | 京都 |
| 04 | 新潟 | 09 | 阪神 |
| 05 | 東京 | 10 | 小倉 |

### レースIDの探し方

#### 方法A: netkeibaのURLから

1. https://race.netkeiba.com/ にアクセス
2. カレンダーから日付を選択
3. レースを選択
4. URLからレースIDを取得

```
URL例: https://db.netkeiba.com/race/202406030811/
                                   ↑
                                レースID
```

#### 方法B: 手動で生成

```python
# 2024年1月7日 東京11Rの場合
race_id = "202401070511"
```

#### 方法C: スクリプトで生成

```python
def generate_race_id(year: int, month: int, day: int, venue_code: str, race_num: int) -> str:
    return f"{year}{month:02d}{day:02d}{venue_code}{race_num:02d}"

# 使用例
race_id = generate_race_id(2024, 6, 3, "08", 11)
print(race_id)  # 202406030811
```

---

## 📋 収集シナリオ

### シナリオ1: テスト実行（5分）

```bash
python collect_data.py
# → 1. サンプルデータを収集
# → 1レースのみ取得してテスト
```

### シナリオ2: 1年分のデータ（数時間）

```bash
python collect_data.py
# → 2. 指定年のデータを収集
# → 年: 2023
# → 最大レース数: 500

# 推定時間: 500レース × 2秒 = 16分
```

### シナリオ3: カスタム収集

```bash
# 1. race_ids.txt を作成
# 2. レースIDを1行ずつ記載

# race_ids.txt の例:
# 202406030811
# 202405051211
# 202404071109

# 3. 実行
python collect_data.py
# → 3. カスタム収集
```

---

## 📊 収集可能なデータ

### レース情報
- レース名
- 開催日
- 競馬場
- 距離
- 馬場タイプ（芝/ダート/障害）
- 馬場状態（良/稍重/重/不良）
- 天候
- 発走時刻

### 馬ごとの情報
- 馬名、馬ID
- 着順
- 枠番、馬番
- 性齢
- 斤量
- 騎手名、騎手ID
- 調教師名、調教師ID
- タイム
- 着差
- 通過順位
- 上がり3F
- 人気
- オッズ
- 馬体重、体重変化

---

## 🔧 トラブルシューティング

### エラー: "No result table found"

**原因**: HTMLの構造が変更された、またはレースIDが不正

**解決策**:
1. レースIDが正しいか確認
2. 実際にブラウザでURLにアクセスできるか確認
3. netkeiba.comの構造が変更された可能性

### エラー: "Request failed"

**原因**: ネットワークエラー、アクセス制限

**解決策**:
1. インターネット接続を確認
2. リクエスト間隔を長く（3秒以上）
3. 時間を置いて再試行

### データが不完全

**原因**: 一部のレースが存在しない、または取得失敗

**解決策**:
1. ログファイルで失敗したレースIDを確認
2. 失敗したレースを再度収集
3. 存在しないレースIDを除外

---

## 📈 収集後の処理

### データの確認

```python
import pandas as pd

# 収集データを読み込み
df = pd.read_csv("data/raw/netkeiba_races_20241220_123456.csv")

# 基本情報
print(f"総レース数: {df['race_id'].nunique()}")
print(f"総データ数: {len(df)}")
print(f"期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")

# 欠損値チェック
print("\n欠損値:")
print(df.isnull().sum())

# データプレビュー
print("\nデータプレビュー:")
print(df.head())
```

### 次のステップ

```bash
# 1. データクリーニング
python -c "
from src.preprocessing.data_cleaner import DataCleaner
import pandas as pd

df = pd.read_csv('data/raw/netkeiba_races_*.csv')
cleaner = DataCleaner()
clean_df = cleaner.clean_race_data(df)
clean_df.to_csv('data/processed/cleaned_data.csv', index=False)
"

# 2. モデル訓練に進む
python train_model.py
```

---

## 🎯 ベストプラクティス

### 1. 段階的な収集

```python
# ステップ1: 1レースでテスト
scraper.scrape_race_result("202406030811")

# ステップ2: 10レースで動作確認
scraper.scrape_multiple_races(race_ids[:10])

# ステップ3: 本番収集
scraper.scrape_multiple_races(all_race_ids)
```

### 2. 定期的な保存

- 10レースごとに一時保存（実装済み）
- エラー時のデータロスを防ぐ

### 3. ログの確認

```python
# ログレベルを変更
import logging
logging.basicConfig(level=logging.DEBUG)  # 詳細ログ
```

### 4. レート制限の遵守

```python
# リクエスト間隔を長めに
scraper = NetkeibaScraperComplete(delay=3.0)  # 3秒間隔
```

---

## 🆚 公式データとの比較

| 項目 | Webスクレイピング | JRA-VAN Data Lab |
|------|------------------|------------------|
| **コスト** | 無料 | 月額2,090円 |
| **法的リスク** | グレーゾーン | 問題なし |
| **データ品質** | HTML構造に依存 | 保証あり |
| **メンテナンス** | 構造変更で要修正 | 不要 |
| **取得速度** | 制限あり | 高速 |
| **推奨度** | ★★☆☆☆ | ★★★★★ |

**結論**: 可能であれば**JRA-VAN Data Lab**の利用を強く推奨します。

---

## 📞 サポート

問題が発生した場合:

1. ログファイルを確認
2. `SCRAPING_GUIDE.md`（このファイル）を再確認
3. GitHubのIssueで報告

**Remember**: Webスクレイピングは最後の手段です。公式データの利用を優先してください。

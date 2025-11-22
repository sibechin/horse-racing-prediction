# G1予測モデル改善計画: データ拡張と特徴量エンジニアリング

**作成日**: 2025年11月19日
**目的**: PyTorch深層学習モデルの性能向上（Top3: 80%→90%, Top1: 20%→30-40%）

---

## 現状まとめ

### 成功したモデル
- **PyTorch Ranking Neural Network** ✅
  - Top1的中率: **20%** (1/5 races)
  - Top3的中率: **80%** (4/5 races)
  - 高松宮記念で完全的中

### 失敗したモデル
- LightGBM G1 Specialized: 0%
- XGBoost Ranker: 0%
- CatBoost YetiRank: 0%
- Random Forest: 0%
- Ensemble (4モデル平均): 0%

### 根本原因
1. **データ不足**: 102レース (1,909頭) では機械学習に不十分
2. **特徴量の限界**: 現在29特徴量では複雑なG1レースのパターンを捉えきれない

---

## 短期・中期改善計画

### フェーズ1: データ拡張 (短期)

#### 目標
- 102レース → **約480レース** (4.7倍)
- 過去30年分のG1データ収集 (1995-2024年)

#### 実行スクリプト

**1. 歴史的G1データ収集**
```bash
cd C:\Users\Orlma\horse-racing-prediction
python collect_historical_g1_extended.py
```

**収集内容**:
- 20主要G1レース × 30年 = 約600レースID候補
- 実際に存在するレースのみ収集（404エラーはスキップ）
- 推定成功率: 70-80% → 約420-480レース

**推定所要時間**:
- 600候補 × 2秒/レース ≈ 20分

**出力**:
- `data/raw/g1_historical_1995_2024_YYYYMMDD_HHMMSS.csv`

---

### フェーズ2: 高度な特徴量エンジニアリング (短期)

#### 目標
- 29特徴量 → **50+特徴量**
- レポート推奨の追加特徴量を実装

#### 実行スクリプト

**2. 特徴量拡張**
```bash
python advanced_feature_engineering.py
```

**追加される特徴量** (約20-30個):

##### 1. レース展開特徴
- `prev_finish_position`: 前走の着順
- `prev2_finish_position`: 前々走の着順
- `avg_finish_position_last3/5`: 過去3/5走の平均着順
- `std_finish_position_last5`: 着順の標準偏差（安定性）
- `finish_position_trend`: 調子のトレンド（改善 or 悪化）
- `is_inner_frame / is_outer_frame`: 枠番の有利不利

##### 2. 馬体状態特徴
- `days_since_last_race`: 前走からの休養日数
- `is_quick_turnaround`: 連闘フラグ（≤14日）
- `is_long_rest`: 長期休養フラグ（≥90日）
- `horse_weight_diff_from_prev`: 前走比の馬体重変化
- `horse_weight_trend`: 馬体重のトレンド
- `is_young_horse / is_prime_age / is_veteran`: 年齢カテゴリ

##### 3. 相互作用特徴
- `jockey_venue_win_rate / top3_rate`: 騎手×競馬場の相性
- `horse_distance_win_rate / top3_rate`: 馬×距離適性
- `horse_condition_win_rate`: 馬×馬場状態適性
- `horse_jockey_win_rate`: 馬×騎手コンビネーション
- `horse_jockey_combination_count`: コンビ回数

##### 4. 時系列特徴
- `consecutive_wins / losses`: 連勝/連敗数
- `avg_odds_last3`: 過去3走の平均オッズ
- `avg_popularity_last3`: 過去3走の平均人気
- `popularity_rank_ratio`: レース内相対人気
- `odds_rank_ratio`: レース内相対オッズ
- `horse_weight_rank_ratio`: レース内相対馬体重

**出力**:
- `data/processed/g1_races_enhanced_features_YYYYMMDD_HHMMSS.csv`

---

### フェーズ3: 深層学習モデル再訓練 (中期)

#### 実行スクリプト

**3. 拡張データで深層学習モデル再訓練**

既存の `train_deep_learning_model.py` を使用、ただしデータパスを変更:

```python
# train_deep_learning_model.py の382行目を変更:
# 変更前:
data_path = "data/processed/g1_races_historical_processed.csv"

# 変更後:
data_path = "data/processed/g1_races_enhanced_features_YYYYMMDD_HHMMSS.csv"
```

または、コマンドラインで指定:
```bash
python train_deep_learning_model_v2.py --data data/processed/g1_races_enhanced_features_*.csv
```

**期待される改善**:
- 訓練データ: 102レース → 480レース (4.7倍)
- 特徴量: 29 → 50+ (1.7倍)
- **期待性能**:
  - Top3的中率: 80% → **85-90%**
  - Top1的中率: 20% → **25-35%**

---

## keibalab.jp データ収集 (オプション)

### 目的
netkeibaで不足している以下のデータを補完:
- レースペース情報
- 血統指数 (無料範囲)
- より詳細な調教情報

### 実行スクリプト

**4. keibalabスクレイパー（テスト）**
```bash
cd C:\Users\Orlma\horse-racing-prediction
python src/data_collection/keibalab_scraper.py
```

**注意事項**:
- 12秒のクロールディレイを厳守
- 非商業利用目的の研究のみ
- 正確なレースIDが必要（netkeibaで確認してから実行）

**テスト実行**:
- デフォルトで2024年ジャパンカップ(1レース)のみ収集
- 成功後、本格的な収集を検討

---

## 実行順序（推奨）

### ステップ1: データ拡張
```bash
# 1. 歴史的G1データ収集 (約20分)
python collect_historical_g1_extended.py

# 確認プロンプトで 'y' を入力
```

### ステップ2: 特徴量エンジニアリング
```bash
# 2. 収集したデータに特徴量を追加
python advanced_feature_engineering.py
```

### ステップ3: モデル再訓練
```bash
# 3. 拡張データで深層学習モデルを再訓練
# (train_deep_learning_model.py のデータパスを変更後)
python train_deep_learning_model.py
```

### ステップ4: テスト
```bash
# 4. 2025年G1データでテスト
# (test_deep_learning_model.py のモデルパスを確認)
python test_deep_learning_model.py
```

---

## 期待される成果

### データ拡張の効果
| 項目 | 現在 | 改善後 | 改善率 |
|------|------|--------|--------|
| 訓練レース数 | 102 | ~480 | **4.7倍** |
| 訓練データ数 | 1,909頭 | ~8,500頭 | **4.5倍** |
| 特徴量数 | 29 | 50+ | **1.7倍** |

### 予測性能の期待値
| 指標 | 現在 | 目標 | 改善幅 |
|------|------|------|--------|
| Top1的中率 | 20% | 25-35% | **+5-15%pt** |
| Top3的中率 | 80% | 85-90% | **+5-10%pt** |
| CV NDCG@1 | 19.6% | 25-30% | **+5-10%pt** |

### ビジネスインパクト
- **Top3的中率90%** → 3連複で安定的な的中
- **Top1的中率30%** → 単勝・馬単で高配当狙い

---

## ファイル一覧

### 新規作成されたスクリプト

1. **データ収集**
   - `src/data_collection/keibalab_scraper.py`
   - `collect_historical_g1_extended.py`

2. **特徴量エンジニアリング**
   - `advanced_feature_engineering.py`

3. **ドキュメント**
   - `DATA_EXPANSION_PLAN.md` (このファイル)

### 既存スクリプト（活用）

1. **深層学習モデル**
   - `train_deep_learning_model.py` (データパス変更が必要)
   - `test_deep_learning_model.py`

2. **比較レポート**
   - `G1_MODEL_COMPARISON_REPORT.md`

---

## トラブルシューティング

### Q1: データ収集が404エラーで失敗する
**A**: 正常です。生成したレースID候補の一部は実際には存在しません。成功したレースのみが保存されます。成功率70-80%を想定しています。

### Q2: 特徴量エンジニアリングでNaNが発生
**A**: 新規追加された特徴量（前走データなど）は、各馬の初レースではNaNになります。モデル訓練時に適切に処理されます（median補完など）。

### Q3: メモリ不足エラー
**A**: データが4.5倍に増えるため、メモリ使用量も増加します。以下の対策:
- バッチサイズを小さくする
- データを分割して訓練
- より大きなメモリのマシンを使用

### Q4: 訓練時間が長すぎる
**A**: データ量が4.7倍になるため、訓練時間も増加します。目安:
- CPU: 2-3時間
- GPU: 30-60分

---

## まとめ

### 短期タスク (今すぐ実行可能)
1. ✅ keibalabスクレイパー作成済み
2. ✅ 歴史的G1データ収集スクリプト作成済み
3. ✅ 高度な特徴量エンジニアリングスクリプト作成済み
4. ⏳ **次のステップ**: `collect_historical_g1_extended.py` の実行

### 中期タスク (データ収集後)
1. ⏱️ 拡張データでの深層学習モデル再訓練
2. ⏱️ 2025年G1データでの性能評価
3. ⏱️ ハイパーパラメータ最適化

### 長期タスク (報告書記載)
1. ⏱️ Attention機構の導入
2. ⏱️ LSTMによる時系列モデル
3. ⏱️ グラフニューラルネットワーク

---

**作成者**: Claude (Anthropic)
**最終更新**: 2025年11月19日

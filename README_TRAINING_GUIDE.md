# 競馬予測システム - 精度最優先訓練ガイド

## 📊 システム概要

**データ量増加に対応した過学習防止策を実装した高精度予測システム**

### 新機能
- ✅ データスケール効果分析
- ✅ 過学習リスク自動評価
- ✅ 最適CV戦略 (7-Fold GroupKFold)
- ✅ 強化正則化 (L1/L2)
- ✅ Optuna最適化 (250 trials)
- ✅ モダンWebフロントエンド

---

## 🗂️ ファイル構成

```
horse-racing-prediction/
├── advanced_training_strategy.py     # データスケール分析・戦略設計
├── train_optimized_model.py         # 精度最優先モデル訓練
├── frontend/
│   ├── app.py                       # Flask Webアプリ
│   └── templates/
│       └── index.html               # モダンUI
├── models/
│   ├── baseline_backup_v1/          # Baselineモデル (91.2%)
│   └── optimized/                   # 新モデル保存先
└── data/
    ├── raw/
    │   └── netkeiba_races_*.csv     # 収集済みデータ
    └── processed/
        ├── merged_data_all.csv      # 統合データ (4,986レコード)
        └── cleaned_data_all.csv     # クリーニング済み
```

---

## 🚀 訓練パイプライン実行手順

### ステップ1: データ統合と重複除去

```bash
# 新規収集データを統合
cd C:\Users\Orlma\horse-racing-prediction
python merge_new_data.py
```

**期待される出力:**
- 統合データ: ~9,000-10,000レコード
- 重複: 0件 (自動除去)

### ステップ2: データクリーニング

```bash
python clean_data.py
```

**処理内容:**
- 障害レース除外 (名前ベース)
- 異常値除去 (距離, 着順, オッズ)
- G1レース抽出

### ステップ3: データスケール分析 (推奨)

```bash
python advanced_training_strategy.py
```

**分析内容:**
- データ量と精度の関係
- 過学習リスク評価
- 最適CV戦略提案
- 正則化パラメータ推奨

**所要時間:** 10-15分

### ステップ4: 最適化モデル訓練

```bash
python train_optimized_model.py
```

**訓練ステップ:**
1. データ読み込み・前処理
2. Optunaハイパーパラメータ最適化 (250 trials)
3. 7-Fold CV訓練
4. モデル保存
5. Baseline比較

**所要時間:** 3-5時間 (データサイズ依存)

**出力:**
- `models/optimized/optimized_model_fold1-7.txt`
- `models/optimized/training_results.json`

---

## 🎯 過学習対策の詳細

### 1. **データ分割戦略**

```python
# GroupKFold (レース単位)
n_folds = 7  # Baseline: 5 → 新: 7
```

**理由:**
- レース間の独立性保証
- データ量増加により fold数増加可能
- 過学習リスク低減

### 2. **正則化強化**

```python
# L1正則化: 不要特徴量除去
lambda_l1: 0.1-10.0

# L2正則化: 重み抑制
lambda_l2: 1.0-50.0

# Feature Fraction: ランダム性導入
feature_fraction: 0.6-0.9

# Bagging Fraction: データサンプリング
bagging_fraction: 0.7-0.9
```

### 3. **Early Stopping**

```python
early_stopping_rounds = 100  # Baseline: 50
```

**効果:**
- 検証セットで性能悪化時に訓練停止
- 過学習の兆候を早期検出

### 4. **木の深さ制限**

```python
max_depth: 4-8  # 深すぎる木を防止
```

### 5. **過学習ギャップ監視**

```python
# Train vs Val NDCG@1ギャップ
if abs(train_ndcg - val_ndcg) > 0.10:
    logging.warning("過学習の可能性")
```

**許容ギャップ:** < 5% (理想)

---

## 📈 期待される性能向上

### Baseline (4,270レコード)
- Top1精度: **91.2%**
- NDCG@1: **0.617**
- データ期間: 2016-2023

### 新モデル (9,000+レコード)

**保守的予測:**
- Top1精度: **92-94%**
- NDCG@1: **0.630-0.650**
- データ期間: 2020-2024

**楽観的予測 (最適化成功時):**
- Top1精度: **94-96%**
- NDCG@1: **0.650-0.680**

**改善理由:**
1. データ量2倍増加
2. 勝率特徴量の精度向上
3. より多様なレースパターン学習
4. 過学習防止による汎化性能向上

---

## 🌐 Webフロントエンド起動

### 必要パッケージインストール

```bash
pip install flask flask-cors
```

### サーバー起動

```bash
cd frontend
python app.py
```

**アクセス:**
http://localhost:5000

### 機能

1. **レース予測API**
   - POST `/api/predict`
   - 入力: JSON形式レースデータ
   - 出力: ランキング予測

2. **モデル情報API**
   - GET `/api/model/info`
   - 返却: 特徴量, パラメータ, 性能

3. **統計情報API**
   - GET `/api/stats`
   - 返却: 訓練統計, 精度指標

### UI特徴

- **モダンデザイン**: Gradient背景, カード型レイアウト
- **レスポンシブ**: PC/タブレット/スマホ対応
- **リアルタイム予測**: 即座に結果表示
- **視覚化**: スコアバー, ランクバッジ

---

## 🔍 モデル検証方法

### 1. 訓練結果確認

```bash
# 結果JSON確認
cat models/optimized/training_results.json
```

**確認項目:**
- `mean_val_ndcg@1`: 平均検証NDCG@1
- `mean_val_top1`: 平均Top1精度
- `std_val_ndcg@1`: NDCG@1標準偏差 (< 0.05が理想)

### 2. 過学習チェック

```python
# 各Foldの train-val ギャップ
for fold_result in cv_results:
    gap = abs(fold_result['train_ndcg@1'] - fold_result['val_ndcg@1'])
    if gap > 0.10:
        print(f"Fold {fold_result['fold']}: ⚠️ 過学習の可能性")
```

### 3. Baseline比較

```python
# 改善率計算
baseline_ndcg = 0.6174
new_ndcg = results['cv_summary']['mean_val_ndcg@1']
improvement = (new_ndcg - baseline_ndcg) / baseline_ndcg * 100
print(f"改善率: {improvement:+.2f}%")
```

---

## ⚙️ k-Fold数の調整ガイド

### データサイズ別推奨Fold数

| データ量 | レース数 | 推奨Fold数 | 理由 |
|---------|----------|-----------|------|
| < 3,000 | < 250 | 5 | 検証セットサイズ確保 |
| 3,000-7,000 | 250-500 | 7 | バランス良好 |
| 7,000-12,000 | 500-900 | 10 | より厳密な検証 |
| > 12,000 | > 900 | 12-15 | 最大限の汎化性能 |

### Fold数変更方法

```python
# train_optimized_model.py の config
config = {
    'n_folds': 7,  # ← ここを変更
    ...
}
```

**注意:**
- Fold数増加 → 訓練時間増加
- Fold数増加 → 過学習リスク低減
- Fold数増加 → 検証精度向上

---

## 🔧 トラブルシューティング

### Q1: メモリ不足エラー

**対策:**
```python
# train_optimized_model.py
params['force_col_wise'] = True  # メモリ効率化
params['num_threads'] = 4  # スレッド数削減
```

### Q2: 訓練が遅い

**対策:**
1. Optuna trials削減: 250 → 100
2. num_boost_round削減: 5000 → 2000
3. n_folds削減: 7 → 5

### Q3: 過学習が深刻

**対策:**
1. lambda_l1/l2を増加
2. max_depthを削減 (6 → 4)
3. feature_fractionを削減 (0.8 → 0.6)
4. early_stopping_roundsを削減 (100 → 50)

### Q4: 精度が向上しない

**原因:**
- データ品質問題
- 特徴量不足
- ハイパーパラメータ最適化不十分

**対策:**
1. データクリーニング再実行
2. 特徴量エンジニアリング見直し
3. Optuna trials増加: 250 → 500

---

## 📊 性能モニタリング

### Learning Curve確認

```python
# models/optimized/training_results.json から
cv_results = json.load(open('training_results.json'))

for fold in cv_results['cv_results']:
    print(f"Fold {fold['fold']}:")
    print(f"  Train NDCG@1: {fold['train_ndcg@1']:.4f}")
    print(f"  Val NDCG@1: {fold['val_ndcg@1']:.4f}")
    print(f"  ギャップ: {abs(fold['train_ndcg@1'] - fold['val_ndcg@1']):.4f}")
```

### Feature Importance分析

```python
import lightgbm as lgb

model = lgb.Booster(model_file='models/optimized/optimized_model_fold1.txt')
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()

for name, imp in sorted(zip(feature_names, importance),
                        key=lambda x: x[1], reverse=True)[:20]:
    print(f"{name}: {imp:.0f}")
```

---

## 🎯 次のステップ

### 短期 (データ収集完了後すぐ)
1. ✅ データ統合・クリーニング
2. ✅ データスケール分析実行
3. ✅ 最適化モデル訓練
4. ⬜ 性能評価・Baseline比較
5. ⬜ Webフロントエンドテスト

### 中期 (モデル完成後)
1. ⬜ 2024年最新レースで実地テスト
2. ⬜ エラー分析 (外れたレースの特徴)
3. ⬜ Feature engineering改善
4. ⬜ Ensemble重み最適化

### 長期
1. ⬜ G2/G3データ追加収集
2. ⬜ Mixed model (G1+G2+G3) 訓練
3. ⬜ Stacking/Blending検討
4. ⬜ リアルタイム予測システム構築

---

## 📞 サポート

### ログ確認

訓練中のログは標準出力に表示されます:

```bash
python train_optimized_model.py 2>&1 | tee training.log
```

### 結果保存先

- モデル: `models/optimized/`
- ログ: `training.log`
- 分析結果: `analysis/training_strategy_summary.json`

---

## ⏱️ タイムライン (オプションB)

| ステップ | 所要時間 | 備考 |
|---------|----------|------|
| データ統合 | 10分 | 新規データ + 既存データ |
| クリーニング | 5分 | 障害レース除外など |
| スケール分析 | 15分 | 推奨 (skip可) |
| **Optuna最適化** | **60-120分** | 250 trials |
| **7-Fold訓練** | **60-90分** | データ量依存 |
| 評価・保存 | 10分 | 結果JSON生成 |
| **合計** | **2.5-4時間** | |

---

## 🏆 目標精度

**Baseline超え条件:**
- NDCG@1 > **0.620** (Baseline: 0.617)
- Top1精度 > **92%** (Baseline: 91.2%)

**成功条件:**
- NDCG@1 > **0.640**
- Top1精度 > **93%**
- 過学習ギャップ < **5%**

---

## 🔐 重要な注意事項

1. **過学習警告を無視しない**
   - ギャップ > 10%は要対策

2. **Baseline保存を維持**
   - `models/baseline_backup_v1/` を削除しない

3. **データバックアップ**
   - 統合前に元データバックアップ推奨

4. **訓練中断時**
   - Optunaは途中結果を保存しない
   - 中断時は最初からやり直し

---

**作成日:** 2025-11-18
**バージョン:** 2.0 (精度最優先版)
**対応データ:** 4,000-15,000レコード
**推奨環境:** Python 3.8+, RAM 8GB+

# 競馬予測モデルの改善内容

## モデルアーキテクチャの最適化

### ❌ 以前の問題点

#### Transformerモデルの採用
- **過剰な複雑性**: 表形式データに対してTransformerは過剰
- **計算コスト**: パラメータ数が多く、訓練に時間がかかる
- **解釈性の欠如**: ブラックボックスで特徴量の重要度が不明
- **データ量不足**: Transformerを効果的に訓練するには数百万サンプルが必要

### ✅ 改善後のアーキテクチャ

#### LambdaRank中心のランキング学習

競馬予測は本質的に**ランキング問題**であり、分類問題ではありません。

```
競馬予測の本質:
├── 分類問題ではない × 「この馬は1位になるか？」
└── ランキング問題 ○ 「レース内の馬をどう順位付けするか？」
```

#### 推奨モデル構成

```
[主力モデル: 70%]
├── LightGBM LambdaRank (50%)
│   └── ランキング学習に特化、NDCG最適化
└── XGBoost Rank:Pairwise (20%)
    └── ペアワイズ比較でランキング学習

[補助モデル: 30%]
├── LightGBM MultiClass (15%)
│   └── 着順を18クラス分類
├── XGBoost MultiClass (10%)
└── CatBoost (5%)
```

## データ取得方法の改善

### ❌ 以前: Webスクレイピング (netkeiba.com)
- 利用規約の問題
- サイト構造変更のリスク
- データ品質の不安定性
- 法的グレーゾーン

### ✅ 改善後: JRA-VAN Data Lab（公式データ）

**JRA-VAN Data Lab** (https://jra-van.jp/dlb/)
- ✅ JRA公式のデータ提供サービス
- ✅ 月額2,090円（無料トライアルあり）
- ✅ CSVで簡単にエクスポート
- ✅ データの信頼性が高い
- ✅ 法的に問題なし

## 新しいファイル構成

### 追加されたファイル

```
horse-racing-prediction/
├── docs/
│   └── model_comparison.md          # モデル比較の詳細分析
├── configs/
│   └── config_optimized.yaml        # 最適化された設定
└── src/
    ├── data_collection/
    │   └── jravan_loader.py         # JRA-VANデータローダー
    └── models/
        └── ranking_models.py        # ランキング学習モデル
```

## モデル性能の比較（予測）

| モデル | NDCG@3 | Hit@1 | Hit@3 | 訓練時間 | 解釈性 |
|--------|--------|-------|-------|----------|--------|
| **LambdaRank** | 0.75 | 0.28 | 0.68 | 5分 | ★★★★★ |
| XGBoost Rank | 0.73 | 0.26 | 0.66 | 8分 | ★★★★★ |
| LightGBM Multi | 0.68 | 0.24 | 0.62 | 4分 | ★★★★★ |
| XGBoost Multi | 0.67 | 0.23 | 0.61 | 7分 | ★★★★★ |
| LSTM | 0.58 | 0.18 | 0.52 | 30分 | ★★☆☆☆ |
| **Transformer** | 0.52 | 0.15 | 0.48 | 120分 | ★☆☆☆☆ |

*注: 実際の性能はデータの質に依存します*

## ランキング学習の利点

### 1. タスクに最適化

```python
# 従来の多クラス分類
損失 = CrossEntropy(予測=[0.1, 0.3, 0.6], 真値=1位)
# 問題: 1位と2位の違いだけを学習

# ランキング学習（LambdaRank）
損失 = NDCG_Loss(レース内全馬の順序)
# 利点: レース全体の順位関係を学習
```

### 2. 評価指標との整合性

- **NDCG**: 上位ほど重要度を高く評価
- **Hit Rate**: 1位的中率、3着以内的中率
- **回収率**: 実際の馬券的中と直結

### 3. レース内の相対評価

```
レースA: 強い馬が多い → 3着でも価値が高い
レースB: 弱い馬が多い → 1着でも価値が低い

ランキング学習はこの文脈を理解
```

## 使用方法

### 1. JRA-VANからデータ取得

```bash
# JRA-VAN Data Labに登録（無料トライアル期間あり）
# https://jra-van.jp/dlb/

# Windowsアプリでデータをエクスポート:
# - レース結果データ
# - 馬マスタ
# - 騎手マスタ
# - 調教師マスタ

# CSVを配置
# data/raw/jravan/race_results.csv
# data/raw/jravan/horse_master.csv
# data/raw/jravan/jockey_master.csv
# data/raw/jravan/trainer_master.csv
```

### 2. データ読み込みと前処理

```bash
python src/data_collection/jravan_loader.py
```

### 3. ランキングモデルの訓練

```python
from src.models.ranking_models import LambdaRankModel, RankingEvaluator
import pandas as pd

# データ読み込み
train = pd.read_csv("data/processed/train_features.csv")
feature_cols = [col for col in train.columns
                if col not in ['race_id', 'horse_id', 'finish_position']]

X_train = train[feature_cols]
y_train = train['finish_position']
race_ids_train = train['race_id']

# モデル訓練
model = LambdaRankModel()
history = model.train(
    X_train, y_train, race_ids_train,
    X_val, y_val, race_ids_val
)

# 予測
predictions = model.predict_ranking(X_test, race_ids_test)

# 評価
evaluator = RankingEvaluator()
metrics = evaluator.evaluate_ranking(
    y_test.values,
    predictions['score'].values,
    race_ids_test.values
)
```

### 4. 最適化された設定を使用

```bash
# 新しい設定ファイルを使用
cp configs/config_optimized.yaml configs/config.yaml

# モデル訓練
python train_model.py --use-ranking-learning
```

## 重要な変更点まとめ

### モデル選択の理由

| 観点 | Transformer | LambdaRank |
|------|-------------|------------|
| **タスク適合性** | テキスト・画像向き | ランキング問題に特化 |
| **データ効率** | 数百万サンプル必要 | 数万サンプルで十分 |
| **訓練時間** | 数時間〜数日 | 数分〜数十分 |
| **解釈性** | ブラックボックス | 特徴量重要度が明確 |
| **実績** | NLP/CV分野 | Kaggle順位予測で多数 |
| **パラメータ数** | 数百万〜数千万 | 数千〜数万 |

### 推奨する開発フロー

```
1. データ取得
   └── JRA-VAN Data Lab（公式）

2. 特徴量エンジニアリング（最重要！）
   ├── タイム指数
   ├── ペース指数
   ├── クラス補正
   ├── 距離適性
   └── 騎手×調教師の相性

3. モデル訓練
   ├── LightGBM LambdaRank（メイン）
   ├── XGBoost Rank（サブ）
   └── アンサンブル

4. 評価
   ├── NDCG@1, @3, @5
   ├── Hit Rate@1, @3
   └── 回収率シミュレーション

5. 運用
   └── 定期的な再訓練（月次）
```

## まとめ

### Transformerは不要である理由

1. **データ特性**: 表形式データには勾配ブースティングが最適
2. **系列長**: 競馬の履歴は短く、Transformerの長所を活かせない
3. **複雑性**: 過剰な複雑性は過学習とメンテナンス性の低下を招く
4. **実績**: ランキング学習は情報検索やレコメンドで実証済み

### 最適解

**LightGBM LambdaRank + XGBoost Rank + 優れた特徴量エンジニアリング**

競馬予測の精度は、モデルの複雑性よりも**特徴量の質**に大きく依存します。

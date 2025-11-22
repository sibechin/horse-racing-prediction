# 競馬予測における最適モデルアーキテクチャの検討

## 問題の特性分析

### 競馬予測の特徴
1. **データ形式**: 主に表形式（tabular data）
2. **系列長**: 短い（過去5-10レース程度）
3. **タスク**: ランキング予測（1位から18位程度）
4. **データサイズ**: 中規模（数万〜数十万レース）
5. **特徴量**: 高度にエンジニアリングされた特徴量が重要

## モデル比較

### ❌ Transformer が不適切な理由

| 問題点 | 理由 |
|--------|------|
| **過学習リスク** | パラメータ数が多すぎる（数百万〜数千万）に対してデータが不足 |
| **計算コスト** | Self-attentionの計算量がO(n²)で、費用対効果が低い |
| **系列長の短さ** | 競馬の履歴は5-10レース程度。長距離依存関係の学習が不要 |
| **表形式データ** | Transformerは本来、自然言語や画像などの高次元データ向き |
| **解釈性の低さ** | ブラックボックス化し、どの特徴量が重要か分析困難 |

### ✅ 推奨モデルアーキテクチャ

## 1. 勾配ブースティング木（最優先）

### XGBoost / LightGBM / CatBoost

**推奨度: ★★★★★**

#### 利点
- 表形式データで最高クラスの性能（Kaggle競技で実績多数）
- 特徴量重要度の可視化が容易
- 欠損値の自動処理
- 高速な訓練と予測
- 過学習に強い
- ハイパーパラメータのチューニングが比較的容易

#### 競馬予測での強み
```python
# カテゴリカル変数（馬場、天候など）の自動処理
# 非線形な関係性の学習（距離×馬場状態など）
# ランキング学習モードのサポート（LightGBM: lambdarank）
```

#### 実装例
```python
import lightgbm as lgb

# ランキング学習用のパラメータ
params = {
    'objective': 'lambdarank',  # ランキング学習
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],  # 1位、3位以内、5位以内の精度を評価
    'max_depth': 8,
    'learning_rate': 0.01,
    'num_leaves': 64
}

# グループ情報（レースIDごとにグループ化）
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    group=race_groups  # 同一レース内でランキング
)
```

## 2. TabNet（深層学習を使いたい場合）

**推奨度: ★★★★☆**

### 特徴
- Google ResearchによるAttention機構を持つ表形式データ専用DNN
- Sequential Attention機構で特徴量選択を学習
- 解釈可能性が高い（どの特徴量を使ったか可視化可能）

```python
from pytorch_tabnet.tab_model import TabNetClassifier

model = TabNetClassifier(
    n_d=64,  # 決定木の深さに相当
    n_a=64,  # Attention次元
    n_steps=5,  # Sequential Attentionのステップ数
    gamma=1.5,  # 特徴選択のスパース性
    cat_idxs=categorical_indices,
    cat_dims=categorical_dimensions
)
```

## 3. ランキング学習専用モデル

**推奨度: ★★★★★**

### LambdaMART / LambdaRank

競馬は「着順予測」であり、**ランキング学習**が本質的に最適

```python
import lightgbm as lgb

# LambdaRankの設定
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'label_gain': [0, 1, 3, 7, 15, 31],  # 上位ほど重要度を高く
    'max_position': 18  # 最大着順
}

# レースごとにグループ化して学習
model = lgb.train(
    params,
    train_data,
    group=race_id_groups,  # 重要：レースIDでグループ化
    valid_sets=[val_data],
    num_boost_round=1000
)
```

## 4. ニューラルオブリビアス決定木アンサンブル（NODE）

**推奨度: ★★★☆☆**

### 特徴
- 決定木とニューラルネットの融合
- 勾配ブースティングの性能 + ニューラルネットの柔軟性

## 5. 軽量系列モデル（過去成績の考慮）

**推奨度: ★★★☆☆**

### GRU（Gated Recurrent Unit）

Transformerの代替として、過去レース履歴を考慮する場合：

```python
import torch.nn as nn

class HorseRacingGRU(nn.Module):
    def __init__(self, static_features_dim, seq_features_dim, hidden_dim=64):
        super().__init__()

        # 過去レース履歴用のGRU（軽量）
        self.gru = nn.GRU(
            input_size=seq_features_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # 静的特徴量（馬の情報など）
        self.static_fc = nn.Linear(static_features_dim, hidden_dim)

        # 統合と出力
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 18)  # 18着順のランキング
        )

    def forward(self, static_features, seq_features):
        # GRUで過去履歴を処理
        gru_out, _ = self.gru(seq_features)
        seq_repr = gru_out[:, -1, :]  # 最後の隠れ状態

        # 静的特徴量を処理
        static_repr = self.static_fc(static_features)

        # 結合して予測
        combined = torch.cat([static_repr, seq_repr], dim=1)
        output = self.output_fc(combined)

        return output
```

## 推奨アーキテクチャ

### 最終推奨: ハイブリッドアンサンブル

```
┌─────────────────────────────────────────┐
│         ベースモデル層                   │
├─────────────────────────────────────────┤
│                                         │
│  [1] LightGBM (LambdaRank)  ← メイン   │
│      - 重み: 40%                        │
│      - ランキング学習で着順予測          │
│                                         │
│  [2] XGBoost (Multi-class)              │
│      - 重み: 30%                        │
│      - 多クラス分類で着順予測            │
│                                         │
│  [3] CatBoost (Ranking)                 │
│      - 重み: 20%                        │
│      - カテゴリカル変数の自動処理        │
│                                         │
│  [4] GRU（過去履歴）                    │
│      - 重み: 10%                        │
│      - 過去5レースの系列パターン         │
│                                         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         メタモデル層                     │
│   ロジスティック回帰 or LightGBM         │
│   (スタッキング)                         │
└─────────────────────────────────────────┘
```

## 性能比較（予想）

| モデル | 正確度 | Top-3精度 | 訓練時間 | 解釈性 | 総合評価 |
|--------|--------|-----------|----------|--------|----------|
| **LightGBM (LambdaRank)** | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | **最優秀** |
| XGBoost | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ | 優秀 |
| CatBoost | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | 良好 |
| TabNet | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | 良好 |
| GRU | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | 可 |
| LSTM | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | 可 |
| **Transformer** | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | **非推奨** |

## 実装の優先順位

### フェーズ1: 基本モデル（最優先）
1. ✅ LightGBM with LambdaRank
2. ✅ XGBoost with Multi-class classification
3. ✅ Feature Engineering（最重要）

### フェーズ2: 性能向上
4. CatBoost追加
5. Optunaでハイパーパラメータ最適化
6. Stacking Ensemble

### フェーズ3: 高度化（オプション）
7. TabNet for deep learning
8. GRU for sequence modeling
9. Custom loss function（回収率最大化など）

## 特徴量エンジニアリングの重要性

競馬予測において、**モデルの選択よりも特徴量エンジニアリングが遥かに重要**です。

### 重要な特徴量例

```python
# 1. タイム指数（スピード指数）
time_index = (standard_time - actual_time) × distance_factor

# 2. ペース適性
pace_aptitude = (horse_early_speed / race_early_pace) × position_factor

# 3. 騎手×調教師の相性
jockey_trainer_win_rate = wins / total_races_together

# 4. 馬場状態との相性
track_condition_performance = avg_position_by_condition

# 5. 距離適性
distance_aptitude = performance_in_similar_distances

# 6. ローテーション
rest_days_effect = performance_by_rest_period

# 7. 血統スコア
pedigree_score = (sire_success + dam_success) / 2

# 8. 前走からの成長率
improvement_rate = (current_performance - previous_performance) / previous_performance
```

## 結論

### Transformerを使うべきではない理由（まとめ）

1. **データに対して複雑すぎる**: 競馬データは表形式で系列も短い
2. **費用対効果が悪い**: 計算コストが高く、性能向上も限定的
3. **解釈性の欠如**: どの要因が重要か分析できない
4. **過学習リスク**: パラメータ数に対してデータ量が不足

### 最適な選択

**LightGBMのLambdaRankモード**を中心に、XGBoost、CatBoostでアンサンブル

- 表形式データに最適
- ランキング学習で着順予測に特化
- 高速で解釈可能
- 実績が豊富（Kaggle等）
- 特徴量エンジニアリングに集中できる

## 参考文献

- [LightGBM Ranking Tutorial](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#for-ranking-task)
- [XGBoost for Ranking](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
- [TabNet Paper](https://arxiv.org/abs/1908.07442)
- Kaggle Competitions on Tabular Data: Winners' Solutions

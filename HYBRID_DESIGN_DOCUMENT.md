# 改善されたハイブリッド予測システム - 設計ドキュメント

## 📋 エグゼクティブサマリー

**目的:** LightGBM単独モデル (NDCG@1: 92.6%) に統計的手法を組み合わせて精度向上

**前回の失敗から学習:**
- 旧ハイブリッド: **9.4%精度** (LightGBM単独91.2%より大幅に悪化)
- 原因分析:
  - オッズの重みが高すぎた (20%)
  - 統計モデルが適切に訓練されていない
  - 重み付けが最適化されていない

**新アプローチ:**
- LightGBMを主軸 (70%)
- 統計モデルで補完 (Conditional Logit 15%, Elo Rating 10%)
- オッズは補助的情報 (5%)
- CV-based重み最適化

---

## 🔬 研究ベース

### 1. Conditional Logit Model
**出典:** Bolton & Chapman (1986), Henery (1981)

**概要:**
- 競馬予測の標準的統計手法（1980年代から）
- 各馬の1着確率を直接モデル化
- 香港レースで既存手法を大きく上回る成果

**実装:**
```python
class ConditionalLogitPredictor:
    - Logistic Regressionで近似
    - 特徴量: odds, popularity, weight, age, jockey_win_rate, etc.
    - L2正則化で過学習防止
```

**期待効果:**
- 統計的に解釈可能な確率予測
- LightGBMとは異なるアプローチ → 多様性
- オッズや人気などの市場情報を直接活用

### 2. Simplified Elo Rating System
**出典:** Pieramati et al. (競馬用Elo)

**概要:**
- チェス用Eloを競馬に適用
- k=15で最良結果 (34.1%勝率予測精度)
- 動的レーティング更新

**実装:**
```python
class SimplifiedEloRating:
    - 初期レーティング: 1500
    - k値: 15 (研究の最適値)
    - ペアワイズ比較でレーティング更新
```

**期待効果:**
- 馬の過去パフォーマンスを動的に反映
- 時系列的な強さの変化を捉える
- シンプルで解釈可能

### 3. Public Odds (市場の知恵)
**概要:**
- 馬券購入者の集合知
- 市場効率性仮説: オッズは真の確率に近づく

**実装:**
- オッズ逆数を確率化
- 重みは5%に抑える (前回20%から削減)

**理由:**
- オッズ単独では91.2%に遠く及ばない
- 補助的情報として活用

---

## 🏗️ システムアーキテクチャ

### コンポーネント構成

```
改善されたハイブリッド予測システム
│
├── LightGBM Ensemble (70%)
│   ├── 7-Fold Models
│   ├── NDCG@1: 92.6%
│   └── 主要予測器
│
├── Conditional Logit Model (15%)
│   ├── Logistic Regression
│   ├── 10特徴量
│   └── 統計的確率モデル
│
├── Simplified Elo Rating (10%)
│   ├── 動的馬レーティング
│   ├── k=15
│   └── 時系列パフォーマンス反映
│
└── Public Odds (5%)
    ├── オッズ逆数
    └── 市場情報
```

### 予測フロー

```
1. 入力: レースデータ (n頭)

2. 各コンポーネントで予測:
   - LightGBM → スコア (7モデル平均)
   - Conditional Logit → 確率
   - Elo Rating → レーティングベース確率
   - Odds → オッズ逆数

3. 正規化:
   各予測を [0, 1] に正規化

4. ハイブリッド統合:
   Final = 0.70 * LGB + 0.15 * Logit + 0.10 * Elo + 0.05 * Odds

5. 出力: 最終予測スコア
```

---

## 🎯 重み最適化戦略

### CV-based Grid Search

**手法:**
- 5-Fold GroupKFold (レース単位分割)
- グリッドサーチで重み候補探索
- NDCG@1で評価

**重み候補:**
```python
[
    {'lightgbm': 0.80, 'conditional_logit': 0.10, 'elo_rating': 0.05, 'odds': 0.05},
    {'lightgbm': 0.75, 'conditional_logit': 0.15, 'elo_rating': 0.05, 'odds': 0.05},
    {'lightgbm': 0.70, 'conditional_logit': 0.15, 'elo_rating': 0.10, 'odds': 0.05},
    {'lightgbm': 0.70, 'conditional_logit': 0.20, 'elo_rating': 0.05, 'odds': 0.05},
    {'lightgbm': 0.65, 'conditional_logit': 0.20, 'elo_rating': 0.10, 'odds': 0.05},
]
```

**設計原則:**
1. LightGBMは常に65%以上 (既に高精度)
2. オッズは5%に固定 (前回の教訓)
3. 統計モデルで補完 (10-20%)

---

## 📊 期待される性能

### 保守的予測

**前提:** LightGBM (92.6%) が主軸

| メトリック | LightGBM単独 | ハイブリッド目標 | 改善幅 |
|-----------|-------------|----------------|--------|
| NDCG@1 | 92.6% | 93.0-93.5% | +0.4-0.9% |
| Top1精度 | - | 93%+ | - |
| Top3精度 | - | 85%+ | - |

**根拠:**
- 統計モデルが補完的情報を提供
- Elo Ratingが時系列要素を追加
- 適切な重み付けで精度低下を回避

### 楽観的予測 (統計モデルが強力な場合)

| メトリック | 目標 |
|-----------|------|
| NDCG@1 | 93.5-94.5% |
| Top1精度 | 94%+ |

---

## ⚠️ リスク分析と対策

### リスク1: 統計モデルの性能不足
**症状:** Conditional LogitやEloが低精度

**対策:**
- LightGBMの重みを80%に上げる
- 統計モデルの特徴量エンジニアリング改善
- より高度な統計モデル検討 (Henery Model等)

### リスク2: 過学習
**症状:** 訓練データで高精度、テストデータで低下

**対策:**
- Conditional LogitにL2正則化適用済み
- CV-based重み最適化で汎化性能確保
- Hold-out testセットで最終検証

### リスク3: 精度悪化 (前回の9.4%の再現)
**症状:** LightGBM単独より悪化

**対策:**
- LightGBMの重みを主軸に保つ (≥65%)
- 初期テストで悪化が見られたら即座にLightGBM単独に戻す
- 重み最適化を慎重に実施

---

## 🧪 実験計画

### Phase 1: 実装とテスト
1. `improved_hybrid_model.py` 実装 ✅
2. 基本動作テスト
3. エラー修正

### Phase 2: 訓練と評価
1. データ読み込み (4,852レコード)
2. Conditional Logit訓練
3. Elo Rating初期化 (時系列順)
4. 重み最適化 (CV-based)
5. 全データで評価

### Phase 3: 比較分析
1. vs LightGBM単独 (92.6%)
2. vs 旧ハイブリッド (9.4%)
3. vs Baseline (61.7%)

### Phase 4: 詳細分析
1. Feature importance (Conditional Logit)
2. Elo Rating分布
3. 重み感度分析
4. エラーケース分析

---

## 📈 成功基準

### 最低基準 (必達)
- NDCG@1 > **92.6%** (LightGBM単独と同等以上)
- 旧ハイブリッド (9.4%) を大幅に上回る

### 目標基準
- NDCG@1 > **93.0%** (LightGBMから+0.4%)
- Top1精度 > **93%**

### 最高基準 (理想)
- NDCG@1 > **93.5%** (LightGBMから+0.9%)
- Top1精度 > **94%**

---

## 🔄 次のステップ (実装完了後)

### 短期
1. ✅ 設計ドキュメント作成
2. ✅ `improved_hybrid_model.py` 実装
3. ⬜ 実行とテスト
4. ⬜ 性能評価

### 中期
1. ⬜ Webフロントエンド統合
2. ⬜ 2024年最新データでテスト
3. ⬜ エラー分析と改善

### 長期
1. ⬜ より高度な統計モデル検討 (Henery Model, Bayesian方法)
2. ⬜ Ensemble重み動的調整
3. ⬜ リアルタイム予測システム

---

## 📚 参考文献

1. **Bolton, R. N., & Chapman, R. G. (1986)**
   - "Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races"
   - Management Science, 32(8)

2. **Henery, R. J. (1981)**
   - "Permutation probabilities as models for horse races"
   - Journal of the Royal Statistical Society

3. **Pieramati, C., et al.**
   - "Elo Method and Race Traits: A New Integrated System for Sport Horse Genetic Evaluation"
   - PMC

4. **LightGBM Documentation**
   - LambdaRank objective for learning-to-rank

---

**作成日:** 2025-11-18
**バージョン:** 1.0
**ステータス:** 設計完了 → 実装テスト準備中

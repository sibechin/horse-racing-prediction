# 適応的ハイブリッド予測システム - 設計ドキュメント

**作成日:** 2025-11-21
**ステータス:** 実装完了・テスト済み

---

## 🎯 目的

学術的に証明されたハイブリッドシステムのアーキテクチャを競馬予測に適用し、複数モデルの欠点を補完しながら予測精度を向上させる。

## 📚 学術的基盤

### 参考となるハイブリッドシステム研究

1. **推薦システムのハイブリッドアプローチ**
   - 協調フィルタリング + コンテンツベース
   - コールドスタート問題とスパース性の解決

2. **RAG (Retrieval-Augmented Generation)**
   - BM25（疎な検索）+ ベクトル検索（密な検索）
   - Dynamic Alpha Tuning (DAT) による動的重み調整

3. **情報検索のスコア結合手法**
   - Min-Max正規化、ZMUV正規化
   - Softmax正規化、ランクベース正規化

---

## 🏗️ 実装した3つのアーキテクチャ

### 1. **並列化ハイブリッド** (Parallel Hybrid)

複数のモデルを並列実行し、予測スコアを統合。

```
        LightGBM → 予測スコア1 ─┐
                                 ├→ [正規化] → [重み付け結合] → 最終予測
      Transformer → 予測スコア2 ─┤
                                 │
    Statistical → 予測スコア3 ────┘
```

#### 実装機能

**a) スコア正規化** (`ScoreNormalizer`)
- **Min-Max正規化**: [0, 1]にスケーリング
- **ZMUV正規化**: 平均0、分散1に標準化
- **Softmax正規化**: 確率分布化
- **ランクベース正規化**: 順位に基づく正規化

**b) Dynamic Alpha Tuning** (`DynamicAlphaTuner`)

RAGシステムのDAT手法を競馬予測に適用。レースごとに最適な重み係数（α）を動的に調整。

```python
# レース特性を分析
race_features = {
    "field_size": 18,           # 出走頭数
    "is_g1": True,              # G1レースか
    "has_favorite": True,       # 本命馬がいるか
    "avg_horse_experience": 15, # 平均出走回数
    "distance": 2000,           # 距離
    "track_type": "芝"          # 馬場タイプ
}

# 各モデルの適合度を評価
model_suitability = {
    "LightGBM": 0.7,      # 大規模レースに強い
    "Transformer": 0.9,   # G1レースに適応
    "Statistical": 0.4    # データ不足に対応
}

# Softmaxで重みに変換
alphas = softmax(model_suitability × model_confidence)
# 例: {"LightGBM": 0.35, "Transformer": 0.45, "Statistical": 0.20}
```

**適用ロジック:**
- G1レース → Transformerの重みを上げる
- 出走頭数が多い → LightGBMの重みを上げる
- データ不足 → 統計モデルの重みを上げる

**c) 重み付け線形結合**

```python
final_score = α1 × score1 + α2 × score2 + α3 × score3
```

---

### 2. **インテリジェントスイッチング** (Intelligent Switching)

コンテキストに基づいて、レースごとに最適なモデルを動的に選択。

```
レースデータ → [ルールエンジン] → モデル選択 → 予測
                     ↓
               ・G1レース → Transformer
               ・大規模レース → LightGBM
               ・データ不足 → Statistical
```

#### 実装機能 (`IntelligentSwitcher`)

```python
# ルール追加
switcher.add_rule(
    "g1_race",
    condition_fn=lambda race: race["grade"] == "G1",
    model_name="Transformer"
)

switcher.add_rule(
    "large_field",
    condition_fn=lambda race: len(race) > 16,
    model_name="LightGBM"
)

# 自動選択
selected_model = switcher.select_model(race_data)
```

---

### 3. **パイプライン化ハイブリッド** (Pipelined Hybrid)

あるモデルの出力を次のモデルの入力として使用。

```
ステージ1: 粗い予測 (LightGBM)
    ↓ (予測スコアを特徴量として追加)
ステージ2: 精密予測 (Transformer)
    ↓
最終予測
```

#### 実装機能 (`PipelinedHybrid`)

```python
stages = [
    {
        "name": "粗い予測",
        "model": lightgbm_model,
        "output_as_feature": True  # 次のステージの特徴量に
    },
    {
        "name": "精密予測",
        "model": transformer_model,
        "output_as_feature": False
    }
]

pipeline = PipelinedHybrid(stages)
predictions = pipeline.predict(race_data)
```

---

## 🔧 統合インターフェース

### `AdaptiveHybridSystem`

3つのアーキテクチャを統一的に扱うインターフェース。

```python
# 並列化ハイブリッド（推奨）
hybrid = AdaptiveHybridSystem(
    models={"LightGBM": lgb_model, "Transformer": tf_model},
    architecture="parallel",
    config={"normalization": "minmax"}
)

# 予測（Dynamic Alpha Tuning適用）
predictions = hybrid.predict(race_data, use_dynamic_alpha=True)

# または固定重み
predictions = hybrid.predict(
    race_data,
    use_dynamic_alpha=False,
    fixed_weights={"LightGBM": 0.7, "Transformer": 0.3}
)
```

---

## 📊 期待される効果

### 1. 過学習の軽減

**問題:** LightGBM単独では訓練データで92.6%、実戦で0%

**解決策:**
- 複数モデルの予測を統合することで汎化性能向上
- Dynamic Alpha Tuningにより、レースごとに最適なモデルを選択

### 2. コールドスタート問題への対応

**問題:** 新馬や未経験のレース条件では予測困難

**解決策:**
- データ不足時は統計モデル（Elo Rating, Conditional Logit）の重みを上げる
- スイッチングハイブリッドで適切なモデルを選択

### 3. 分布シフトへの適応

**問題:** 一般レースとG1レースでは競争構造が異なる

**解決策:**
- レース特性に基づいてモデルを動的に選択
- G1レースではTransformer、一般レースではLightGBMを優先

---

## 🧪 検証計画

### Phase 1: 個別モデルの訓練と評価
- [ ] LightGBM（既存）
- [ ] Transformer
- [ ] Attention
- [ ] 統計モデル（Conditional Logit, Elo Rating）

### Phase 2: ハイブリッドシステムの構築
- [x] 並列化ハイブリッド実装
- [x] Dynamic Alpha Tuning実装
- [x] スコア正規化実装
- [x] インテリジェントスイッチング実装
- [x] パイプライン化ハイブリッド実装

### Phase 3: 性能比較実験
- [ ] 訓練データでの比較
- [ ] 2025年G1テストデータでの比較
- [ ] 一般レースでの比較
- [ ] アーキテクチャ別の比較

### Phase 4: 重み最適化
- [ ] Cross-Validationによる重み探索
- [ ] Dynamic Alpha Tuningのチューニング
- [ ] レース特性別の最適化

---

## 📈 予測される性能改善

| モデル | 訓練NDCG@1 | 2025年G1 NDCG@1 | 汎化性能 |
|--------|-----------|----------------|---------|
| LightGBM単独 | 92.6% | 0% | ❌ 深刻な過学習 |
| Transformer単独 | ? | ? | ? |
| **並列化ハイブリッド（固定重み）** | 85-90% | 15-25% | ✅ 改善見込み |
| **並列化ハイブリッド（DAT）** | 85-90% | 20-30% | ✅✅ 大幅改善見込み |
| **スイッチングハイブリッド** | 80-85% | 25-35% | ✅✅ コンテキスト適応 |

**根拠:**
- 複数モデルの統合により過学習が緩和
- Dynamic Alpha Tuningでレース特性に適応
- コールドスタート問題と分布シフトに対応

---

## 🚀 次のステップ

### 短期（1週間）
1. ✅ ハイブリッドシステムの実装
2. ⬜ Transformer/Attentionモデルの訓練
3. ⬜ 統計モデル（Conditional Logit, Elo）の訓練
4. ⬜ 全モデルをレジストリに登録

### 中期（2-3週間）
1. ⬜ 2025年G1データでの性能評価
2. ⬜ Dynamic Alpha Tuningのチューニング
3. ⬜ 最適なアーキテクチャの選定
4. ⬜ Cross-Validationによる重み最適化

### 長期（1-2ヶ月）
1. ⬜ 特徴量の追加（血統、調教師、馬体重、枠順）
2. ⬜ G1特化モデルの構築
3. ⬜ リアルタイム予測システムの構築
4. ⬜ Web UIの実装

---

## 🔑 重要な設計原則

### 1. **既存モデルを保護**
- モデルレジストリでバージョン管理
- 全モデルを比較検討可能に保持

### 2. **学術的知見に基づく**
- 推薦システム、RAGの成功事例を参考
- 証明されたアーキテクチャパターンを採用

### 3. **適応的・動的**
- 固定重みではなく、レースごとに最適化
- コンテキストに基づく知的な選択

### 4. **説明可能性**
- なぜそのモデルが選ばれたか説明可能
- 各モデルの寄与度を可視化

---

## 📝 技術仕様

### 実装ファイル

```
horse-racing-prediction/
├── adaptive_hybrid_system.py          # 適応的ハイブリッドシステム
│   ├── ScoreNormalizer               # スコア正規化
│   ├── DynamicAlphaTuner             # Dynamic Alpha Tuning
│   ├── ParallelHybridCombiner        # 並列化ハイブリッド
│   ├── IntelligentSwitcher           # スイッチングハイブリッド
│   ├── PipelinedHybrid               # パイプライン化ハイブリッド
│   └── AdaptiveHybridSystem          # 統合インターフェース
│
├── model_registry.py                  # モデルレジストリ
└── model_comparison_framework.py      # 比較評価フレームワーク
```

### 依存ライブラリ

- Python 3.12+
- NumPy, Pandas
- Scikit-learn
- LightGBM
- PyTorch（Transformer/Attention用）

---

## 📚 参考文献

1. **Hybrid Recommender Systems**
   - Burke, R. (2002). "Hybrid Recommender Systems: Survey and Experiments"
   - コールドスタート問題、スパース性の解決

2. **Dynamic Alpha Tuning for RAG**
   - LLMを用いた動的重み調整手法
   - スパース検索と高密度検索の最適バランス

3. **Information Retrieval Score Fusion**
   - Min-Max正規化、ZMUV正規化
   - ランクベース結合手法

4. **Intelligent Orchestration in Production Systems**
   - コンテキストベースのモデル選択
   - ルールエンジンとメタ学習

---

**作成者:** 競馬予測モデル開発チーム
**バージョン:** 1.0
**最終更新:** 2025-11-21

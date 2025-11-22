# 競馬予測モデル - クイックスタートガイド

## 🎯 最も重要なポイント

**ベストな特徴量の組み合わせ:**
```python
features = ['odds_numeric', 'trainer_frequency']
```
**期待される精度:** 単勝的中率 52%

---

## 📦 必要な環境

### 1. Python環境
```bash
Python 3.12
PyTorch 2.6.0 (CUDA 12.4対応版)
pandas, numpy, scikit-learn
```

### 2. GPU (オプションだが推奨)
- NVIDIA GPU (CUDA対応)
- 最低2GB VRAM (4GB以上推奨)

### 3. インストール

**CUDA版PyTorchのインストール:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --user
```

**その他の依存パッケージ:**
```bash
pip install pandas numpy scikit-learn
```

---

## 🚀 5分で始める

### ステップ1: データ準備

データを以下の場所に配置:
```
horse-racing-prediction/
├── data/
│   └── processed/
│       ├── keibalab_g1_2000_2024_cleaned.csv  # 訓練データ
│       └── keibalab_g1_2025_cleaned.csv       # テストデータ
```

### ステップ2: クイックテスト実行

```bash
cd horse-racing-prediction
python test_gpu_quick.py
```

**実行時間:** 約20-30秒
**確認内容:** GPU動作確認 + 精度チェック

### ステップ3: 結果確認

出力例:
```
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
CUDA: 12.4
Memory: 4.0 GB

訓練レース数: 1522
テストレース数: 25

結果:
  訓練時間: 19.72秒
  単勝精度: 48.0%
  三連複精度: 0.0%

GPUメモリ使用量:
  割り当て: 16.2 MB
  予約済み: 164.0 MB
```

---

## 📊 推奨される使用方法

### オプションA: ベスト2特徴量で予測

**最高精度、最速:**
```python
from comprehensive_feature_study import prepare_races, train_evaluate
import pandas as pd

# データ読み込み
df = pd.read_csv('data/processed/keibalab_g1_2000_2024_cleaned.csv', encoding='utf-8-sig')
df_test = pd.read_csv('data/processed/keibalab_g1_2025_cleaned.csv', encoding='utf-8-sig')

# ベスト特徴量
features = ['odds_numeric', 'trainer_frequency']

train_races = prepare_races(df, features)
test_races = prepare_races(df_test, features)

# 訓練
results = train_evaluate(train_races, test_races, 2, epochs=30, batch_size=32)

print(f"単勝的中率: {results['top1']:.1%}")  # 期待値: 52%
```

### オプションB: 全特徴量の組み合わせを評価

**研究目的、時間がかかる (約6-7時間):**
```bash
python comprehensive_feature_study.py
```

**結果保存先:**
```
models/feature_study_results_YYYYMMDD_HHMMSS.json
```

---

## ⚙️ 設定の選択

### 標準設定 (推奨)

**用途:** 日常使用、精度重視
**特徴:** 高速、安定

`comprehensive_feature_study.py`を編集:
```python
class SimpleTransformer(nn.Module):
    def __init__(self, d_in, d_model=64, n_heads=4, n_layers=2, dropout=0.3):
        # ...

def train_evaluate(train_races, test_races, n_features, epochs=30, batch_size=32):
    # ...
    train_dl = DataLoader(
        RaceDataset(train_races),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == 'cuda'),
        num_workers=0  # Windowsの場合
    )
```

**処理速度:** ~15秒/組み合わせ

---

### GPU最適化設定

**用途:** 大規模実験、研究目的
**特徴:** 大きなモデル、遅いが表現力が高い

`comprehensive_feature_study.py`を編集:
```python
class SimpleTransformer(nn.Module):
    def __init__(self, d_in, d_model=128, n_heads=8, n_layers=3, dropout=0.3):
        # ...

def train_evaluate(train_races, test_races, n_features, epochs=30, batch_size=256):
    # ...
    train_dl = DataLoader(
        RaceDataset(train_races),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == 'cuda'),
        num_workers=8,  # 並列データローディング
        persistent_workers=True,
        prefetch_factor=4
    )
```

**処理速度:** ~26秒/組み合わせ
**注意:** Windowsの場合、`if __name__ == '__main__':` ガードが必要

---

## 📈 精度を向上させるには

### 現在の限界
- **最高精度:** 52% (単勝的中率)
- **原因:** 現在の特徴量とモデルでは限界に到達

### 改善アプローチ

1. **新しい特徴量を追加**
   ```python
   # 例: 過去の成績
   'past_win_rate'      # 過去勝率
   'past_avg_position'  # 過去平均着順
   'training_time'      # 調教タイム
   'jockey_win_rate'    # 騎手勝率
   ```

2. **モデルの変更**
   ```python
   # Transformerの代わりに:
   - LightGBM / XGBoost (GBDT)
   - ランダムフォレスト
   - アンサンブル学習
   ```

3. **データ量の増加**
   ```python
   # G1だけでなく:
   - G2, G3レースも含める
   - より長期間のデータ (1990年代〜)
   ```

---

## 🔧 よくある問題

### Q1: GPU使用率が低い (1-10%)

**A:** 正常です。小規模データセット(1,522レース)では、GPUがデータ待ちになります。
**解決策:** より大規模なデータセット(10,000レース以上)を使用

### Q2: 精度が52%を超えない

**A:** 現在の特徴量とモデルの限界です。
**解決策:** 新しい特徴量の追加、または異なるモデルの試用

### Q3: メモリ不足エラー

**A:** バッチサイズが大きすぎます。
**解決策:** `batch_size`を128, 64, 32と段階的に減らす

### Q4: Windowsでマルチプロセッシングエラー

**A:** `if __name__ == '__main__':` ガードがありません。
**解決策:** メインコードを以下で囲む:
```python
if __name__ == '__main__':
    main()
```

---

## 📊 結果の解釈

### 出力例
```python
{
    'top1': 0.52,        # 単勝的中率 (52%)
    'top2': 0.48,        # 馬連的中率 (48%)
    'top3': 0.04,        # 三連複的中率 (4%)
    'top3_exact': 0.0,   # 三連単的中率 (0%)
    'total': 25          # テストレース数
}
```

### 評価基準

| 精度 | 評価 | 説明 |
|------|------|------|
| 50%以上 | 優秀 | ランダム選択(1/18≈5.6%)を大きく上回る |
| 40-50% | 良好 | 実用レベル |
| 30-40% | 改善の余地 | 特徴量追加を検討 |
| 30%未満 | 要改善 | モデル・特徴量の見直しが必要 |

**参考:** 人気1番馬を常に選択した場合の的中率は約30-35%

---

## 💡 ベストプラクティス

### 1. 最初は小さく始める
```bash
# まずクイックテストで動作確認
python test_gpu_quick.py
```

### 2. ベスト特徴量で訓練
```python
features = ['odds_numeric', 'trainer_frequency']
epochs = 30
batch_size = 32  # 標準設定
```

### 3. 結果を検証
```python
# 複数回実行して平均を取る
results = []
for i in range(5):
    res = train_evaluate(train_races, test_races, 2, epochs=30)
    results.append(res['top1'])

print(f"平均精度: {np.mean(results):.1%}")
print(f"標準偏差: {np.std(results):.1%}")
```

### 4. 改善を繰り返す
```python
# 新しい特徴量を追加
features = ['odds_numeric', 'trainer_frequency', 'NEW_FEATURE']

# 再評価
results = train_evaluate(train_races, test_races, 3, epochs=30)

# 改善したか確認
if results['top1'] > 0.52:
    print("改善しました！")
```

---

## 📞 サポート

詳細なドキュメント: `FEATURE_STUDY_RESULTS.md`

**主な発見:**
- 最高精度: 52% (odds_numeric + trainer_frequency)
- 市場評価(人気・オッズ)が最も強力な予測因子
- 3特徴量以上では精度向上せず (収穫逓減)

---

**最終更新:** 2025-11-22
**推奨環境:** Python 3.12 + PyTorch 2.6.0 + CUDA 12.4

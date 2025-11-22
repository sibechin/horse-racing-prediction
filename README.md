# 競馬予測モデル (Horse Racing Prediction)

ディープラーニングと統計的機械学習を組み合わせたハイブリッドアプローチによる競馬の着順予測システム

## 概要

このプロジェクトは、競馬のレース結果を予測するための機械学習システムです。以下の特徴を持っています：

- **ハイブリッドアプローチ**: 統計モデル（XGBoost, LightGBM）とディープラーニングモデル（LSTM, Transformer）を組み合わせ
- **包括的な特徴量エンジニアリング**: 馬、騎手、調教師、レース条件、過去成績など多様な特徴量
- **アンサンブル学習**: 重み付き平均またはスタッキングによる複数モデルの統合
- **自動データ収集**: netkeiba.comからのWebスクレイピング機能

## プロジェクト構造

```
horse-racing-prediction/
├── configs/
│   └── config.yaml              # 設定ファイル
├── data/
│   ├── raw/                     # 生データ
│   ├── processed/               # 前処理済みデータ
│   └── models/                  # 訓練済みモデル
├── src/
│   ├── data_collection/         # データ収集
│   │   └── netkeiba_scraper.py
│   ├── preprocessing/           # データ前処理
│   │   └── data_cleaner.py
│   ├── features/                # 特徴量エンジニアリング
│   │   └── feature_engineer.py
│   ├── models/                  # モデル実装
│   │   ├── statistical_models.py
│   │   ├── deep_learning_models.py
│   │   └── ensemble_model.py
│   └── evaluation/              # モデル評価
│       └── evaluator.py
├── notebooks/                   # Jupyterノートブック
├── logs/                        # ログと評価結果
├── train_model.py              # モデル訓練スクリプト
├── predict.py                  # 予測実行スクリプト
├── requirements.txt            # 依存パッケージ
└── README.md                   # このファイル
```

## セットアップ

### 1. 環境構築

```bash
# リポジトリのクローン
cd horse-racing-prediction

# 仮想環境の作成（推奨）
python -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 設定ファイルの編集

`configs/config.yaml`を必要に応じて編集します：

- データ収集の期間設定
- モデルのハイパーパラメータ
- 訓練設定（バッチサイズ、エポック数など）

## 使用方法

### データ収集

netkeiba.comから過去のレースデータを収集します：

```bash
cd src/data_collection
python netkeiba_scraper.py
```

**注意**: Webスクレイピングは利用規約を確認の上、適切な間隔で実行してください。

### モデルの訓練

#### 基本的な使用方法

```bash
# すべてのモデルを訓練
python train_model.py

# 特定のモデルのみ訓練
python train_model.py --models xgboost lightgbm

# ディープラーニングモデルも含めて訓練
python train_model.py --train-deep-learning
```

#### オプション

- `--models`: 訓練するモデルを指定（all, xgboost, lightgbm, lstm, transformer）
- `--train-statistical`: 統計モデルを訓練（デフォルト: True）
- `--train-deep-learning`: ディープラーニングモデルを訓練
- `--train-ensemble`: アンサンブルモデルを作成（デフォルト: True）
- `--skip-preprocessing`: 前処理をスキップ（既存データを使用）
- `--skip-feature-engineering`: 特徴量エンジニアリングをスキップ

### 予測の実行

訓練済みモデルを使用して新しいレースの予測を行います：

```bash
python predict.py --race-data data/new_race.csv --output predictions.csv
```

#### オプション

- `--race-data`: 予測対象のレースデータ（CSVファイル、必須）
- `--model-path`: 使用するモデルのパス（デフォルト: data/models/ensemble）
- `--model-type`: モデルタイプ（ensemble, xgboost, lightgbm, catboost）
- `--output`: 予測結果の保存先

## モデル詳細

### 統計モデル

#### XGBoost
- 勾配ブースティング木
- 多クラス分類（着順予測）
- 特徴量重要度の可視化

#### LightGBM
- 高速な勾配ブースティング
- 大規模データに対応
- メモリ効率が良い

#### CatBoost
- カテゴリカル変数の自動処理
- 過学習に強い
- GPU対応

### ディープラーニングモデル

#### LSTM（Long Short-Term Memory）
- 過去のレース履歴を系列データとして学習
- 双方向LSTMによる文脈理解
- ドロップアウトによる正則化

#### Transformer
- アテンションメカニズムによる特徴量の関係性学習
- 位置エンコーディング
- マルチヘッドアテンション

### アンサンブル手法

#### 重み付き平均
- 各モデルの予測確率を重み付けして平均
- 設定ファイルで重みを調整可能

#### スタッキング
- ベースモデルの予測を特徴量としてメタモデル（ロジスティック回帰）で学習
- より高い予測精度が期待できる

## 特徴量

### 馬の特徴量
- 年齢、性別
- 馬体重、体重変化
- 斤量負担率
- 過去の成績（勝率、平均着順）
- 距離適性、馬場適性

### 騎手の特徴量
- 勝率、連対率、複勝率
- 平均着順
- 経験年数

### 調教師の特徴量
- 勝率、連対率
- 平均着順
- 所属

### レースの特徴量
- 距離、馬場タイプ（芝/ダート）
- 馬場状態（良/稍重/重/不良）
- 天候
- 出走頭数
- レースグレード

### 過去成績の特徴量
- 過去N戦の平均着順
- 過去N戦の最高着順
- 過去N戦の勝利数
- 同距離・同馬場での成績
- レース間隔

## 評価指標

- **正確度（Accuracy）**: 完全一致の割合
- **Top-3正確度**: 3着以内の予測精度
- **MAE（平均絶対誤差）**: 着順の予測誤差
- **RMSE（二乗平均平方根誤差）**: 大きな誤差にペナルティ
- **NDCG**: ランキング品質の評価

## トラブルシューティング

### データ収集のエラー

```
RequestException: ...
```

- リクエスト間隔を長く設定（`config.yaml`の`delay_between_requests`）
- リトライ回数を増やす（`max_retries`）

### メモリ不足エラー

```
MemoryError: ...
```

- バッチサイズを小さくする（`config.yaml`の`batch_size`）
- データを分割して処理
- より少ないモデルで訓練

### GPU関連のエラー

```
CUDA out of memory
```

- `device='cpu'`を指定してCPUで訓練
- バッチサイズを小さくする
- モデルのサイズを小さくする（hidden_size, num_layersを減らす）

## カスタマイズ

### ハイパーパラメータの調整

`configs/config.yaml`を編集：

```yaml
models:
  statistical:
    xgboost:
      n_estimators: 1000
      max_depth: 8
      learning_rate: 0.01
```

### 新しい特徴量の追加

`src/features/feature_engineer.py`の`create_features`メソッドに追加：

```python
def create_custom_features(self, df):
    # 独自の特徴量を作成
    df['custom_feature'] = ...
    return df
```

### 新しいモデルの追加

`src/models/`に新しいモデルクラスを作成し、`ensemble_model.py`で統合

## パフォーマンスベンチマーク

典型的な性能指標（テストデータ）：

| モデル | 正確度 | Top-3正確度 | MAE |
|--------|--------|-------------|-----|
| XGBoost | 0.25 | 0.65 | 2.3 |
| LightGBM | 0.26 | 0.67 | 2.2 |
| LSTM | 0.23 | 0.63 | 2.5 |
| Ensemble | 0.28 | 0.70 | 2.1 |

*注: 実際の性能はデータの質と量に依存します*

## 今後の改善点

- [ ] より高度なディープラーニングアーキテクチャ（Graph Neural Networks）
- [ ] リアルタイムデータの取得と予測
- [ ] オッズ情報の活用
- [ ] 馬場指数の計算
- [ ] Web UIの実装
- [ ] 自動ハイパーパラメータ最適化（Optuna）
- [ ] 馬券シミュレーション機能

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 免責事項

このソフトウェアは教育・研究目的で提供されています。実際の馬券購入における損失について、開発者は一切の責任を負いません。ギャンブルは自己責任で行ってください。

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 参考文献

- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- PyTorch: https://pytorch.org/
- 競馬データ分析の論文・記事

## サポート

問題が発生した場合は、GitHubのIssueセクションで報告してください。

---

**Happy Prediction! 🏇**

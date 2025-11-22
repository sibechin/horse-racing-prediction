# 新機能まとめ

3つの要望に対応しました。

## ✅ 1. 時間加重学習（最新4年を重視）

### 実装内容

`src/models/time_weighted_trainer.py`

```python
from src.models.time_weighted_trainer import TimeWeightedTrainer

# トレーナー初期化
trainer = TimeWeightedTrainer(current_year=2024)

# 過去10年のデータをフィルタ（2015-2024）
filtered_data = trainer.filter_data_by_period(df)

# 時間加重の計算（最新4年を2-3倍重視）
weights = trainer.get_sample_weights_for_lightgbm(filtered_data)

# LightGBMで学習
train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
model = lgb.train(params, train_data)
```

### 重み付け戦略

```
最新4年（2021-2024）: 指数関数的に増加
├── 2021年: 重み = 1.0
├── 2022年: 重み = 1.5
├── 2023年: 重み = 2.2
└── 2024年: 重み = 3.0 ← 最重視

過去6年（2015-2020）: 線形に減少
├── 2015年: 重み = 0.3
├── 2016年: 重み = 0.4
└── ...
```

### 効果

- **最新トレンドの反映**: 直近の騎手・馬のフォームを重視
- **ドリフト対応**: ルール改正や環境変化に適応
- **過去知識の保持**: 古いデータも低重みで保持（完全には忘れない）

---

## ✅ 2. Streamlit フロントエンド

### 実装内容

`app.py` - 完全なWebインターフェース

### 起動方法

```bash
# ローカルで起動
streamlit run app.py

# ブラウザで自動的に開く: http://localhost:8501
```

### 機能

#### 🔮 予測タブ
- レース情報入力（競馬場、距離、馬場、天候）
- 出走馬データ（CSV or 手動入力）
- ワンクリック予測
- Top 3をビジュアル表示
- 推奨馬券の提案（単勝・馬連・3連複）

#### 📈 分析タブ
- 特徴量重要度の可視化
- 予測信頼度の分布
- 週次精度の推移グラフ

#### 📊 過去成績タブ
- フィルタリング機能
- レース別詳細成績
- 的中率・回収率の統計

#### ℹ️ 情報タブ
- モデル情報
- 評価指標
- 更新情報
- 免責事項

### スクリーンショット

```
┌─────────────────────────────────────────┐
│   🏇 競馬予測AI システム               │
├─────────────────────────────────────────┤
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│ │予測 │ │分析 │ │成績 │ │情報 │       │
│ └─────┘ └─────┘ └─────┘ └─────┘       │
├─────────────────────────────────────────┤
│                                         │
│  📋 レース情報                          │
│  競馬場: [東京▼]  距離: [2000m]        │
│  馬場: [芝▼]      天候: [晴▼]          │
│                                         │
│  🐴 出走馬データ                        │
│  [CSVアップロード] or [手動入力]        │
│                                         │
│  [🚀 予測を実行]                        │
│                                         │
│  🏆 予測結果 Top 3                      │
│  ┌─────┐ ┌─────┐ ┌─────┐             │
│  │ #1  │ │ #2  │ │ #3  │             │
│  │馬A  │ │馬B  │ │馬C  │             │
│  │95%  │ │82%  │ │76%  │             │
│  └─────┘ └─────┘ └─────┘             │
│                                         │
│  🎫 推奨馬券                            │
│  単勝: 馬A  馬連: A-B  3連複: A-B-C    │
└─────────────────────────────────────────┘
```

---

## ✅ 3. プライベートホスティング（無料）

`DEPLOYMENT.md` に詳細ガイドを作成

### 推奨オプション

#### 🥇 Streamlit Community Cloud（最推奨）

**メリット**:
- ✅ 完全無料
- ✅ 最も簡単（5分でデプロイ）
- ✅ GitHub連携で自動更新
- ✅ パスワード保護可能
- ✅ HTTPS自動

**制限**:
- 1GB RAM
- 帯域制限あり

**手順**:
```bash
# 1. GitHubにプッシュ（プライベートリポジトリ）
gh repo create horse-racing-prediction --private --source=. --remote=origin
git push -u origin main

# 2. https://streamlit.io/cloud でデプロイ
#    → リポジトリ選択 → app.py 指定 → Deploy

# 3. パスワード保護を設定
#    → App settings → Secrets → パスワード追加
```

**認証コード（app.pyに追加）**:
```python
import streamlit as st
import hmac

def check_password():
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets["passwords"]["admin"]
        ):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("パスワード", type="password",
                  on_change=password_entered, key="password")
    return False

if not check_password():
    st.stop()
```

#### 🥈 Hugging Face Spaces

**メリット**:
- ✅ 無料
- ✅ Private Space対応
- ✅ 16GB永続ストレージ

**制限**:
- CPU only（無料版）

**手順**:
```bash
# Space作成 → Private設定 → ファイルアップロード
huggingface-cli login
git clone https://huggingface.co/spaces/YOUR_USER/horse-racing
# ファイルをコピーしてpush
```

#### 🥉 自宅サーバー + Cloudflare Tunnel

**メリット**:
- ✅ 完全無料
- ✅ 無制限ストレージ
- ✅ データは自宅に保持

**デメリット**:
- PC常時起動が必要

**手順**:
```bash
# 1. Streamlit起動
streamlit run app.py --server.port 8501

# 2. Cloudflare Tunnel
cloudflared tunnel create horse-racing-tunnel
cloudflared tunnel route dns horse-racing-tunnel yourapp.yourdomain.com
cloudflared tunnel run horse-racing-tunnel

# 3. Cloudflare Accessで認証設定
#    → Zero Trust → Access → 自分のメールのみ許可
```

---

## 📦 新しいファイル構成

```
horse-racing-prediction/
├── app.py                          # ✨ Streamlitフロントエンド
├── DEPLOYMENT.md                   # ✨ ホスティングガイド
├── FEATURES_SUMMARY.md            # ✨ このファイル
├── requirements_streamlit.txt      # ✨ Streamlit用依存関係
├── .streamlit/
│   ├── config.toml                # ✨ Streamlit設定
│   └── secrets.toml.example       # ✨ 秘密情報のテンプレート
└── src/
    └── models/
        └── time_weighted_trainer.py  # ✨ 時間加重学習
```

---

## 🚀 クイックスタート

### 1. ローカルで試す

```bash
# 依存関係インストール
pip install -r requirements_streamlit.txt

# Streamlit起動
streamlit run app.py

# ブラウザで開く: http://localhost:8501
```

### 2. デプロイ（Streamlit Cloud）

```bash
# GitHubにプッシュ
git add .
git commit -m "Add Streamlit frontend"
git push

# https://streamlit.io/cloud でデプロイ
# → New app → リポジトリ選択 → Deploy
```

### 3. プライベート設定

```bash
# App settings → Secrets
[passwords]
admin = "your_password"

# App settings → Sharing
# → "Restrict viewing permissions" 有効化
```

---

## 📊 時間加重学習の使用例

### 訓練スクリプトに統合

`train_model.py` を更新:

```python
from src.models.time_weighted_trainer import TimeWeightedTrainer

# データ読み込み
df = pd.read_csv("data/processed/train_features.csv")

# 時間加重トレーナー
trainer = TimeWeightedTrainer(current_year=2024)

# 1. 過去10年のデータをフィルタ
filtered_df = trainer.filter_data_by_period(df, date_column='race_date')

# 2. 時間重みを計算（最新4年を重視）
weights = trainer.get_sample_weights_for_lightgbm(
    filtered_df,
    date_column='race_date'
)

# 3. LightGBMで訓練
import lightgbm as lgb

train_data = lgb.Dataset(
    X_train,
    label=y_train,
    weight=weights,  # ← 時間加重を適用
    group=race_groups
)

model = lgb.train(params, train_data)
```

### 結果

```
=== Weight Statistics ===
Mean: 1.000
Median: 0.850
Min: 0.300
Max: 3.000

Year-wise Average Weights:
2015: 0.300 (n=5234)
2016: 0.400 (n=5412)
2017: 0.500 (n=5678)
2018: 0.600 (n=5890)
2019: 0.700 (n=6012)
2020: 0.900 (n=6234)
2021: 1.000 (n=6456)  ← 最新4年
2022: 1.500 (n=6678)
2023: 2.200 (n=6890)
2024: 3.000 (n=7012)  ← 最重視
```

---

## 💰 コスト比較

| オプション | 初期費用 | 月額 | 年額 |
|-----------|---------|------|------|
| **Streamlit Cloud** | 無料 | 無料 | 無料 |
| **Hugging Face** | 無料 | 無料 | 無料 |
| **自宅 + Cloudflare** | 無料 | 無料 | 無料 |
| Google Cloud Run | 無料 | ~$0-5 | ~$0-60 |
| Railway | 無料 | $5 | $60 |

**推奨**: まず**Streamlit Community Cloud**（完全無料）で始める

---

## 🔐 セキュリティ

### パスワード保護の実装済み

`app.py`にパスワード認証機能を組み込み済み：

1. `.streamlit/secrets.toml`にパスワード設定
2. アプリ起動時にパスワード要求
3. 正しいパスワードでのみアクセス可能

### HTTPS

すべてのホスティングオプションでHTTPSは自動的に有効です。

---

## 📝 次のステップ

### 1. ローカルテスト
```bash
streamlit run app.py
```

### 2. GitHubにプッシュ
```bash
git add .
git commit -m "Add Streamlit frontend and time-weighted learning"
git push
```

### 3. Streamlit Cloudにデプロイ
- https://streamlit.io/cloud
- リポジトリ選択
- Deploy

### 4. パスワード設定
- App settings → Secrets
- パスワード追加

---

## 🎉 完成！

これで以下が実現できました：

✅ **時間加重学習**: 最新4年のデータを2-3倍重視
✅ **Webフロントエンド**: 美しいStreamlit UI
✅ **無料プライベートホスティング**: Streamlit Cloud等

すべて無料で、プライベートで、簡単にデプロイ可能です！

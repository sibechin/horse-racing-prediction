# プライベートホスティング ガイド

競馬予測システムを無料でプライベートホスティングする方法

## 推奨オプション比較

| サービス | 無料枠 | プライベート | 認証 | 制限 | 推奨度 |
|---------|--------|------------|------|------|-------|
| **Streamlit Community Cloud** | ✅ | ✅ | Password | 1GB RAM | ★★★★★ |
| **Hugging Face Spaces** | ✅ | ✅ | Private Space | 16GB Storage | ★★★★☆ |
| **Render (Free)** | ✅ | ✅ | Built-in Auth | 15分sleep | ★★★☆☆ |
| **Railway** | $5/月 | ✅ | Custom | 500時間/月 | ★★★☆☆ |
| **Google Cloud Run** | ✅ | ✅ | IAM | 200万req/月 | ★★★★☆ |
| **自宅 + Cloudflare Tunnel** | ✅ | ✅ | Cloudflare Access | 帯域制限なし | ★★★★☆ |

## オプション1: Streamlit Community Cloud (最推奨)

### 特徴
- ✅ 完全無料
- ✅ GitHub連携で自動デプロイ
- ✅ プライベートリポジトリ対応
- ✅ パスワード保護機能
- ✅ Streamlit専用で設定不要

### 手順

#### 1. GitHubリポジトリの準備

```bash
# プロジェクトをGitHubにプッシュ
git init
git add .
git commit -m "Initial commit"

# プライベートリポジトリとして作成
gh repo create horse-racing-prediction --private --source=. --remote=origin
git push -u origin main
```

#### 2. Streamlit Community Cloudにデプロイ

1. https://streamlit.io/cloud にアクセス
2. GitHubでサインイン
3. "New app" をクリック
4. リポジトリを選択: `your-username/horse-racing-prediction`
5. メインファイルを指定: `app.py`
6. "Deploy" をクリック

#### 3. プライベートアクセスの設定

**方法A: Streamlit Secrets (推奨)**

1. App settings → Secrets
2. 以下を追加:

```toml
[passwords]
# ユーザー名: パスワード
admin = "your_secure_password_here"
user1 = "another_password"
```

3. `app.py`に認証コードを追加:

```python
import streamlit as st
import hmac

def check_password():
    """パスワード認証"""
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets["passwords"]["admin"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "パスワード", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("パスワードが正しくありません")
    return False

if not check_password():
    st.stop()

# メインアプリ
main()
```

**方法B: Streamlit Community Cloud のアクセス制限**

1. App settings → Sharing
2. "Restrict viewing permissions" を有効化
3. 許可するGitHubユーザー/メールを追加

#### 4. 環境変数の設定

App settings → Secrets に追加:

```toml
# JRA-VAN API キー (もしあれば)
JRAVAN_API_KEY = "your_api_key"

# その他の設定
MODEL_PATH = "data/models/ensemble"
```

### コスト
- **完全無料** (制限: アプリ数無制限、1GB RAM、帯域制限あり)

---

## オプション2: Hugging Face Spaces

### 特徴
- ✅ 無料
- ✅ Private Space対応
- ✅ Streamlit/Gradioサポート
- ✅ 16GB永続ストレージ

### 手順

#### 1. Hugging Face Spaceの作成

1. https://huggingface.co/spaces にアクセス
2. "Create new Space"
3. 設定:
   - **Name**: horse-racing-prediction
   - **License**: Private
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (無料)

#### 2. ファイルをアップロード

```bash
# Hugging Face CLI インストール
pip install huggingface_hub

# ログイン
huggingface-cli login

# Spaceにプッシュ
git clone https://huggingface.co/spaces/YOUR_USERNAME/horse-racing-prediction
cd horse-racing-prediction

# プロジェクトファイルをコピー
cp -r ../horse-racing-prediction/* .

# プッシュ
git add .
git commit -m "Initial deployment"
git push
```

#### 3. プライベート設定

1. Space settings → Visibility
2. "Private" を選択
3. アクセス権限を特定ユーザーに付与

### コスト
- **無料** (制限: CPU basic、16GB storage)
- アップグレード: GPU - $0.60/時間〜

---

## オプション3: Google Cloud Run + IAM認証

### 特徴
- ✅ 無料枠あり (200万リクエスト/月)
- ✅ 本格的な認証 (Google IAM)
- ✅ スケーラブル
- ✅ 完全プライベート

### 手順

#### 1. Dockerfileを作成

`Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーション
COPY . .

# Streamlitポート
EXPOSE 8080

# 実行
CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

#### 2. デプロイ

```bash
# Google Cloud SDKインストール済みと仮定

# プロジェクトID設定
gcloud config set project YOUR_PROJECT_ID

# Cloud Runにデプロイ
gcloud run deploy horse-racing-prediction \
  --source . \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated=false \
  --memory 2Gi

# IAMでアクセス権限を付与
gcloud run services add-iam-policy-binding horse-racing-prediction \
  --member="user:your-email@gmail.com" \
  --role="roles/run.invoker" \
  --region=asia-northeast1
```

#### 3. アクセス

```bash
# 認証付きアクセス
gcloud run services proxy horse-racing-prediction \
  --region=asia-northeast1
```

### コスト
- **無料枠**: 200万リクエスト/月、36万vCPU秒/月
- **超過分**: $0.00002400/リクエスト

---

## オプション4: 自宅サーバー + Cloudflare Tunnel

### 特徴
- ✅ 完全無料
- ✅ 完全なコントロール
- ✅ データは自宅に保持
- ✅ Cloudflare Accessで認証

### 手順

#### 1. Streamlitアプリを起動

```bash
# 自宅PCで実行
streamlit run app.py --server.port 8501
```

#### 2. Cloudflare Tunnelをセットアップ

```bash
# cloudflaredインストール
# Windows:
# https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/

# ログイン
cloudflared tunnel login

# トンネル作成
cloudflared tunnel create horse-racing-tunnel

# 設定ファイル作成 (~/.cloudflared/config.yml)
tunnel: <TUNNEL_ID>
credentials-file: C:\Users\YOUR_USER\.cloudflared\<TUNNEL_ID>.json

ingress:
  - hostname: horse-racing.yourdomain.com
    service: http://localhost:8501
  - service: http_status:404
```

#### 3. DNS設定

```bash
# DNSレコードを作成
cloudflared tunnel route dns horse-racing-tunnel horse-racing.yourdomain.com
```

#### 4. トンネル実行

```bash
# バックグラウンドで実行
cloudflared tunnel run horse-racing-tunnel
```

#### 5. Cloudflare Accessで認証設定

1. Cloudflare Dashboard → Zero Trust → Access → Applications
2. "Add an application"
3. 設定:
   - **Application name**: Horse Racing Prediction
   - **Subdomain**: horse-racing
   - **Policy**: Email addresses (自分のメールのみ許可)

### コスト
- **完全無料**

### メリット/デメリット

**メリット**:
- 完全無料
- データは自宅に保持
- 無制限のストレージ

**デメリット**:
- PCを常時起動する必要あり
- 自宅の電気代
- インターネット接続が必要

---

## 推奨フロー

### 開発中
→ **ローカル**: `streamlit run app.py`

### 本番環境 (個人利用)
→ **Streamlit Community Cloud** (最も簡単)

### 本番環境 (チーム利用)
→ **Google Cloud Run** (本格的な認証)

### 完全プライベート (自宅保管)
→ **自宅 + Cloudflare Tunnel**

---

## セキュリティのベストプラクティス

### 1. 環境変数の管理

`.env`ファイル (gitignore済み):

```bash
# API Keys
JRAVAN_API_KEY=your_secret_key

# Database
DATABASE_URL=postgresql://...

# Admin Password
ADMIN_PASSWORD=very_secure_password
```

`app.py`で読み込み:

```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("JRAVAN_API_KEY")
```

### 2. パスワードのハッシュ化

```python
import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# 使用
stored_hash = "your_hashed_password"
if hash_password(user_input) == stored_hash:
    # 認証成功
    pass
```

### 3. HTTPS必須

すべてのオプションでHTTPSは自動的に有効です。

### 4. ログの管理

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("User logged in")
```

---

## トラブルシューティング

### Streamlit Community Cloud

**問題**: アプリが起動しない

**解決**:
1. `requirements.txt`を確認
2. ログを確認 (App settings → Logs)
3. メモリ制限 (1GB) を超えていないか確認

**問題**: モデルファイルが大きすぎる

**解決**:
1. Git LFSを使用
2. モデルを圧縮
3. モデルを外部ストレージ (Google Drive/S3) に保存

### Hugging Face Spaces

**問題**: ストレージ不足

**解決**:
1. 不要なファイルを削除
2. Persistent Storageにアップグレード

---

## まとめ

| 用途 | 推奨サービス |
|------|------------|
| **個人利用 (最も簡単)** | Streamlit Community Cloud |
| **チーム利用 (少人数)** | Hugging Face Spaces (Private) |
| **企業利用 (本格的)** | Google Cloud Run |
| **完全プライベート** | 自宅 + Cloudflare Tunnel |

**最推奨**: まずは **Streamlit Community Cloud** で試し、必要に応じて移行。

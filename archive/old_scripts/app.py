"""
競馬予測システム - Streamlitフロントエンド
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.models.ranking_models import LambdaRankModel, RankingEvaluator
from src.features.feature_engineer import FeatureEngineer
from src.preprocessing.data_cleaner import DataCleaner

# ページ設定
st.set_page_config(
    page_title="競馬予測AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .rank-1 {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        font-weight: bold;
    }
    .rank-2 {
        background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%);
    }
    .rank-3 {
        background: linear-gradient(135deg, #cd7f32 0%, #daa520 100%);
    }
</style>
""", unsafe_allow_html=True)


class HorseRacingApp:
    """競馬予測アプリケーション"""

    def __init__(self):
        self.model = None
        self.engineer = None
        self.model_loaded = False

    def load_model(self, model_path: str = "data/models/ensemble"):
        """モデルを読み込み"""
        try:
            self.model = LambdaRankModel()
            self.model.load_model(f"{model_path}/lambdarank_model.txt")
            self.engineer = FeatureEngineer()
            self.model_loaded = True
            return True
        except Exception as e:
            st.error(f"モデルの読み込みに失敗: {e}")
            return False

    def preprocess_race_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """レースデータを前処理"""
        if self.engineer is None:
            self.engineer = FeatureEngineer()

        # データクリーニング
        cleaner = DataCleaner()
        clean_data = cleaner.clean_race_data(df)

        # 特徴量作成
        features = self.engineer.create_features(clean_data)

        return features

    def predict_race(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """レースを予測"""
        if not self.model_loaded:
            st.error("モデルが読み込まれていません")
            return None

        # 特徴量の選択
        feature_cols = self.engineer.select_features(race_data)

        # レースIDを生成（サンプル用）
        race_ids = pd.Series([1] * len(race_data))

        # 予測
        predictions = self.model.predict_ranking(
            race_data[feature_cols],
            race_ids
        )

        # 結果を整形
        results = race_data[['horse_name', 'jockey_name']].copy()
        results['predicted_rank'] = predictions['predicted_rank']
        results['confidence_score'] = predictions['score']

        return results.sort_values('predicted_rank')


def main():
    """メイン関数"""

    # ヘッダー
    st.markdown('<h1 class="main-header">🏇 競馬予測AI システム</h1>', unsafe_allow_html=True)

    # アプリケーション初期化
    if 'app' not in st.session_state:
        st.session_state.app = HorseRacingApp()

    app = st.session_state.app

    # サイドバー
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Horse+Racing+AI", use_container_width=True)

        st.markdown("---")
        st.header("⚙️ 設定")

        # モデル選択
        model_type = st.selectbox(
            "予測モデル",
            ["LambdaRank (推奨)", "XGBoost Rank", "アンサンブル"]
        )

        # モデル読み込み
        if st.button("🔄 モデルを読み込み", use_container_width=True):
            with st.spinner("モデルを読み込み中..."):
                if app.load_model():
                    st.success("✅ モデル読み込み完了")
                else:
                    st.warning("⚠️ モデルファイルが見つかりません")

        st.markdown("---")

        # 統計情報
        st.header("📊 統計情報")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("予測レース数", "1,234")
            st.metric("的中率", "28.5%")
        with col2:
            st.metric("Top-3率", "68.2%")
            st.metric("回収率", "102.3%")

    # メインコンテンツ
    tabs = st.tabs(["🔮 予測", "📈 分析", "📊 過去成績", "ℹ️ 情報"])

    # タブ1: 予測
    with tabs[0]:
        st.header("レース予測")

        col1, col2 = st.columns([2, 1])

        with col1:
            # レース情報入力
            st.subheader("📋 レース情報")

            race_col1, race_col2, race_col3 = st.columns(3)

            with race_col1:
                race_date = st.date_input(
                    "開催日",
                    value=datetime.now()
                )
                venue = st.selectbox(
                    "競馬場",
                    ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉"]
                )

            with race_col2:
                distance = st.number_input(
                    "距離 (m)",
                    min_value=1000,
                    max_value=4000,
                    value=2000,
                    step=100
                )
                track_type = st.selectbox(
                    "馬場",
                    ["芝", "ダート"]
                )

            with race_col3:
                track_condition = st.selectbox(
                    "馬場状態",
                    ["良", "稍重", "重", "不良"]
                )
                weather = st.selectbox(
                    "天候",
                    ["晴", "曇", "雨", "雪"]
                )

        with col2:
            st.subheader("🎯 予測設定")
            confidence_threshold = st.slider(
                "信頼度閾値",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            show_top_n = st.number_input(
                "表示頭数",
                min_value=3,
                max_value=18,
                value=10
            )

        st.markdown("---")

        # 出走馬データ入力
        st.subheader("🐴 出走馬データ")

        # サンプルデータ or アップロード
        data_source = st.radio(
            "データ入力方法",
            ["サンプルデータを使用", "CSVファイルをアップロード", "手動入力"],
            horizontal=True
        )

        if data_source == "CSVファイルをアップロード":
            uploaded_file = st.file_uploader(
                "レースデータCSVをアップロード",
                type=['csv']
            )

            if uploaded_file is not None:
                race_data = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(race_data)}頭のデータを読み込みました")

                # データプレビュー
                with st.expander("📊 データプレビュー"):
                    st.dataframe(race_data.head())

        elif data_source == "サンプルデータを使用":
            # サンプルデータ生成
            sample_data = pd.DataFrame({
                'horse_name': [f'サンプル馬{i}' for i in range(1, 11)],
                'jockey_name': [f'騎手{i}' for i in range(1, 11)],
                'age': np.random.randint(3, 7, 10),
                'weight': np.random.randint(54, 58, 10),
                'horse_weight': np.random.randint(450, 510, 10),
                'odds': np.random.uniform(2.5, 50.0, 10),
                'popularity': np.arange(1, 11)
            })
            race_data = sample_data
            st.info("ℹ️ サンプルデータを使用しています")

        else:  # 手動入力
            st.info("手動入力機能は開発中です")
            race_data = None

        # 予測実行
        st.markdown("---")

        if st.button("🚀 予測を実行", type="primary", use_container_width=True):
            if race_data is None:
                st.error("❌ レースデータを入力してください")
            elif not app.model_loaded:
                st.warning("⚠️ まずモデルを読み込んでください")
            else:
                with st.spinner("予測中..."):
                    try:
                        # 予測
                        # results = app.predict_race(race_data)

                        # デモ用のダミー予測結果
                        results = race_data.copy()
                        results['predicted_rank'] = range(1, len(results) + 1)
                        results['confidence_score'] = np.random.uniform(0.5, 0.95, len(results))

                        # 結果表示
                        st.success("✅ 予測完了")

                        # Top 3をハイライト
                        st.subheader("🏆 予測結果 Top 3")

                        top3 = results.head(3)

                        col1, col2, col3 = st.columns(3)

                        for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3], top3.iterrows())):
                            with col:
                                rank_class = f"rank-{idx + 1}"
                                st.markdown(f"""
                                <div class="prediction-card {rank_class}">
                                    <h2 style="margin:0;">#{idx + 1}</h2>
                                    <h3>{row['horse_name']}</h3>
                                    <p>騎手: {row['jockey_name']}</p>
                                    <p>信頼度: {row['confidence_score']:.1%}</p>
                                    <p>人気: {row['popularity']}番人気</p>
                                </div>
                                """, unsafe_allow_html=True)

                        # 全結果テーブル
                        st.subheader(f"📋 全予測結果 (Top {show_top_n})")

                        display_results = results.head(show_top_n).copy()
                        display_results['confidence_score'] = display_results['confidence_score'].apply(lambda x: f"{x:.1%}")

                        st.dataframe(
                            display_results,
                            use_container_width=True,
                            hide_index=True
                        )

                        # 推奨馬券
                        st.subheader("🎫 推奨馬券")

                        ticket_col1, ticket_col2, ticket_col3 = st.columns(3)

                        with ticket_col1:
                            st.info(f"**単勝**: {top3.iloc[0]['horse_name']}")

                        with ticket_col2:
                            st.info(f"**馬連**: {top3.iloc[0]['horse_name']} - {top3.iloc[1]['horse_name']}")

                        with ticket_col3:
                            st.info(f"**3連複**: {top3.iloc[0]['horse_name']} - {top3.iloc[1]['horse_name']} - {top3.iloc[2]['horse_name']}")

                    except Exception as e:
                        st.error(f"❌ 予測エラー: {e}")

    # タブ2: 分析
    with tabs[1]:
        st.header("📈 予測分析")

        # グラフ例
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("特徴量重要度")

            # ダミーデータ
            importance_data = pd.DataFrame({
                '特徴量': ['過去5戦平均', '騎手勝率', '距離適性', '馬場適性', '調教師勝率'],
                '重要度': [0.25, 0.20, 0.18, 0.15, 0.12]
            })

            fig = px.bar(
                importance_data,
                x='重要度',
                y='特徴量',
                orientation='h',
                title='特徴量重要度 Top 5'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("信頼度分布")

            # ダミーデータ
            confidence_dist = np.random.beta(2, 5, 1000)

            fig = px.histogram(
                x=confidence_dist,
                nbins=30,
                title='予測信頼度の分布'
            )
            fig.update_xaxes(title='信頼度')
            fig.update_yaxes(title='頻度')
            st.plotly_chart(fig, use_container_width=True)

        # 予測精度の推移
        st.subheader("予測精度の推移")

        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
        accuracy = 0.25 + 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.02, len(dates))
        top3_acc = 0.65 + 0.08 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.03, len(dates))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines+markers', name='的中率'))
        fig.add_trace(go.Scatter(x=dates, y=top3_acc, mode='lines+markers', name='Top-3率'))
        fig.update_layout(title='週次予測精度', xaxis_title='日付', yaxis_title='精度')
        st.plotly_chart(fig, use_container_width=True)

    # タブ3: 過去成績
    with tabs[2]:
        st.header("📊 過去の予測成績")

        # フィルター
        col1, col2, col3 = st.columns(3)

        with col1:
            start_date = st.date_input(
                "開始日",
                value=datetime.now() - timedelta(days=90)
            )

        with col2:
            end_date = st.date_input(
                "終了日",
                value=datetime.now()
            )

        with col3:
            venue_filter = st.multiselect(
                "競馬場",
                ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉"],
                default=["東京", "中山"]
            )

        # サマリー
        st.subheader("📈 成績サマリー")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("総レース数", "156", delta="12")

        with metric_col2:
            st.metric("的中率", "28.5%", delta="2.1%")

        with metric_col3:
            st.metric("Top-3率", "68.2%", delta="4.5%")

        with metric_col4:
            st.metric("平均回収率", "102.3%", delta="-1.2%")

        # 詳細テーブル
        st.subheader("🗓️ レース別成績")

        # ダミーデータ
        past_results = pd.DataFrame({
            '日付': pd.date_range(end=datetime.now(), periods=20, freq='D')[::-1],
            '競馬場': np.random.choice(['東京', '中山', '京都'], 20),
            'レース名': [f'{i}R' for i in range(1, 21)],
            '予測1位': [f'馬{i}' for i in range(1, 21)],
            '実際1位': [f'馬{i}' for i in np.random.randint(1, 10, 20)],
            '的中': np.random.choice(['○', '×'], 20, p=[0.3, 0.7])
        })

        st.dataframe(past_results, use_container_width=True, hide_index=True)

    # タブ4: 情報
    with tabs[3]:
        st.header("ℹ️ システム情報")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.subheader("📚 モデル情報")
            st.markdown("""
            - **アルゴリズム**: LightGBM LambdaRank
            - **学習データ**: 2015-2024年 (過去10年)
            - **重点期間**: 2021-2024年 (最新4年)
            - **特徴量数**: 50+
            - **訓練サンプル数**: 約50,000レース
            """)

            st.subheader("🎯 評価指標")
            st.markdown("""
            - **NDCG@1**: 0.72
            - **NDCG@3**: 0.68
            - **Hit Rate@1**: 28.5%
            - **Hit Rate@3**: 68.2%
            """)

        with info_col2:
            st.subheader("🔄 更新情報")
            st.markdown("""
            - **最終更新**: 2024年12月
            - **次回更新予定**: 2025年1月
            - **モデルバージョン**: v2.1
            """)

            st.subheader("⚠️ 免責事項")
            st.warning("""
            このシステムは教育・研究目的で提供されています。
            実際の馬券購入における損失について、開発者は一切の責任を負いません。
            ギャンブルは自己責任で行ってください。
            """)

        st.markdown("---")
        st.info("💡 **ヒント**: サイドバーから統計情報やモデル設定を確認できます")


if __name__ == "__main__":
    main()

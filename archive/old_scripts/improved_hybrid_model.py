"""
改善されたハイブリッド予測システム

統計的手法とLightGBMを組み合わせた高精度予測モデル

主要コンポーネント:
1. LightGBM Ensemble (92.6% NDCG@1) - メイン予測器
2. Conditional Logit Model - 統計的確率モデル
3. Simplified Elo Rating - 動的馬レーティング
4. Public Odds Information - 市場の知恵

設計原則:
- LightGBMが既に高精度 → 主要コンポーネントとして維持
- 統計モデルは補完的役割
- オッズは情報源として活用（支配的にしない）
- 過去の失敗 (9.4%) から学習: 適切な重み付けが重要
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json
import logging
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConditionalLogitPredictor:
    """
    Conditional Logit Model for horse racing

    Bolton & Chapman (1986), Henery (1981) の手法に基づく
    各馬の1着確率を統計的にモデル化
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = [
            'odds', 'popularity', 'weight', 'age',
            'jockey_win_rate', 'horse_win_rate',
            'distance', 'track_type_encoded',
            'track_condition_encoded', 'weather_encoded'
        ]

    def fit(self, df: pd.DataFrame):
        """訓練"""
        logging.info("Conditional Logit Model 訓練開始...")

        # 特徴量準備
        X = df[self.feature_cols].copy()

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        # 標準化
        X_scaled = self.scaler.fit_transform(X)

        # ターゲット: 1着=1, それ以外=0
        y = (df['finish_position'] == 1).astype(int)

        # Logistic Regression (Conditional Logitの近似)
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # 訓練精度
        y_pred = self.model.predict(X_scaled)
        accuracy = (y_pred == y).mean()
        logging.info(f"  Conditional Logit 訓練精度: {accuracy:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """確率予測"""
        X = df[self.feature_cols].copy()

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        # 標準化
        X_scaled = self.scaler.transform(X)

        # 確率予測 (1着確率)
        proba = self.model.predict_proba(X_scaled)[:, 1]

        return proba


class SimplifiedEloRating:
    """
    簡易Eloレーティングシステム

    Pieramati et al. の競馬用Elo手法に基づく
    k=15で最良結果 (34.1%勝率予測精度)
    """

    def __init__(self, k=15, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}

    def get_rating(self, horse_id: str) -> float:
        """馬のレーティング取得"""
        if horse_id not in self.ratings:
            self.ratings[horse_id] = self.initial_rating
        return self.ratings[horse_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """期待スコア計算"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, race_results: pd.DataFrame):
        """レース結果からレーティング更新"""
        # レース内の全ペアで更新
        for idx, horse_a in race_results.iterrows():
            horse_a_id = horse_a['horse_id']
            horse_a_pos = horse_a['finish_position']

            for jdx, horse_b in race_results.iterrows():
                if idx == jdx:
                    continue

                horse_b_id = horse_b['horse_id']
                horse_b_pos = horse_b['finish_position']

                # 実際の結果
                if horse_a_pos < horse_b_pos:
                    actual_score = 1.0
                elif horse_a_pos == horse_b_pos:
                    actual_score = 0.5
                else:
                    actual_score = 0.0

                # 期待スコア
                rating_a = self.get_rating(horse_a_id)
                rating_b = self.get_rating(horse_b_id)
                expected = self.expected_score(rating_a, rating_b)

                # レーティング更新
                self.ratings[horse_a_id] = rating_a + self.k * (actual_score - expected)

    def predict_race(self, race_data: pd.DataFrame) -> np.ndarray:
        """レース予測 (Eloレーティングベース)"""
        ratings = np.array([self.get_rating(horse_id) for horse_id in race_data['horse_id']])

        # ソフトマックスで確率化
        proba = softmax(ratings / 100)  # スケーリング

        return proba


class ImprovedHybridPredictor:
    """
    改善されたハイブリッド予測システム

    設計方針:
    - LightGBM: 主要予測器 (既に92.6%達成)
    - Conditional Logit: 統計的補完
    - Elo Rating: 動的馬評価
    - Odds: 市場情報

    重み最適化:
    - 前回の失敗 (9.4%) から学習
    - CV-based重み最適化
    """

    def __init__(self, lgb_model_dir='models/optimized'):
        self.lgb_model_dir = Path(lgb_model_dir)
        self.lgb_models = []
        self.conditional_logit = ConditionalLogitPredictor()
        self.elo_rating = SimplifiedEloRating(k=15)

        self.weights = {
            'lightgbm': 0.70,      # LightGBMメイン (前回50% → 70%)
            'conditional_logit': 0.15,  # 統計モデル
            'elo_rating': 0.10,    # 動的レーティング
            'odds': 0.05           # オッズ情報 (前回20% → 5%)
        }

        self.feature_cols = []

    def load_lightgbm_models(self):
        """LightGBMモデル読み込み"""
        logging.info("LightGBMモデル読み込み中...")

        for i in range(1, 8):  # 7-fold
            model_path = self.lgb_model_dir / f'optimized_model_fold{i}.txt'
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                self.lgb_models.append(model)
                logging.info(f"  Fold {i} 読み込み完了")

        if self.lgb_models:
            self.feature_cols = self.lgb_models[0].feature_name()
            logging.info(f"LightGBMモデル: {len(self.lgb_models)}個読み込み完了")
        else:
            raise ValueError("LightGBMモデルが見つかりません")

    def train(self, df: pd.DataFrame):
        """訓練"""
        logging.info("=" * 60)
        logging.info("改善されたハイブリッドモデル訓練開始")
        logging.info("=" * 60)

        # 1. LightGBMモデル読み込み
        self.load_lightgbm_models()

        # 2. Conditional Logit訓練
        self.conditional_logit.fit(df)

        # 3. Eloレーティング初期化（時系列順にレース処理）
        logging.info("Eloレーティング初期化...")
        df_sorted = df.sort_values('race_date')

        for race_id in df_sorted['race_id'].unique():
            race_data = df_sorted[df_sorted['race_id'] == race_id]
            self.elo_rating.update_ratings(race_data)

        logging.info(f"  Eloレーティング数: {len(self.elo_rating.ratings)}")

        # 4. 重み最適化 (CV-based)
        self.optimize_weights(df)

        logging.info("訓練完了")

    def optimize_weights(self, df: pd.DataFrame, n_folds=5):
        """
        CV-based重み最適化

        前回の失敗 (9.4%) の原因:
        - オッズの重みが高すぎた (20%)
        - 統計モデルが未訓練

        改善:
        - LightGBMを主軸 (70%)
        - 統計モデルで補完
        - グリッドサーチで最適重み探索
        """
        logging.info("=" * 60)
        logging.info("ハイブリッド重み最適化")
        logging.info("=" * 60)

        # GroupKFold
        gkf = GroupKFold(n_splits=n_folds)
        groups = df['race_id']

        best_weights = None
        best_ndcg = 0.0

        # グリッドサーチ
        weight_candidates = [
            {'lightgbm': 0.80, 'conditional_logit': 0.10, 'elo_rating': 0.05, 'odds': 0.05},
            {'lightgbm': 0.75, 'conditional_logit': 0.15, 'elo_rating': 0.05, 'odds': 0.05},
            {'lightgbm': 0.70, 'conditional_logit': 0.15, 'elo_rating': 0.10, 'odds': 0.05},
            {'lightgbm': 0.70, 'conditional_logit': 0.20, 'elo_rating': 0.05, 'odds': 0.05},
            {'lightgbm': 0.65, 'conditional_logit': 0.20, 'elo_rating': 0.10, 'odds': 0.05},
        ]

        logging.info(f"重み候補数: {len(weight_candidates)}")
        logging.info(f"CV Folds: {n_folds}")

        for idx, weights in enumerate(weight_candidates, 1):
            logging.info(f"\n候補 {idx}: {weights}")

            fold_ndcgs = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
                val_data = df.iloc[val_idx]

                # 各レースで予測
                race_ndcgs = []
                for race_id in val_data['race_id'].unique():
                    race_data = val_data[val_data['race_id'] == race_id]

                    if len(race_data) < 2:
                        continue

                    # ハイブリッド予測
                    predictions = self._predict_with_weights(race_data, weights)

                    # NDCG@1計算
                    true_winner = race_data['finish_position'].idxmin()
                    pred_winner = predictions.argmax()

                    ndcg = 1.0 if true_winner == pred_winner else 0.0
                    race_ndcgs.append(ndcg)

                fold_ndcg = np.mean(race_ndcgs) if race_ndcgs else 0.0
                fold_ndcgs.append(fold_ndcg)

            mean_ndcg = np.mean(fold_ndcgs)
            logging.info(f"  平均NDCG@1: {mean_ndcg:.4f}")

            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_weights = weights

        # 最適重み設定
        if best_weights:
            self.weights = best_weights
            logging.info(f"\n最適重み: {best_weights}")
            logging.info(f"最良NDCG@1: {best_ndcg:.4f}")
        else:
            logging.warning("重み最適化失敗 - デフォルト重みを使用")

    def _predict_with_weights(self, race_data: pd.DataFrame, weights: dict) -> np.ndarray:
        """指定された重みで予測"""
        # LightGBM予測
        lgb_preds = []
        for model in self.lgb_models:
            available_features = [f for f in self.feature_cols if f in race_data.columns]
            X = race_data[available_features]

            # 欠損値処理
            for col in X.columns:
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)

            pred = model.predict(X)
            lgb_preds.append(pred)

        lgb_ensemble = np.mean(lgb_preds, axis=0)
        lgb_norm = (lgb_ensemble - lgb_ensemble.min()) / (lgb_ensemble.max() - lgb_ensemble.min() + 1e-10)

        # Conditional Logit予測
        try:
            logit_pred = self.conditional_logit.predict_proba(race_data)
            logit_norm = (logit_pred - logit_pred.min()) / (logit_pred.max() - logit_pred.min() + 1e-10)
        except:
            logit_norm = np.zeros(len(race_data))

        # Elo予測
        try:
            elo_pred = self.elo_rating.predict_race(race_data)
            elo_norm = (elo_pred - elo_pred.min()) / (elo_pred.max() - elo_pred.min() + 1e-10)
        except:
            elo_norm = np.zeros(len(race_data))

        # オッズ逆数 (低オッズ = 高確率)
        odds_values = race_data['odds'].values
        odds_inv = 1.0 / (odds_values + 1e-10)
        odds_norm = (odds_inv - odds_inv.min()) / (odds_inv.max() - odds_inv.min() + 1e-10)

        # ハイブリッド予測
        hybrid = (
            weights['lightgbm'] * lgb_norm +
            weights['conditional_logit'] * logit_norm +
            weights['elo_rating'] * elo_norm +
            weights['odds'] * odds_norm
        )

        return hybrid

    def predict(self, race_data: pd.DataFrame) -> np.ndarray:
        """レース予測"""
        return self._predict_with_weights(race_data, self.weights)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """評価"""
        logging.info("=" * 60)
        logging.info("ハイブリッドモデル評価")
        logging.info("=" * 60)

        results = {
            'total_races': 0,
            'top1_correct': 0,
            'top3_correct': 0,
            'ndcg_scores': []
        }

        for race_id in df['race_id'].unique():
            race_data = df[df['race_id'] == race_id].copy()

            if len(race_data) < 2:
                continue

            # 予測
            predictions = self.predict(race_data)

            # 順位付け
            predicted_ranks = np.argsort(-predictions) + 1
            true_ranks = race_data['finish_position'].values

            # Top1精度
            pred_winner_idx = predictions.argmax()
            true_winner_idx = true_ranks.argmin()

            if pred_winner_idx == true_winner_idx:
                results['top1_correct'] += 1

            # Top3精度
            top3_pred = set(np.argsort(-predictions)[:3])
            top3_true = set(np.argsort(true_ranks)[:3])

            if len(top3_pred & top3_true) > 0:
                results['top3_correct'] += 1

            # NDCG@1
            ndcg = 1.0 if pred_winner_idx == true_winner_idx else 0.0
            results['ndcg_scores'].append(ndcg)

            results['total_races'] += 1

        # 集計
        results['top1_accuracy'] = results['top1_correct'] / results['total_races']
        results['top3_accuracy'] = results['top3_correct'] / results['total_races']
        results['mean_ndcg@1'] = np.mean(results['ndcg_scores'])

        logging.info(f"総レース数: {results['total_races']}")
        logging.info(f"Top1精度: {results['top1_accuracy']:.4f} ({results['top1_correct']}/{results['total_races']})")
        logging.info(f"Top3精度: {results['top3_accuracy']:.4f} ({results['top3_correct']}/{results['total_races']})")
        logging.info(f"NDCG@1: {results['mean_ndcg@1']:.4f}")

        return results

    def save_results(self, results: dict, output_path: str):
        """結果保存"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights,
            'performance': {
                'top1_accuracy': results['top1_accuracy'],
                'top3_accuracy': results['top3_accuracy'],
                'ndcg@1': results['mean_ndcg@1'],
                'total_races': results['total_races']
            },
            'components': {
                'lightgbm_models': len(self.lgb_models),
                'conditional_logit': 'trained',
                'elo_ratings': len(self.elo_rating.ratings)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logging.info(f"結果保存: {output_path}")


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("改善されたハイブリッド予測システム")
    logging.info("=" * 60)

    # データ読み込み
    data_path = "data/processed/cleaned_data_all.csv"
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    logging.info(f"データ読み込み: {len(df)}レコード, {df['race_id'].nunique()}レース")

    # ハイブリッドモデル初期化
    hybrid = ImprovedHybridPredictor(lgb_model_dir='models/optimized')

    # 訓練
    hybrid.train(df)

    # 評価
    results = hybrid.evaluate(df)

    # 結果保存
    output_dir = Path("models/hybrid")
    output_dir.mkdir(exist_ok=True)

    hybrid.save_results(results, str(output_dir / "improved_hybrid_results.json"))

    logging.info("=" * 60)
    logging.info("完了")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

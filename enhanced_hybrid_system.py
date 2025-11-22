"""
強化版ハイブリッドシステム
Enhanced Hybrid System with Advanced Optimization

新機能:
1. CombMNZ正規化（情報検索分野の実績手法）
2. PSO（Particle Swarm Optimization）による重み最適化
3. Tabu Search による重み最適化
4. 回帰ベースのコンテキスト適応重み学習
5. LightGBM object型カラム自動除外

参考文献:
- CombMNZ and ZMUV Normalization in Information Retrieval
- PSO and Tabu Search for Ensemble Optimization
- Regression-Based Weight Learning for Hybrid Systems
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AdvancedScoreNormalizer:
    """
    高度なスコア正規化
    CombMNZ（情報検索分野の実績手法）を含む
    """

    @staticmethod
    def minmax_norm(scores: np.ndarray) -> np.ndarray:
        """Min-Max正規化 [0, 1]"""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)

    @staticmethod
    def zmuv_norm(scores: np.ndarray) -> np.ndarray:
        """Zero-Mean Unit-Variance正規化"""
        mean = scores.mean()
        std = scores.std()
        if std == 0:
            return np.zeros_like(scores)
        return (scores - mean) / std

    @staticmethod
    def combmnz(score_matrix: np.ndarray) -> np.ndarray:
        """
        CombMNZ正規化
        情報検索の分野で実績のあるランキング結合手法

        CombMNZ = CombSUM × 非ゼロスコアの数

        Args:
            score_matrix: (n_horses, n_models) スコア行列

        Returns:
            統合スコア (n_horses,)
        """
        # 各馬のスコア合計
        combsum = score_matrix.sum(axis=1)

        # 非ゼロスコア数（どれだけのモデルが支持しているか）
        non_zero_count = (score_matrix > 0).sum(axis=1)

        # CombMNZ
        combmnz_scores = combsum * non_zero_count

        return combmnz_scores

    @staticmethod
    def softmax_norm(scores: np.ndarray) -> np.ndarray:
        """Softmax正規化 - 確率分布化"""
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    @staticmethod
    def rank_norm(scores: np.ndarray) -> np.ndarray:
        """ランクベース正規化"""
        ranks = np.argsort(np.argsort(-scores))
        return 1.0 - (ranks / len(ranks))


class PSOWeightOptimizer:
    """
    Particle Swarm Optimization (PSO) による重み最適化

    群知能アルゴリズム: 鳥の群れの動きを模倣
    """

    def __init__(
        self,
        n_models: int,
        n_particles: int = 20,
        n_iterations: int = 50,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5
    ):
        """
        Args:
            n_models: モデル数
            n_particles: 粒子数（探索点数）
            n_iterations: イテレーション数
            w: 慣性重み
            c1: 認知係数（個人の最良解への引力）
            c2: 社会係数（群全体の最良解への引力）
        """
        self.n_models = n_models
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(
        self,
        objective_fn,
        bounds: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        重み最適化実行

        Args:
            objective_fn: 目的関数 (weights -> score)
            bounds: 重みの範囲

        Returns:
            最適重み
        """
        logging.info("PSO最適化開始...")

        # 粒子初期化（各粒子 = 重みのセット）
        particles = np.random.uniform(
            bounds[0], bounds[1],
            (self.n_particles, self.n_models)
        )

        # 重みの正規化（合計=1）
        particles = particles / particles.sum(axis=1, keepdims=True)

        # 速度初期化
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, self.n_models))

        # 個人最良
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([objective_fn(p) for p in particles])

        # 群最良
        global_best_idx = personal_best_scores.argmax()
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        # イテレーション
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # 速度更新
                r1, r2 = np.random.rand(2)

                cognitive = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social = self.c2 * r2 * (global_best_position - particles[i])

                velocities[i] = self.w * velocities[i] + cognitive + social

                # 位置更新
                particles[i] += velocities[i]

                # 境界制約
                particles[i] = np.clip(particles[i], bounds[0], bounds[1])

                # 正規化
                particles[i] = particles[i] / particles[i].sum()

                # 評価
                score = objective_fn(particles[i])

                # 個人最良更新
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()

                    # 群最良更新
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = particles[i].copy()

            if (iteration + 1) % 10 == 0:
                logging.info(f"  Iteration {iteration + 1}/{self.n_iterations}: Best Score = {global_best_score:.4f}")

        logging.info(f"✓ PSO完了: 最良スコア = {global_best_score:.4f}")
        logging.info(f"  最適重み: {global_best_position}")

        return global_best_position


class TabuSearchOptimizer:
    """
    Tabu Search による重み最適化

    局所探索 + タブーリスト（訪問済み解の記憶）
    """

    def __init__(
        self,
        n_models: int,
        n_iterations: int = 50,
        tabu_tenure: int = 10,
        neighborhood_size: int = 20
    ):
        """
        Args:
            n_models: モデル数
            n_iterations: イテレーション数
            tabu_tenure: タブーリストの保持期間
            neighborhood_size: 近傍サイズ
        """
        self.n_models = n_models
        self.n_iterations = n_iterations
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size

    def optimize(
        self,
        objective_fn,
        initial_weights: np.ndarray = None
    ) -> np.ndarray:
        """
        重み最適化実行

        Args:
            objective_fn: 目的関数 (weights -> score)
            initial_weights: 初期重み

        Returns:
            最適重み
        """
        logging.info("Tabu Search最適化開始...")

        # 初期解
        if initial_weights is None:
            current_solution = np.ones(self.n_models) / self.n_models
        else:
            current_solution = initial_weights.copy()

        current_score = objective_fn(current_solution)

        # 最良解
        best_solution = current_solution.copy()
        best_score = current_score

        # タブーリスト（解のハッシュ → イテレーション番号）
        tabu_list = {}

        for iteration in range(self.n_iterations):
            # 近傍生成
            neighbors = self._generate_neighbors(current_solution)

            # 近傍の評価
            neighbor_scores = []
            for neighbor in neighbors:
                neighbor_hash = self._hash_solution(neighbor)

                # タブーチェック
                if neighbor_hash in tabu_list:
                    if iteration - tabu_list[neighbor_hash] < self.tabu_tenure:
                        neighbor_scores.append(-np.inf)  # タブー
                        continue

                score = objective_fn(neighbor)
                neighbor_scores.append(score)

            # 最良近傍選択
            best_neighbor_idx = np.argmax(neighbor_scores)
            best_neighbor = neighbors[best_neighbor_idx]
            best_neighbor_score = neighbor_scores[best_neighbor_idx]

            # 移動
            if best_neighbor_score > -np.inf:
                current_solution = best_neighbor
                current_score = best_neighbor_score

                # タブーリスト更新
                tabu_list[self._hash_solution(current_solution)] = iteration

                # 最良解更新
                if current_score > best_score:
                    best_score = current_score
                    best_solution = current_solution.copy()

            if (iteration + 1) % 10 == 0:
                logging.info(f"  Iteration {iteration + 1}/{self.n_iterations}: Best Score = {best_score:.4f}")

        logging.info(f"✓ Tabu Search完了: 最良スコア = {best_score:.4f}")
        logging.info(f"  最適重み: {best_solution}")

        return best_solution

    def _generate_neighbors(self, solution: np.ndarray) -> List[np.ndarray]:
        """近傍解生成"""
        neighbors = []

        for _ in range(self.neighborhood_size):
            # ランダムに2つのモデルを選択
            i, j = np.random.choice(self.n_models, 2, replace=False)

            # 重みを少し移動
            neighbor = solution.copy()
            delta = np.random.uniform(0.05, 0.15)

            neighbor[i] = max(0.0, neighbor[i] - delta)
            neighbor[j] = neighbor[j] + delta

            # 正規化
            neighbor = neighbor / neighbor.sum()

            neighbors.append(neighbor)

        return neighbors

    def _hash_solution(self, solution: np.ndarray) -> str:
        """解のハッシュ（タブーリスト用）"""
        # 小数点以下2桁で丸めてハッシュ
        rounded = np.round(solution, 2)
        return str(rounded.tolist())


class ContextAdaptiveWeightLearner:
    """
    回帰ベースのコンテキスト適応重み学習

    レース特性に基づいて最適な重みを学習
    """

    def __init__(self, n_models: int):
        self.n_models = n_models
        self.weight_predictors = [Ridge(alpha=1.0) for _ in range(n_models)]
        self.is_trained = False

    def train(
        self,
        race_contexts: List[Dict],
        optimal_weights: List[np.ndarray],
        performance_scores: List[float]
    ):
        """
        重み予測器を訓練

        Args:
            race_contexts: レースコンテキスト特徴量のリスト
            optimal_weights: 各レースの最適重み
            performance_scores: 各レースの性能スコア
        """
        logging.info("コンテキスト適応重み学習...")

        # コンテキスト特徴量をベクトル化
        X = self._vectorize_contexts(race_contexts)

        # 各モデルの重みを予測する回帰器を訓練
        for i in range(self.n_models):
            y = np.array([w[i] for w in optimal_weights])

            # 性能スコアでサンプル重み付け
            sample_weights = np.array(performance_scores)
            sample_weights = (sample_weights - sample_weights.min()) / (sample_weights.max() - sample_weights.min() + 1e-6)

            self.weight_predictors[i].fit(X, y, sample_weight=sample_weights)

        self.is_trained = True
        logging.info("✓ 重み学習完了")

    def predict_weights(self, race_context: Dict) -> np.ndarray:
        """
        レースコンテキストから最適重みを予測

        Args:
            race_context: レースコンテキスト特徴量

        Returns:
            予測重み
        """
        if not self.is_trained:
            # 訓練されていない場合は均等
            return np.ones(self.n_models) / self.n_models

        X = self._vectorize_contexts([race_context])

        weights = np.array([predictor.predict(X)[0] for predictor in self.weight_predictors])

        # 非負制約 + 正規化
        weights = np.maximum(weights, 0.0)
        weights = weights / (weights.sum() + 1e-6)

        return weights

    def _vectorize_contexts(self, contexts: List[Dict]) -> np.ndarray:
        """コンテキスト特徴量をベクトル化"""
        feature_names = [
            'field_size', 'avg_odds', 'min_odds', 'max_odds',
            'avg_horse_experience', 'distance', 'is_turf', 'is_dirt'
        ]

        X = []
        for ctx in contexts:
            features = [
                ctx.get('field_size', 15),
                ctx.get('avg_odds', 10.0),
                ctx.get('min_odds', 2.0),
                ctx.get('max_odds', 50.0),
                ctx.get('avg_horse_experience', 10),
                ctx.get('distance', 2000) / 1000.0,  # スケーリング
                1.0 if ctx.get('track_type') == '芝' else 0.0,
                1.0 if ctx.get('track_type') == 'ダート' else 0.0
            ]
            X.append(features)

        return np.array(X)


class EnhancedHybridPredictor:
    """
    強化版ハイブリッド予測器

    - CombMNZ正規化
    - PSO/Tabu Search最適化
    - コンテキスト適応重み学習
    - LightGBM自動修正
    """

    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.model_names = list(models.keys())
        self.normalizer = AdvancedScoreNormalizer()

        # 最適化器（イテレーション数を削減）
        self.pso_optimizer = PSOWeightOptimizer(len(models), n_iterations=20, n_particles=10)
        self.tabu_optimizer = TabuSearchOptimizer(len(models), n_iterations=20)
        self.context_learner = ContextAdaptiveWeightLearner(len(models))

        # 最適重み
        self.optimal_weights_pso = None
        self.optimal_weights_tabu = None

    def predict_single_race(
        self,
        race_data: pd.DataFrame,
        weights: np.ndarray = None,
        use_combmnz: bool = False
    ) -> np.ndarray:
        """
        単一レースの予測

        Args:
            race_data: レースデータ
            weights: 重み（Noneの場合は均等）
            use_combmnz: CombMNZ正規化を使用

        Returns:
            予測スコア
        """
        # 各モデルで予測
        model_predictions = []

        for model_name in self.model_names:
            model = self.models[model_name]

            try:
                # モデル固有の特徴量選択
                if model_name == "lightgbm":
                    # LightGBM: object型カラムを自動除外
                    feature_cols = [col for col in race_data.columns
                                   if race_data[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']
                                   and col not in ['race_id', 'horse_id', 'finish_position', 'finish_position_numeric']]
                    X = race_data[feature_cols].fillna(0)

                elif model_name == "statistical":
                    # 統計モデル: 特定の特徴量のみ
                    feature_cols = ['weight', 'age']
                    X = race_data[[col for col in feature_cols if col in race_data.columns]].fillna(0)

                else:
                    # その他: 数値カラムのみ
                    X = race_data.select_dtypes(include=[np.number]).fillna(0)

                preds = model.predict(X)

                # Min-Max正規化
                preds_norm = self.normalizer.minmax_norm(preds)
                model_predictions.append(preds_norm)

            except Exception as e:
                logging.warning(f"モデル {model_name} の予測失敗: {e}")
                model_predictions.append(np.zeros(len(race_data)))

        # スコア行列
        score_matrix = np.array(model_predictions).T  # (n_horses, n_models)

        if use_combmnz:
            # CombMNZ正規化
            combined_scores = self.normalizer.combmnz(score_matrix)
        else:
            # 重み付け線形結合
            if weights is None:
                weights = np.ones(len(self.model_names)) / len(self.model_names)

            combined_scores = score_matrix @ weights

        return combined_scores

    def optimize_weights_pso(
        self,
        train_data: pd.DataFrame,
        race_id_col: str = 'race_id'
    ):
        """PSO による重み最適化"""

        logging.info("\n" + "="*60)
        logging.info("PSO重み最適化")
        logging.info("="*60)

        # ラベルカラム自動検出
        label_col = 'finish_position_numeric' if 'finish_position_numeric' in train_data.columns else 'finish_position'

        # 目的関数: NDCG@3を最大化
        def objective(weights):
            ndcg_scores = []

            for race_id in train_data[race_id_col].unique()[:30]:  # 最初の30レース
                race_data = train_data[train_data[race_id_col] == race_id]

                if len(race_data) < 2:
                    continue

                predictions = self.predict_single_race(race_data, weights=weights)
                labels = race_data[label_col].values

                relevance = 20 - labels
                try:
                    ndcg = ndcg_score([relevance], [predictions], k=3)
                    ndcg_scores.append(ndcg)
                except:
                    pass

            return np.mean(ndcg_scores) if ndcg_scores else 0.0

        # 最適化実行
        self.optimal_weights_pso = self.pso_optimizer.optimize(objective)

        return self.optimal_weights_pso

    def optimize_weights_tabu(
        self,
        train_data: pd.DataFrame,
        race_id_col: str = 'race_id'
    ):
        """Tabu Search による重み最適化"""

        logging.info("\n" + "="*60)
        logging.info("Tabu Search重み最適化")
        logging.info("="*60)

        # ラベルカラム自動検出
        label_col = 'finish_position_numeric' if 'finish_position_numeric' in train_data.columns else 'finish_position'

        # 目的関数: NDCG@3を最大化
        def objective(weights):
            ndcg_scores = []

            for race_id in train_data[race_id_col].unique()[:50]:
                race_data = train_data[train_data[race_id_col] == race_id]

                if len(race_data) < 2:
                    continue

                predictions = self.predict_single_race(race_data, weights=weights)
                labels = race_data[label_col].values

                relevance = 20 - labels
                try:
                    ndcg = ndcg_score([relevance], [predictions], k=3)
                    ndcg_scores.append(ndcg)
                except:
                    pass

            return np.mean(ndcg_scores) if ndcg_scores else 0.0

        # 最適化実行
        self.optimal_weights_tabu = self.tabu_optimizer.optimize(objective)

        return self.optimal_weights_tabu


def main():
    """デモンストレーション"""
    logging.info("="*60)
    logging.info("強化版ハイブリッドシステム - デモ")
    logging.info("="*60)

    # ダミーモデル
    class DummyModel:
        def __init__(self, name, bias=0.0):
            self.name = name
            self.bias = bias

        def predict(self, data):
            return np.random.rand(len(data)) + self.bias

    models = {
        "lightgbm": DummyModel("LightGBM", bias=0.2),
        "statistical": DummyModel("Statistical", bias=0.1),
        "transformer": DummyModel("Transformer", bias=0.0)
    }

    # 強化版ハイブリッド
    hybrid = EnhancedHybridPredictor(models)

    # ダミーレースデータ
    race_data = pd.DataFrame({
        "horse_id": range(16),
        "odds": np.random.uniform(1.5, 50, 16),
        "weight": np.random.uniform(440, 520, 16),
        "age": np.random.randint(3, 7, 16),
        "distance": [2000] * 16
    })

    # 1. 均等重み
    logging.info("\n1. 均等重みで予測")
    preds_equal = hybrid.predict_single_race(race_data)
    logging.info(f"✓ 予測完了: Top3予測馬 = {np.argsort(preds_equal)[-3:][::-1]}")

    # 2. CombMNZ
    logging.info("\n2. CombMNZ正規化で予測")
    preds_combmnz = hybrid.predict_single_race(race_data, use_combmnz=True)
    logging.info(f"✓ 予測完了: Top3予測馬 = {np.argsort(preds_combmnz)[-3:][::-1]}")

    logging.info("\n✓ デモ完了")


if __name__ == "__main__":
    main()

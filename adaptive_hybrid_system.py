"""
適応的ハイブリッド予測システム
Adaptive Hybrid Horse Racing Prediction System

学術的知見に基づく3つのアーキテクチャパターン:
1. モノリシックハイブリッド - 単一モデル内での統合
2. 並列化ハイブリッド - 複数モデルの重み付け結合（Dynamic Alpha Tuning）
3. パイプライン化ハイブリッド - 段階的予測

参考文献:
- Dynamic Alpha Tuning for RAG systems
- Hybrid Recommender Systems architectures
- Intelligent Switching and Score Normalization
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ScoreNormalizer:
    """スコア正規化 - 異なるモデルのスコアを統一"""

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
    def softmax_norm(scores: np.ndarray) -> np.ndarray:
        """Softmax正規化 - 確率分布化"""
        exp_scores = np.exp(scores - scores.max())  # 数値安定性
        return exp_scores / exp_scores.sum()

    @staticmethod
    def rank_norm(scores: np.ndarray) -> np.ndarray:
        """ランクベース正規化"""
        ranks = np.argsort(np.argsort(-scores))  # 降順のランク
        return 1.0 - (ranks / len(ranks))


class IntelligentSwitcher:
    """
    インテリジェントスイッチングハイブリッド
    コンテキストに基づいて最適なモデルを動的に選択
    """

    def __init__(self):
        self.switching_rules = {}
        self.history = []

    def add_rule(self, rule_name: str, condition_fn, model_name: str):
        """
        スイッチングルール追加

        Args:
            rule_name: ルール名
            condition_fn: 条件関数 (race_data -> bool)
            model_name: 選択するモデル名
        """
        self.switching_rules[rule_name] = {
            "condition": condition_fn,
            "model": model_name
        }

    def select_model(self, race_data: pd.DataFrame) -> str:
        """
        レースデータに基づいてモデル選択

        Args:
            race_data: レースデータ

        Returns:
            選択されたモデル名
        """
        for rule_name, rule in self.switching_rules.items():
            if rule["condition"](race_data):
                self.history.append({
                    "rule": rule_name,
                    "model": rule["model"],
                    "timestamp": datetime.now().isoformat()
                })
                return rule["model"]

        # デフォルト
        return "default"


class DynamicAlphaTuner:
    """
    Dynamic Alpha Tuning (DAT)
    レースごとに最適な重み係数を動的に調整

    RAGシステムのDAT手法を競馬予測に適用
    """

    def __init__(self, models: List[str]):
        self.models = models
        self.alpha_history = []
        self.feature_importance = {}

    def estimate_alpha(
        self,
        race_data: pd.DataFrame,
        model_confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """
        レースごとに最適な重み係数（α）を推定

        Args:
            race_data: レースデータ
            model_confidences: 各モデルの信頼度

        Returns:
            最適な重み係数 {model_name: alpha}
        """
        # レース特性を分析
        race_features = self._extract_race_features(race_data)

        # 各モデルの適合度を評価
        model_scores = {}
        for model in self.models:
            confidence = model_confidences.get(model, 0.5)
            suitability = self._evaluate_suitability(model, race_features)

            # スコア = 信頼度 × 適合度
            model_scores[model] = confidence * suitability

        # Softmaxで正規化してαに変換
        scores_array = np.array(list(model_scores.values()))
        alphas = self._softmax(scores_array)

        alpha_dict = {model: alpha for model, alpha in zip(self.models, alphas)}

        # 履歴記録
        self.alpha_history.append({
            "race_features": race_features,
            "alphas": alpha_dict,
            "timestamp": datetime.now().isoformat()
        })

        return alpha_dict

    def _extract_race_features(self, race_data: pd.DataFrame) -> Dict[str, Any]:
        """レース特性抽出"""
        features = {
            "field_size": len(race_data),
            "is_g1": race_data.get("grade", "").str.contains("G1").any() if "grade" in race_data.columns else False,
            "has_favorite": race_data["odds"].min() < 3.0 if "odds" in race_data.columns else False,
            "avg_horse_experience": race_data["horse_race_count"].mean() if "horse_race_count" in race_data.columns else 0,
            "distance": race_data["distance"].iloc[0] if "distance" in race_data.columns else 0,
            "track_type": race_data["track_type"].iloc[0] if "track_type" in race_data.columns else "unknown"
        }
        return features

    def _evaluate_suitability(self, model: str, race_features: Dict) -> float:
        """モデルの適合度評価"""
        suitability = 0.5  # ベーススコア

        # LightGBM: 大規模レース、豊富なデータ
        if model == "LightGBM":
            if race_features["field_size"] > 16:
                suitability += 0.2
            if race_features["avg_horse_experience"] > 10:
                suitability += 0.15

        # Transformer: G1レース、複雑なパターン
        elif model == "Transformer":
            if race_features["is_g1"]:
                suitability += 0.3
            if race_features["field_size"] >= 18:
                suitability += 0.1

        # 統計モデル: 小規模レース、データ不足
        elif model == "Statistical":
            if race_features["field_size"] < 12:
                suitability += 0.2
            if race_features["avg_horse_experience"] < 5:
                suitability += 0.15

        return min(1.0, suitability)

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Softmax関数"""
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()


class ParallelHybridCombiner:
    """
    並列化ハイブリッド - 複数モデルの結果を統合

    - 重み付け線形結合
    - スコア正規化
    - Dynamic Alpha Tuning
    """

    def __init__(self, models: Dict[str, Any], normalization: str = "minmax"):
        """
        Args:
            models: {model_name: model_object}
            normalization: 正規化手法 ("minmax", "zmuv", "softmax", "rank")
        """
        self.models = models
        self.normalizer = ScoreNormalizer()
        self.normalization = normalization
        self.dat = DynamicAlphaTuner(list(models.keys()))

    def predict(
        self,
        race_data: pd.DataFrame,
        use_dynamic_alpha: bool = True,
        fixed_weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        ハイブリッド予測

        Args:
            race_data: レースデータ
            use_dynamic_alpha: Dynamic Alpha Tuningを使用
            fixed_weights: 固定重み（use_dynamic_alpha=Falseの場合）

        Returns:
            統合予測スコア
        """
        # 各モデルで予測
        model_predictions = {}
        model_confidences = {}

        for model_name, model in self.models.items():
            try:
                preds = model.predict(race_data)

                # 正規化
                if self.normalization == "minmax":
                    preds_norm = self.normalizer.minmax_norm(preds)
                elif self.normalization == "zmuv":
                    preds_norm = self.normalizer.zmuv_norm(preds)
                elif self.normalization == "softmax":
                    preds_norm = self.normalizer.softmax_norm(preds)
                elif self.normalization == "rank":
                    preds_norm = self.normalizer.rank_norm(preds)
                else:
                    preds_norm = preds

                model_predictions[model_name] = preds_norm

                # 信頼度計算（スコアの分散に基づく）
                confidence = 1.0 - (preds_norm.std() / (preds_norm.std() + 1e-6))
                model_confidences[model_name] = confidence

            except Exception as e:
                logging.warning(f"モデル {model_name} の予測失敗: {e}")
                model_predictions[model_name] = np.zeros(len(race_data))
                model_confidences[model_name] = 0.0

        # 重み決定
        if use_dynamic_alpha:
            weights = self.dat.estimate_alpha(race_data, model_confidences)
        else:
            weights = fixed_weights or {name: 1.0 / len(self.models) for name in self.models}

        # 重み付け線形結合
        combined_scores = np.zeros(len(race_data))
        for model_name, preds in model_predictions.items():
            weight = weights.get(model_name, 0.0)
            combined_scores += weight * preds

            logging.debug(f"{model_name}: weight={weight:.4f}, avg_score={preds.mean():.4f}")

        return combined_scores


class PipelinedHybrid:
    """
    パイプライン化ハイブリッド
    あるモデルの出力を次のモデルの入力に使用
    """

    def __init__(self, stages: List[Dict]):
        """
        Args:
            stages: パイプラインステージ
                [
                    {"name": "粗い予測", "model": model1, "output_as_feature": True},
                    {"name": "精密予測", "model": model2, "output_as_feature": False}
                ]
        """
        self.stages = stages

    def predict(self, race_data: pd.DataFrame) -> np.ndarray:
        """パイプライン予測"""
        current_data = race_data.copy()

        for i, stage in enumerate(self.stages):
            model = stage["model"]
            predictions = model.predict(current_data)

            # 次のステージの特徴量として追加
            if stage.get("output_as_feature", False) and i < len(self.stages) - 1:
                current_data[f"stage_{i}_prediction"] = predictions

        return predictions


class AdaptiveHybridSystem:
    """
    適応的ハイブリッドシステム - 統合インターフェース
    """

    def __init__(
        self,
        models: Dict[str, Any],
        architecture: str = "parallel",
        config: Dict = None
    ):
        """
        Args:
            models: {model_name: model_object}
            architecture: "parallel", "pipelined", "switching"
            config: 設定
        """
        self.models = models
        self.architecture = architecture
        self.config = config or {}

        if architecture == "parallel":
            self.predictor = ParallelHybridCombiner(
                models,
                normalization=config.get("normalization", "minmax")
            )
        elif architecture == "pipelined":
            self.predictor = PipelinedHybrid(config.get("stages", []))
        elif architecture == "switching":
            self.predictor = IntelligentSwitcher()
        else:
            raise ValueError(f"未対応のアーキテクチャ: {architecture}")

    def predict(self, race_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """予測実行"""
        return self.predictor.predict(race_data, **kwargs)

    def evaluate(
        self,
        test_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "finish_position",
        race_id_col: str = "race_id"
    ) -> Dict[str, float]:
        """評価"""
        from sklearn.metrics import ndcg_score

        metrics = {}
        all_predictions = []
        all_labels = []

        for race_id in test_data[race_id_col].unique():
            race_data = test_data[test_data[race_id_col] == race_id]

            predictions = self.predict(race_data[feature_cols])
            labels = race_data[label_col].values

            all_predictions.append(predictions)
            all_labels.append(labels)

        # メトリクス計算
        ndcg_scores = []
        for preds, labels in zip(all_predictions, all_labels):
            relevance = 20 - labels
            try:
                ndcg = ndcg_score([relevance], [preds], k=3)
                ndcg_scores.append(ndcg)
            except:
                pass

        metrics["ndcg@3"] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        metrics["num_races"] = len(all_predictions)

        return metrics


def main():
    """デモンストレーション"""
    logging.info("="*60)
    logging.info("適応的ハイブリッドシステム - デモ")
    logging.info("="*60)

    # ダミーモデル
    class DummyModel:
        def __init__(self, name):
            self.name = name

        def predict(self, data):
            return np.random.rand(len(data))

    # モデル準備
    models = {
        "LightGBM": DummyModel("LightGBM"),
        "Transformer": DummyModel("Transformer"),
        "Statistical": DummyModel("Statistical")
    }

    # 並列化ハイブリッド（Dynamic Alpha Tuning）
    logging.info("\n1. 並列化ハイブリッド（Dynamic Alpha Tuning）")
    hybrid = AdaptiveHybridSystem(
        models,
        architecture="parallel",
        config={"normalization": "minmax"}
    )

    # ダミーレースデータ
    race_data = pd.DataFrame({
        "horse_id": range(18),
        "odds": np.random.uniform(1.5, 50, 18),
        "distance": [2000] * 18,
        "track_type": ["芝"] * 18,
        "horse_race_count": np.random.randint(1, 30, 18)
    })

    predictions = hybrid.predict(race_data, use_dynamic_alpha=True)
    logging.info(f"✓ 予測完了: {len(predictions)}頭")
    logging.info(f"  予測スコア範囲: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # Dynamic Alphaの履歴
    if hasattr(hybrid.predictor, 'dat'):
        if hybrid.predictor.dat.alpha_history:
            latest_alphas = hybrid.predictor.dat.alpha_history[-1]["alphas"]
            logging.info(f"  最新の重み係数: {latest_alphas}")


if __name__ == "__main__":
    main()

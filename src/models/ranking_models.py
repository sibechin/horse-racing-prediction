"""
ランキング学習モデル（LambdaRank, LambdaMART）
競馬の着順予測に最適化されたモデル
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import yaml
from pathlib import Path
import joblib

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import ndcg_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LambdaRankModel:
    """
    LightGBM LambdaRankモデル
    ランキング学習による着順予測に特化
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.feature_names = None

        # LambdaRankのパラメータ
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],  # 1位、3位以内、5位以内の精度
            'max_depth': 8,
            'learning_rate': 0.01,
            'num_leaves': 64,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'force_col_wise': True,
            'label_gain': [0, 7, 5, 3, 2, 1.5, 1, 0.5, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0],  # 上位ほど重要
        }

    def prepare_ranking_data(self, X: pd.DataFrame, y: pd.Series,
                            race_ids: pd.Series) -> Tuple[lgb.Dataset, np.ndarray]:
        """
        ランキング学習用のデータを準備

        Args:
            X: 特徴量
            y: ターゲット（着順）
            race_ids: レースID

        Returns:
            LightGBM Dataset と グループサイズの配列
        """
        # レースIDでソート（重要：同じレースの馬を連続させる）
        sort_idx = np.argsort(race_ids.values)
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]
        race_ids_sorted = race_ids.iloc[sort_idx]

        # 各レースの出走頭数を計算（グループサイズ）
        unique_races, group_sizes = np.unique(race_ids_sorted, return_counts=True)

        # LightGBM Datasetの作成
        # ラベルは小さいほど良い（1位=1, 2位=2...）のでそのまま使用
        dataset = lgb.Dataset(
            X_sorted,
            label=y_sorted.values,
            group=group_sizes
        )

        self.feature_names = X.columns.tolist()

        return dataset, group_sizes

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             race_ids_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None,
             race_ids_val: pd.Series = None) -> Dict:
        """
        モデルを訓練

        Args:
            X_train: 訓練データの特徴量
            y_train: 訓練データのターゲット（着順）
            race_ids_train: 訓練データのレースID
            X_val: 検証データの特徴量
            y_val: 検証データのターゲット
            race_ids_val: 検証データのレースID

        Returns:
            訓練履歴
        """
        logger.info("Training LambdaRank model")

        # 訓練データの準備
        train_data, train_groups = self.prepare_ranking_data(
            X_train, y_train, race_ids_train
        )

        # 検証データの準備
        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None and race_ids_val is not None:
            val_data, val_groups = self.prepare_ranking_data(
                X_val, y_val, race_ids_val
            )
            valid_sets.append(val_data)
            valid_names.append('valid')

        # モデル訓練
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=3000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # 特徴量重要度
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importance(importance_type='gain')
        ))

        logger.info("LambdaRank training completed")

        return {'feature_importance': feature_importance}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        着順を予測

        Args:
            X: 特徴量

        Returns:
            予測スコア（スコアが低いほど上位予想）
        """
        # LambdaRankはスコアを出力（スコアが高いほど上位）
        scores = self.model.predict(X)
        return scores

    def predict_ranking(self, X: pd.DataFrame, race_ids: pd.Series) -> pd.DataFrame:
        """
        レースごとに着順を予測

        Args:
            X: 特徴量
            race_ids: レースID

        Returns:
            予測結果のDataFrame（レースID、予測順位、予測スコア）
        """
        # スコア予測
        scores = self.predict(X)

        # 結果をDataFrameに
        results = pd.DataFrame({
            'race_id': race_ids,
            'score': scores
        })

        # レースごとにスコアでランキング（スコアが高いほど上位）
        results['predicted_rank'] = results.groupby('race_id')['score'].rank(
            ascending=False, method='first'
        ).astype(int)

        return results

    def save_model(self, filepath: str):
        """モデルを保存"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """モデルを読み込み"""
        self.model = lgb.Booster(model_file=filepath)
        logger.info(f"Model loaded from {filepath}")


class XGBoostRankModel:
    """
    XGBoost ランキングモデル
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.feature_names = None

        # XGBoost rank:pairwise パラメータ
        self.params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg',
            'max_depth': 8,
            'eta': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist'
        }

    def prepare_ranking_data(self, X: pd.DataFrame, y: pd.Series,
                            race_ids: pd.Series) -> Tuple[xgb.DMatrix, np.ndarray]:
        """ランキング学習用データを準備"""
        # レースIDでソート
        sort_idx = np.argsort(race_ids.values)
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]
        race_ids_sorted = race_ids.iloc[sort_idx]

        # グループサイズ計算
        unique_races, group_sizes = np.unique(race_ids_sorted, return_counts=True)

        # DMatrix作成
        dtrain = xgb.DMatrix(X_sorted, label=y_sorted.values)
        dtrain.set_group(group_sizes)

        self.feature_names = X.columns.tolist()

        return dtrain, group_sizes

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             race_ids_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None,
             race_ids_val: pd.Series = None) -> Dict:
        """モデル訓練"""
        logger.info("Training XGBoost Rank model")

        dtrain, train_groups = self.prepare_ranking_data(
            X_train, y_train, race_ids_train
        )

        evals = [(dtrain, 'train')]

        if X_val is not None:
            dval, val_groups = self.prepare_ranking_data(
                X_val, y_val, race_ids_val
            )
            evals.append((dval, 'valid'))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=3000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=100
        )

        feature_importance = self.model.get_score(importance_type='gain')

        logger.info("XGBoost Rank training completed")

        return {'feature_importance': feature_importance}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """スコア予測"""
        dtest = xgb.DMatrix(X)
        scores = self.model.predict(dtest)
        return scores

    def predict_ranking(self, X: pd.DataFrame, race_ids: pd.Series) -> pd.DataFrame:
        """レースごとに着順予測"""
        scores = self.predict(X)

        results = pd.DataFrame({
            'race_id': race_ids,
            'score': scores
        })

        results['predicted_rank'] = results.groupby('race_id')['score'].rank(
            ascending=False, method='first'
        ).astype(int)

        return results

    def save_model(self, filepath: str):
        """モデル保存"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """モデル読み込み"""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class RankingEvaluator:
    """ランキングモデルの評価"""

    @staticmethod
    def evaluate_ranking(y_true: np.ndarray, y_pred_scores: np.ndarray,
                        race_ids: np.ndarray) -> Dict:
        """
        ランキング性能を評価

        Args:
            y_true: 実際の着順
            y_pred_scores: 予測スコア
            race_ids: レースID

        Returns:
            評価指標の辞書
        """
        metrics = {}

        # レースごとに評価
        unique_races = np.unique(race_ids)
        ndcg_scores_at_1 = []
        ndcg_scores_at_3 = []
        ndcg_scores_at_5 = []
        hit_rate_at_1 = []
        hit_rate_at_3 = []

        for race_id in unique_races:
            mask = race_ids == race_id

            race_true = y_true[mask]
            race_scores = y_pred_scores[mask]

            # 予測順位を計算
            pred_ranks = (-race_scores).argsort().argsort() + 1

            # NDCG計算（真の順位を関連性スコアとして使用）
            # 1位=最高スコア、下位=低スコア
            relevance_scores = 1.0 / race_true  # 1位=1.0, 2位=0.5, 3位=0.33...

            # NDCGを計算
            try:
                ndcg_1 = ndcg_score([relevance_scores], [race_scores], k=1)
                ndcg_3 = ndcg_score([relevance_scores], [race_scores], k=3)
                ndcg_5 = ndcg_score([relevance_scores], [race_scores], k=5)

                ndcg_scores_at_1.append(ndcg_1)
                ndcg_scores_at_3.append(ndcg_3)
                ndcg_scores_at_5.append(ndcg_5)
            except:
                pass

            # Hit Rate（実際の上位馬を予測できたか）
            actual_top1 = np.where(race_true == 1)[0]
            actual_top3 = np.where(race_true <= 3)[0]

            pred_top1 = np.where(pred_ranks == 1)[0]
            pred_top3 = np.where(pred_ranks <= 3)[0]

            # 1位的中
            if len(actual_top1) > 0 and len(pred_top1) > 0:
                hit_rate_at_1.append(int(actual_top1[0] == pred_top1[0]))

            # 3着以内的中
            hit_3 = len(set(actual_top3) & set(pred_top3)) / min(3, len(actual_top3))
            hit_rate_at_3.append(hit_3)

        # 平均値を計算
        metrics['ndcg@1'] = np.mean(ndcg_scores_at_1) if ndcg_scores_at_1 else 0
        metrics['ndcg@3'] = np.mean(ndcg_scores_at_3) if ndcg_scores_at_3 else 0
        metrics['ndcg@5'] = np.mean(ndcg_scores_at_5) if ndcg_scores_at_5 else 0
        metrics['hit_rate@1'] = np.mean(hit_rate_at_1) if hit_rate_at_1 else 0
        metrics['hit_rate@3'] = np.mean(hit_rate_at_3) if hit_rate_at_3 else 0

        logger.info(f"NDCG@1: {metrics['ndcg@1']:.4f}")
        logger.info(f"NDCG@3: {metrics['ndcg@3']:.4f}")
        logger.info(f"Hit Rate@1: {metrics['hit_rate@1']:.4f}")
        logger.info(f"Hit Rate@3: {metrics['hit_rate@3']:.4f}")

        return metrics


if __name__ == "__main__":
    # 使用例
    # ダミーデータ生成
    np.random.seed(42)

    n_races = 100
    horses_per_race = 10
    n_samples = n_races * horses_per_race

    # レースIDを生成
    race_ids = np.repeat(np.arange(n_races), horses_per_race)

    # ダミー特徴量
    X = pd.DataFrame(np.random.randn(n_samples, 20))

    # ダミー着順（1-10位）
    y = np.tile(np.arange(1, horses_per_race + 1), n_races)
    y = pd.Series(y)
    race_ids = pd.Series(race_ids)

    # 訓練・検証分割
    train_idx = race_ids < 80
    val_idx = race_ids >= 80

    # LambdaRankモデル
    model = LambdaRankModel()
    history = model.train(
        X[train_idx], y[train_idx], race_ids[train_idx],
        X[val_idx], y[val_idx], race_ids[val_idx]
    )

    # 予測
    predictions = model.predict_ranking(X[val_idx], race_ids[val_idx])
    print("\n予測結果（サンプル）:")
    print(predictions.head(20))

    # 評価
    evaluator = RankingEvaluator()
    metrics = evaluator.evaluate_ranking(
        y[val_idx].values,
        predictions['score'].values,
        race_ids[val_idx].values
    )
    print("\n評価指標:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

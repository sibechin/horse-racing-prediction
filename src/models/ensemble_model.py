"""
ハイブリッドアンサンブルモデル
統計モデルとディープラーニングモデルを組み合わせる
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import yaml
from pathlib import Path
import joblib

from .statistical_models import StatisticalModel
from .deep_learning_models import DeepLearningTrainer, RaceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """アンサンブルモデル"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.ensemble_config = self.config['models']['ensemble']
        self.method = self.ensemble_config['method']
        self.weights = self.ensemble_config['weights']

        self.models = {}
        self.meta_model = None

    def add_model(self, name: str, model):
        """
        モデルを追加

        Args:
            name: モデル名
            model: モデルオブジェクト
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")

    def weighted_average_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        重み付き平均による予測

        Args:
            X: 特徴量

        Returns:
            予測された着順
        """
        predictions = []
        model_weights = []

        for name, model in self.models.items():
            if name in self.weights:
                # 確率予測を取得
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    # RaceDatasetに変換してから予測
                    if isinstance(model, DeepLearningTrainer):
                        dataset = RaceDataset(
                            X.values,
                            np.zeros(len(X), dtype=int)  # ダミーターゲット
                        )
                        proba = model.predict_proba(dataset)
                    else:
                        proba = model.predict_proba(X)

                predictions.append(proba)
                model_weights.append(self.weights[name])

        # 重み付き平均
        predictions = np.array(predictions)
        model_weights = np.array(model_weights) / np.sum(model_weights)

        weighted_proba = np.zeros_like(predictions[0])
        for i, weight in enumerate(model_weights):
            weighted_proba += predictions[i] * weight

        # 最も確率が高い着順を返す
        return np.argmax(weighted_proba, axis=1) + 1

    def weighted_average_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        重み付き平均による確率予測

        Args:
            X: 特徴量

        Returns:
            各着順の確率分布
        """
        predictions = []
        model_weights = []

        for name, model in self.models.items():
            if name in self.weights:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    if isinstance(model, DeepLearningTrainer):
                        dataset = RaceDataset(
                            X.values,
                            np.zeros(len(X), dtype=int)
                        )
                        proba = model.predict_proba(dataset)
                    else:
                        proba = model.predict_proba(X)

                predictions.append(proba)
                model_weights.append(self.weights[name])

        # 重み付き平均
        predictions = np.array(predictions)
        model_weights = np.array(model_weights) / np.sum(model_weights)

        weighted_proba = np.zeros_like(predictions[0])
        for i, weight in enumerate(model_weights):
            weighted_proba += predictions[i] * weight

        return weighted_proba

    def stacking_train(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series):
        """
        スタッキングによるメタモデルの訓練

        Args:
            X_train: 訓練データの特徴量
            y_train: 訓練データのターゲット
            X_val: 検証データの特徴量
            y_val: 検証データのターゲット
        """
        from sklearn.linear_model import LogisticRegression

        logger.info("Training stacking meta-model")

        # 各ベースモデルの予測を取得
        train_meta_features = []
        val_meta_features = []

        for name, model in self.models.items():
            logger.info(f"Getting predictions from {name}")

            if hasattr(model, 'predict_proba'):
                train_proba = model.predict_proba(X_train)
                val_proba = model.predict_proba(X_val)
            else:
                if isinstance(model, DeepLearningTrainer):
                    train_dataset = RaceDataset(X_train.values, y_train.values - 1)
                    val_dataset = RaceDataset(X_val.values, y_val.values - 1)

                    train_proba = model.predict_proba(train_dataset)
                    val_proba = model.predict_proba(val_dataset)
                else:
                    train_proba = model.predict_proba(X_train)
                    val_proba = model.predict_proba(X_val)

            train_meta_features.append(train_proba)
            val_meta_features.append(val_proba)

        # メタ特徴量を結合
        X_train_meta = np.hstack(train_meta_features)
        X_val_meta = np.hstack(val_meta_features)

        # メタモデルの訓練（ロジスティック回帰）
        self.meta_model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )

        # 着順を0始まりに変換
        y_train_idx = (y_train - 1).astype(int)
        y_train_idx = np.clip(y_train_idx, 0, 17)

        self.meta_model.fit(X_train_meta, y_train_idx)

        logger.info("Stacking meta-model training completed")

    def stacking_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        スタッキングによる予測

        Args:
            X: 特徴量

        Returns:
            予測された着順
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call stacking_train first.")

        # 各ベースモデルの予測を取得
        meta_features = []

        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                if isinstance(model, DeepLearningTrainer):
                    dataset = RaceDataset(
                        X.values,
                        np.zeros(len(X), dtype=int)
                    )
                    proba = model.predict_proba(dataset)
                else:
                    proba = model.predict_proba(X)

            meta_features.append(proba)

        # メタ特徴量を結合
        X_meta = np.hstack(meta_features)

        # メタモデルで予測
        predictions = self.meta_model.predict(X_meta)

        return predictions + 1  # 0始まりから1始まりに変換

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        設定に基づいて予測

        Args:
            X: 特徴量

        Returns:
            予測された着順
        """
        if self.method == 'weighted_average':
            return self.weighted_average_predict(X)
        elif self.method == 'stacking':
            return self.stacking_predict(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        設定に基づいて確率予測

        Args:
            X: 特徴量

        Returns:
            各着順の確率分布
        """
        if self.method == 'weighted_average':
            return self.weighted_average_predict_proba(X)
        elif self.method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Meta-model not trained.")

            # メタ特徴量を取得
            meta_features = []
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                else:
                    if isinstance(model, DeepLearningTrainer):
                        dataset = RaceDataset(X.values, np.zeros(len(X), dtype=int))
                        proba = model.predict_proba(dataset)
                    else:
                        proba = model.predict_proba(X)
                meta_features.append(proba)

            X_meta = np.hstack(meta_features)
            return self.meta_model.predict_proba(X_meta)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def save_ensemble(self, directory: str):
        """
        アンサンブルモデルを保存

        Args:
            directory: 保存ディレクトリ
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 設定を保存
        config_data = {
            'method': self.method,
            'weights': self.weights,
            'model_names': list(self.models.keys())
        }

        joblib.dump(config_data, dir_path / 'ensemble_config.pkl')

        # メタモデルを保存（スタッキングの場合）
        if self.meta_model is not None:
            joblib.dump(self.meta_model, dir_path / 'meta_model.pkl')

        logger.info(f"Ensemble model saved to {directory}")

    def load_ensemble(self, directory: str):
        """
        アンサンブルモデルを読み込み

        Args:
            directory: 読み込みディレクトリ
        """
        dir_path = Path(directory)

        # 設定を読み込み
        config_data = joblib.load(dir_path / 'ensemble_config.pkl')
        self.method = config_data['method']
        self.weights = config_data['weights']

        # メタモデルを読み込み（スタッキングの場合）
        meta_model_path = dir_path / 'meta_model.pkl'
        if meta_model_path.exists():
            self.meta_model = joblib.load(meta_model_path)

        logger.info(f"Ensemble model loaded from {directory}")


if __name__ == "__main__":
    # 使用例
    from .statistical_models import create_statistical_model

    # データ読み込み
    train = pd.read_csv("../../data/processed/train_features.csv")
    val = pd.read_csv("../../data/processed/val_features.csv")

    feature_cols = [col for col in train.columns if col not in
                   ['race_id', 'horse_id', 'finish_position', 'race_date']]

    X_train = train[feature_cols]
    y_train = train['finish_position']
    X_val = val[feature_cols]
    y_val = val['finish_position']

    # 個別モデルの訓練
    xgb_model = create_statistical_model('xgboost')
    xgb_model.train(X_train, y_train, X_val, y_val)

    lgb_model = create_statistical_model('lightgbm')
    lgb_model.train(X_train, y_train, X_val, y_val)

    # アンサンブルモデルの作成
    ensemble = EnsembleModel()
    ensemble.add_model('xgboost', xgb_model)
    ensemble.add_model('lightgbm', lgb_model)

    # 予測
    predictions = ensemble.predict(X_val)
    print(f"Ensemble predictions: {predictions[:10]}")

    # 保存
    ensemble.save_ensemble("../../data/models/ensemble")

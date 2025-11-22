"""
統計的機械学習モデル（XGBoost, LightGBM, CatBoost）
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import yaml
from pathlib import Path
import joblib

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalModel:
    """統計的機械学習モデルの基底クラス"""

    def __init__(self, config_path: str = "configs/config.yaml", model_type: str = "xgboost"):
        """
        Args:
            config_path: 設定ファイルのパス
            model_type: モデルタイプ ("xgboost", "lightgbm", "catboost")
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.num_classes = 18  # 着順は通常1-18位程度

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """
        モデルを訓練

        Args:
            X_train: 訓練データの特徴量
            y_train: 訓練データのターゲット（着順）
            X_val: 検証データの特徴量
            y_val: 検証データのターゲット

        Returns:
            訓練履歴の辞書
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        着順を予測

        Args:
            X: 特徴量

        Returns:
            予測された着順
        """
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        各着順の確率を予測

        Args:
            X: 特徴量

        Returns:
            各着順の確率分布
        """
        raise NotImplementedError

    def save_model(self, filepath: str):
        """モデルを保存"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """モデルを読み込み"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        logger.info(f"Model loaded from {filepath}")


class XGBoostModel(StatisticalModel):
    """XGBoostモデル"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path, model_type="xgboost")
        self.params = self.config['models']['statistical']['xgboost']

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """XGBoostモデルを訓練"""
        logger.info("Training XGBoost model")

        self.feature_names = X_train.columns.tolist()

        # 着順を0始まりのインデックスに変換（1位→0, 2位→1, ...）
        y_train_idx = (y_train - 1).astype(int)
        y_train_idx = np.clip(y_train_idx, 0, self.num_classes - 1)

        # DMatrixの作成
        dtrain = xgb.DMatrix(X_train, label=y_train_idx)

        # パラメータ設定
        params = {
            'objective': 'multi:softprob',
            'num_class': self.num_classes,
            'eval_metric': 'mlogloss',
            'max_depth': self.params['max_depth'],
            'learning_rate': self.params['learning_rate'],
            'subsample': self.params['subsample'],
            'colsample_bytree': self.params['colsample_bytree'],
            'tree_method': 'hist',
            'random_state': 42
        }

        # Early stopping用の検証セット
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            y_val_idx = (y_val - 1).astype(int)
            y_val_idx = np.clip(y_val_idx, 0, self.num_classes - 1)
            dval = xgb.DMatrix(X_val, label=y_val_idx)
            evals.append((dval, 'val'))

        # モデル訓練
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )

        # 特徴量重要度を取得
        feature_importance = self.model.get_score(importance_type='gain')

        logger.info("XGBoost training completed")

        return {'feature_importance': feature_importance}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """着順を予測"""
        proba = self.predict_proba(X)
        # 最も確率が高い着順を返す（0始まりなので+1）
        return np.argmax(proba, axis=1) + 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """各着順の確率を予測"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


class LightGBMModel(StatisticalModel):
    """LightGBMモデル"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path, model_type="lightgbm")
        self.params = self.config['models']['statistical']['lightgbm']

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """LightGBMモデルを訓練"""
        logger.info("Training LightGBM model")

        self.feature_names = X_train.columns.tolist()

        # 着順を0始まりのインデックスに変換
        y_train_idx = (y_train - 1).astype(int)
        y_train_idx = np.clip(y_train_idx, 0, self.num_classes - 1)

        # Datasetの作成
        train_data = lgb.Dataset(X_train, label=y_train_idx)

        # パラメータ設定
        params = {
            'objective': 'multiclass',
            'num_class': self.num_classes,
            'metric': 'multi_logloss',
            'max_depth': self.params['max_depth'],
            'learning_rate': self.params['learning_rate'],
            'num_leaves': self.params['num_leaves'],
            'verbosity': 1,
            'random_state': 42
        }

        # Early stopping用の検証セット
        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            y_val_idx = (y_val - 1).astype(int)
            y_val_idx = np.clip(y_val_idx, 0, self.num_classes - 1)
            val_data = lgb.Dataset(X_val, label=y_val_idx)
            valid_sets.append(val_data)
            valid_names.append('val')

        # モデル訓練
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.params['n_estimators'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # 特徴量重要度を取得
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importance(importance_type='gain')
        ))

        logger.info("LightGBM training completed")

        return {'feature_importance': feature_importance}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """着順を予測"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) + 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """各着順の確率を予測"""
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class CatBoostModel(StatisticalModel):
    """CatBoostモデル"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path, model_type="catboost")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """CatBoostモデルを訓練"""
        logger.info("Training CatBoost model")

        self.feature_names = X_train.columns.tolist()

        # 着順を0始まりのインデックスに変換
        y_train_idx = (y_train - 1).astype(int)
        y_train_idx = np.clip(y_train_idx, 0, self.num_classes - 1)

        # プールの作成
        train_pool = Pool(X_train, y_train_idx)

        # モデルの初期化
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.01,
            depth=8,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50
        )

        # 検証セット
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_idx = (y_val - 1).astype(int)
            y_val_idx = np.clip(y_val_idx, 0, self.num_classes - 1)
            val_pool = Pool(X_val, y_val_idx)
            eval_set = val_pool

        # モデル訓練
        self.model.fit(train_pool, eval_set=eval_set)

        # 特徴量重要度を取得
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        logger.info("CatBoost training completed")

        return {'feature_importance': feature_importance}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """着順を予測"""
        predictions = self.model.predict(X)
        return predictions + 1  # 0始まりから1始まりに変換

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """各着順の確率を予測"""
        return self.model.predict_proba(X)


def create_statistical_model(model_type: str, config_path: str = "configs/config.yaml") -> StatisticalModel:
    """
    統計モデルを作成

    Args:
        model_type: モデルタイプ ("xgboost", "lightgbm", "catboost")
        config_path: 設定ファイルのパス

    Returns:
        StatisticalModelのインスタンス
    """
    models = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](config_path)


if __name__ == "__main__":
    # 使用例
    # データ読み込み
    train = pd.read_csv("../../data/processed/train_features.csv")
    val = pd.read_csv("../../data/processed/val_features.csv")

    # 特徴量とターゲットを分離
    feature_cols = [col for col in train.columns if col not in
                   ['race_id', 'horse_id', 'finish_position', 'race_date']]

    X_train = train[feature_cols]
    y_train = train['finish_position']
    X_val = val[feature_cols]
    y_val = val['finish_position']

    # XGBoostモデルの訓練
    xgb_model = create_statistical_model('xgboost')
    xgb_model.train(X_train, y_train, X_val, y_val)
    xgb_model.save_model("../../data/models/xgboost_model.pkl")

    # LightGBMモデルの訓練
    lgb_model = create_statistical_model('lightgbm')
    lgb_model.train(X_train, y_train, X_val, y_val)
    lgb_model.save_model("../../data/models/lightgbm_model.pkl")

    # 予測
    predictions = xgb_model.predict(X_val)
    print(f"Predictions: {predictions[:10]}")

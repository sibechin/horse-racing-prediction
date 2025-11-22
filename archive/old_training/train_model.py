"""
競馬予測モデルの訓練メインスクリプト
"""
import sys
from pathlib import Path
import logging
import argparse

# プロジェクトのルートをパスに追加
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np

from src.preprocessing.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.models.statistical_models import create_statistical_model
from src.models.deep_learning_models import LSTMModel, TransformerModel, DeepLearningTrainer, RaceDataset
from src.models.ensemble_model import EnsembleModel
from src.evaluation.evaluator import RacePredictorEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """メイン処理"""
    logger.info("="*60)
    logger.info("競馬予測モデル訓練開始")
    logger.info("="*60)

    # 1. データの読み込み
    logger.info("\n[ステップ1] データの読み込み")
    raw_data_path = Path("data/raw/all_races.csv")

    if not raw_data_path.exists():
        logger.error(f"データファイルが見つかりません: {raw_data_path}")
        logger.info("まず src/data_collection/netkeiba_scraper.py を実行してデータを収集してください")
        return

    raw_data = pd.read_csv(raw_data_path)
    logger.info(f"データ読み込み完了: {raw_data.shape}")

    # 2. データクリーニング
    if not args.skip_preprocessing:
        logger.info("\n[ステップ2] データクリーニング")
        cleaner = DataCleaner()
        clean_data = cleaner.clean_race_data(raw_data)

        # データ分割
        train_df, val_df, test_df = cleaner.split_data(clean_data)
        cleaner.save_cleaned_data(train_df, val_df, test_df, "data/processed")
    else:
        logger.info("\n[ステップ2] クリーニング済みデータの読み込み")
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        test_df = pd.read_csv("data/processed/test.csv")

    # 3. 特徴量エンジニアリング
    if not args.skip_feature_engineering:
        logger.info("\n[ステップ3] 特徴量エンジニアリング")
        engineer = FeatureEngineer()

        train_feat = engineer.create_features(train_df)
        val_feat = engineer.create_features(val_df)
        test_feat = engineer.create_features(test_df)

        # エンコーディング
        train_enc, val_enc, test_enc = engineer.encode_categorical_features(
            train_feat, val_feat, test_feat
        )

        # スケーリング
        train_scaled, val_scaled, test_scaled = engineer.scale_features(
            train_enc, val_enc, test_enc
        )

        # 保存
        train_scaled.to_csv("data/processed/train_features.csv", index=False)
        val_scaled.to_csv("data/processed/val_features.csv", index=False)
        test_scaled.to_csv("data/processed/test_features.csv", index=False)

        # 特徴量選択
        feature_cols = engineer.select_features(train_scaled)
    else:
        logger.info("\n[ステップ3] 特徴量データの読み込み")
        train_scaled = pd.read_csv("data/processed/train_features.csv")
        val_scaled = pd.read_csv("data/processed/val_features.csv")
        test_scaled = pd.read_csv("data/processed/test_features.csv")

        engineer = FeatureEngineer()
        feature_cols = engineer.select_features(train_scaled)

    # 特徴量とターゲットの分離
    X_train = train_scaled[feature_cols]
    y_train = train_scaled['finish_position']
    X_val = val_scaled[feature_cols]
    y_val = val_scaled['finish_position']
    X_test = test_scaled[feature_cols]
    y_test = test_scaled['finish_position']

    logger.info(f"特徴量数: {len(feature_cols)}")

    # 4. モデル訓練
    logger.info("\n[ステップ4] モデル訓練")
    models = {}

    # 統計モデルの訓練
    if args.train_statistical:
        logger.info("\n--- 統計モデルの訓練 ---")

        # XGBoost
        if 'xgboost' in args.models or 'all' in args.models:
            logger.info("XGBoostモデルを訓練中...")
            xgb_model = create_statistical_model('xgboost')
            xgb_model.train(X_train, y_train, X_val, y_val)
            xgb_model.save_model("data/models/xgboost_model.pkl")
            models['xgboost'] = xgb_model
            logger.info("XGBoost訓練完了")

        # LightGBM
        if 'lightgbm' in args.models or 'all' in args.models:
            logger.info("LightGBMモデルを訓練中...")
            lgb_model = create_statistical_model('lightgbm')
            lgb_model.train(X_train, y_train, X_val, y_val)
            lgb_model.save_model("data/models/lightgbm_model.pkl")
            models['lightgbm'] = lgb_model
            logger.info("LightGBM訓練完了")

    # ディープラーニングモデルの訓練
    if args.train_deep_learning:
        logger.info("\n--- ディープラーニングモデルの訓練 ---")

        # データセットの準備
        train_dataset = RaceDataset(
            X_train.values,
            (y_train - 1).values.astype(int)  # 0始まりに変換
        )
        val_dataset = RaceDataset(
            X_val.values,
            (y_val - 1).values.astype(int)
        )

        # LSTM
        if 'lstm' in args.models or 'all' in args.models:
            logger.info("LSTMモデルを訓練中...")
            lstm_model = LSTMModel(
                input_size=len(feature_cols),
                hidden_size=128,
                num_classes=18
            )
            lstm_trainer = DeepLearningTrainer(lstm_model)
            lstm_trainer.train(train_dataset, val_dataset)
            lstm_trainer.save_checkpoint("data/models/lstm_model.pth")
            models['lstm'] = lstm_trainer
            logger.info("LSTM訓練完了")

    # 5. アンサンブルモデルの作成
    if args.train_ensemble and len(models) > 1:
        logger.info("\n[ステップ5] アンサンブルモデルの作成")
        ensemble = EnsembleModel()

        for name, model in models.items():
            ensemble.add_model(name, model)

        # スタッキングの場合はメタモデルを訓練
        if ensemble.method == 'stacking':
            logger.info("スタッキングメタモデルを訓練中...")
            ensemble.stacking_train(X_train, y_train, X_val, y_val)

        ensemble.save_ensemble("data/models/ensemble")
        logger.info("アンサンブルモデル作成完了")

        # アンサンブルで予測
        y_pred_ensemble = ensemble.predict(X_test)
        models['ensemble'] = ensemble
    else:
        y_pred_ensemble = None

    # 6. モデル評価
    logger.info("\n[ステップ6] モデル評価")
    evaluator = RacePredictorEvaluator(output_dir="logs")

    for model_name, model in models.items():
        logger.info(f"\n--- {model_name}の評価 ---")

        # 予測
        if model_name == 'ensemble':
            y_pred = y_pred_ensemble
        elif hasattr(model, 'predict'):
            if isinstance(model, DeepLearningTrainer):
                test_dataset = RaceDataset(X_test.values, (y_test - 1).values.astype(int))
                y_pred = model.predict(test_dataset)
            else:
                y_pred = model.predict(X_test)
        else:
            continue

        # 評価レポート生成
        report = evaluator.generate_report(
            y_test.values,
            y_pred,
            model_name=model_name
        )
        print(report)

    logger.info("\n="*60)
    logger.info("すべての処理が完了しました！")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬予測モデルの訓練")

    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        choices=['all', 'xgboost', 'lightgbm', 'lstm', 'transformer'],
        help="訓練するモデル"
    )

    parser.add_argument(
        '--train-statistical',
        action='store_true',
        default=True,
        help="統計モデルを訓練"
    )

    parser.add_argument(
        '--train-deep-learning',
        action='store_true',
        help="ディープラーニングモデルを訓練"
    )

    parser.add_argument(
        '--train-ensemble',
        action='store_true',
        default=True,
        help="アンサンブルモデルを作成"
    )

    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help="前処理をスキップ（既存データを使用）"
    )

    parser.add_argument(
        '--skip-feature-engineering',
        action='store_true',
        help="特徴量エンジニアリングをスキップ"
    )

    args = parser.parse_args()

    main(args)

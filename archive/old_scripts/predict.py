"""
競馬予測の実行スクリプト
訓練済みモデルを使って新しいレースの予測を行う
"""
import sys
from pathlib import Path
import logging
import argparse
import joblib

sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np

from src.preprocessing.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.models.statistical_models import create_statistical_model
from src.models.ensemble_model import EnsembleModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, model_type: str = 'ensemble'):
    """
    訓練済みモデルを読み込む

    Args:
        model_path: モデルファイルのパス
        model_type: モデルタイプ

    Returns:
        読み込まれたモデル
    """
    if model_type == 'ensemble':
        ensemble = EnsembleModel()
        ensemble.load_ensemble(model_path)
        return ensemble
    else:
        model = create_statistical_model(model_type)
        model.load_model(model_path)
        return model


def preprocess_race_data(race_data: pd.DataFrame, engineer: FeatureEngineer) -> pd.DataFrame:
    """
    レースデータを前処理

    Args:
        race_data: 生のレースデータ
        engineer: FeatureEngineerインスタンス

    Returns:
        前処理済みデータ
    """
    # データクリーニング
    cleaner = DataCleaner()
    clean_data = cleaner.clean_race_data(race_data)

    # 特徴量作成
    features = engineer.create_features(clean_data)

    return features


def predict_race(model, race_data: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    レースの着順を予測

    Args:
        model: 訓練済みモデル
        race_data: レースデータ
        feature_cols: 使用する特徴量のカラム

    Returns:
        予測結果のDataFrame
    """
    # 特徴量を抽出
    X = race_data[feature_cols]

    # 予測
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # 結果をDataFrameに整形
    results = race_data[['horse_name', 'jockey_name']].copy()
    results['predicted_position'] = predictions
    results['win_probability'] = probabilities[:, 0]  # 1位の確率

    # 予測着順でソート
    results = results.sort_values('predicted_position')

    return results


def main(args):
    """メイン処理"""
    logger.info("="*60)
    logger.info("競馬予測実行")
    logger.info("="*60)

    # 1. モデルの読み込み
    logger.info(f"\n[ステップ1] モデルの読み込み: {args.model_path}")
    model = load_model(args.model_path, args.model_type)
    logger.info("モデル読み込み完了")

    # 2. レースデータの読み込み
    logger.info(f"\n[ステップ2] レースデータの読み込み: {args.race_data}")
    race_data = pd.read_csv(args.race_data)
    logger.info(f"レースデータ読み込み完了: {len(race_data)}頭立て")

    # 3. データの前処理
    logger.info("\n[ステップ3] データの前処理")
    engineer = FeatureEngineer()
    processed_data = preprocess_race_data(race_data, engineer)

    # 特徴量の選択
    feature_cols = engineer.select_features(processed_data)

    # 4. 予測の実行
    logger.info("\n[ステップ4] 予測の実行")
    results = predict_race(model, processed_data, feature_cols)

    # 5. 結果の表示
    logger.info("\n[ステップ5] 予測結果")
    logger.info("="*60)
    print("\n予測結果:")
    print(results.to_string(index=False))
    logger.info("="*60)

    # 6. 結果の保存
    if args.output:
        results.to_csv(args.output, index=False, encoding='utf-8-sig')
        logger.info(f"\n予測結果を保存しました: {args.output}")

    # 推奨馬券の提案
    logger.info("\n推奨馬券:")
    top3 = results.head(3)
    logger.info(f"単勝: {top3.iloc[0]['horse_name']} (確率: {top3.iloc[0]['win_probability']:.2%})")
    logger.info(f"馬連: {top3.iloc[0]['horse_name']} - {top3.iloc[1]['horse_name']}")
    logger.info(f"3連複: {top3.iloc[0]['horse_name']} - {top3.iloc[1]['horse_name']} - {top3.iloc[2]['horse_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬予測の実行")

    parser.add_argument(
        '--race-data',
        type=str,
        required=True,
        help="予測対象のレースデータ（CSVファイル）"
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='data/models/ensemble',
        help="使用するモデルのパス"
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='ensemble',
        choices=['ensemble', 'xgboost', 'lightgbm', 'catboost'],
        help="モデルタイプ"
    )

    parser.add_argument(
        '--output',
        type=str,
        help="予測結果の保存先（CSVファイル）"
    )

    args = parser.parse_args()

    main(args)

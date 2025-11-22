"""
全モデルの性能比較

- LightGBM (Top3 no odds)
- Conditional Logit (with odds)
- Transformer + Attention (TOP1特化)
- Dynamic Hybrid Fusion (CombMNZ)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import lightgbm as lgb
import joblib
import torch
from transformer_top1_model import TransformerTop1Trainer
from dynamic_hybrid_fusion import DynamicHybridFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_model(predictions_dict, df_test):
    """
    モデル評価

    Args:
        predictions_dict: {race_id: predictions} の辞書
        df_test: テストデータ

    Returns:
        評価メトリクス
    """
    top1_correct = 0
    top3_hit = 0
    total_races = 0

    for race_id, predictions in predictions_dict.items():
        race_data = df_test[df_test['race_id'] == race_id]

        if len(predictions) != len(race_data):
            continue

        # 実際の着順
        if 'finish_position_numeric' in race_data.columns:
            actual_positions = race_data['finish_position_numeric'].values
        else:
            actual_positions = pd.to_numeric(race_data['finish_position'], errors='coerce').values

        # Top1
        pred_winner = np.argmax(predictions)
        actual_winner = np.argmin(actual_positions)

        if pred_winner == actual_winner:
            top1_correct += 1

        # Top3
        if len(predictions) >= 3:
            pred_top3 = set(np.argsort(predictions)[-3:])
            actual_top3 = set(np.argsort(actual_positions)[:3])

            if len(pred_top3 & actual_top3) > 0:
                top3_hit += 1

        total_races += 1

    return {
        'top1_accuracy': top1_correct / total_races if total_races > 0 else 0,
        'top3_hit_rate': top3_hit / total_races if total_races > 0 else 0,
        'total_races': total_races
    }


def predict_lightgbm(model_path, df_test):
    """LightGBM予測"""
    logging.info("\n" + "="*60)
    logging.info("LightGBM (Top3 no odds) 予測")
    logging.info("="*60)

    model = lgb.Booster(model_file=str(model_path))

    # 特徴量
    feature_cols = [
        'year', 'month', 'day_of_week', 'venue_code', 'race_number',
        'sex_encoded', 'age', 'weight', 'last_3f', 'horse_weight_kg',
        'horse_weight_change', 'field_size', 'race_avg_odds', 'race_avg_weight',
        'trainer_encoded', 'trainer_frequency', 'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]

    predictions_dict = {}

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]

        available_features = [col for col in feature_cols if col in race_data.columns]
        X = race_data[available_features]

        try:
            predictions = model.predict(X)
            predictions_dict[race_id] = predictions
        except Exception as e:
            logging.warning(f"予測エラー (race {race_id}): {e}")

    metrics = evaluate_model(predictions_dict, df_test)
    logging.info(f"Top1精度: {metrics['top1_accuracy']:.2%}")
    logging.info(f"Top3適中率: {metrics['top3_hit_rate']:.2%}")
    logging.info(f"評価レース数: {metrics['total_races']}")

    return metrics


def predict_conditional_logit(model_path, df_test):
    """Conditional Logit予測"""
    logging.info("\n" + "="*60)
    logging.info("Conditional Logit (with odds) 予測")
    logging.info("="*60)

    model_dict = joblib.load(model_path)
    model = model_dict.get('model')
    scaler = model_dict.get('scaler')
    feature_cols = model_dict.get('feature_cols', [])

    predictions_dict = {}

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]

        available_features = [col for col in feature_cols if col in race_data.columns]
        X = race_data[available_features]

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        try:
            X_scaled = scaler.transform(X)
            predictions = model.predict_proba(X_scaled)[:, 1]
            predictions_dict[race_id] = predictions
        except Exception as e:
            logging.warning(f"予測エラー (race {race_id}): {e}")

    metrics = evaluate_model(predictions_dict, df_test)
    logging.info(f"Top1精度: {metrics['top1_accuracy']:.2%}")
    logging.info(f"Top3適中率: {metrics['top3_hit_rate']:.2%}")
    logging.info(f"評価レース数: {metrics['total_races']}")

    return metrics


def predict_transformer(model_path, df_test):
    """Transformer予測"""
    logging.info("\n" + "="*60)
    logging.info("Transformer + Attention 予測")
    logging.info("="*60)

    trainer = TransformerTop1Trainer.load(model_path)

    predictions_dict = {}

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]

        try:
            predictions = trainer.predict(race_data)
            predictions_dict[race_id] = predictions
        except Exception as e:
            logging.warning(f"予測エラー (race {race_id}): {e}")

    metrics = evaluate_model(predictions_dict, df_test)
    logging.info(f"Top1精度: {metrics['top1_accuracy']:.2%}")
    logging.info(f"Top3適中率: {metrics['top3_hit_rate']:.2%}")
    logging.info(f"評価レース数: {metrics['total_races']}")

    return metrics


def predict_hybrid_fusion(path_no_odds, path_with_odds, df_train, df_test):
    """Dynamic Hybrid Fusion予測"""
    logging.info("\n" + "="*60)
    logging.info("Dynamic Hybrid Fusion (CombMNZ) 予測")
    logging.info("="*60)

    system = DynamicHybridFusion()
    system.load_models(path_no_odds, path_with_odds)
    system.train_dat(df_train)

    predictions_dict = {}

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]

        try:
            predictions = system.predict(race_data, fusion_method='combmnz')
            predictions_dict[race_id] = predictions
        except Exception as e:
            logging.warning(f"予測エラー (race {race_id}): {e}")

    metrics = evaluate_model(predictions_dict, df_test)
    logging.info(f"Top1精度: {metrics['top1_accuracy']:.2%}")
    logging.info(f"Top3適中率: {metrics['top3_hit_rate']:.2%}")
    logging.info(f"評価レース数: {metrics['total_races']}")

    return metrics


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("全モデル性能比較")
    logging.info("="*60)

    # データ読み込み
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")

    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')

    # finish_position_numericの確認
    for df in [df_train, df_test]:
        if 'finish_position_numeric' not in df.columns:
            df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

    logging.info(f"\n訓練データ: {len(df_train)} レコード, {df_train['race_id'].nunique()} レース")
    logging.info(f"テストデータ: {len(df_test)} レコード, {df_test['race_id'].nunique()} レース")

    # モデルパス
    lightgbm_path = Path("models/top3_no_odds/top3_no_odds_model_20251121_053558.txt")
    logit_path = Path("models/statistical/conditional_logit_20251121_054350.pkl")

    # Transformerモデル（最新）
    transformer_dir = Path("models/transformer_top1")
    transformer_files = list(transformer_dir.glob("transformer_top1_*.pth"))
    transformer_path = max(transformer_files, key=lambda p: p.stat().st_mtime) if transformer_files else None

    # 結果格納
    results = {}

    # 1. LightGBM
    if lightgbm_path.exists():
        results['LightGBM (no odds)'] = predict_lightgbm(lightgbm_path, df_test)
    else:
        logging.warning(f"LightGBMモデルが見つかりません: {lightgbm_path}")

    # 2. Conditional Logit
    if logit_path.exists():
        results['Conditional Logit (with odds)'] = predict_conditional_logit(logit_path, df_test)
    else:
        logging.warning(f"Conditional Logitモデルが見つかりません: {logit_path}")

    # 3. Transformer
    if transformer_path and transformer_path.exists():
        results['Transformer + Attention'] = predict_transformer(transformer_path, df_test)
    else:
        logging.warning("Transformerモデルが見つかりません")

    # 4. Hybrid Fusion
    if lightgbm_path.exists() and logit_path.exists():
        results['Hybrid Fusion (CombMNZ)'] = predict_hybrid_fusion(
            lightgbm_path, logit_path, df_train, df_test
        )
    else:
        logging.warning("Hybrid Fusionに必要なモデルが見つかりません")

    # 結果比較
    logging.info("\n" + "="*60)
    logging.info("結果比較")
    logging.info("="*60)

    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = 'モデル'

    print("\n", comparison_df.to_string())

    # 結果保存
    output_dir = Path("models/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"model_comparison_{timestamp}.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n✓ 結果保存: {results_path}")

    # ベストモデル
    best_top1_model = max(results.items(), key=lambda x: x[1]['top1_accuracy'])
    best_top3_model = max(results.items(), key=lambda x: x[1]['top3_hit_rate'])

    logging.info(f"\nベストTop1モデル: {best_top1_model[0]} ({best_top1_model[1]['top1_accuracy']:.2%})")
    logging.info(f"ベストTop3モデル: {best_top3_model[0]} ({best_top3_model[1]['top3_hit_rate']:.2%})")

    logging.info("\n完了!")


if __name__ == "__main__":
    main()

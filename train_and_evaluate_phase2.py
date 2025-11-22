"""
Phase 2: 新しいモデルを訓練して2025年データで評価
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_ranking_model(y_true, y_pred, race_groups):
    """ランキングモデルの評価"""
    metrics = {}

    # NDCG Score
    ndcg_scores = []
    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() > 1:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]
            race_y_true_rel = race_y_true - 1

            try:
                ndcg = ndcg_score([race_y_true_rel], [race_y_pred], k=3)
                ndcg_scores.append(ndcg)
            except:
                pass

    metrics['ndcg@3'] = np.mean(ndcg_scores) if ndcg_scores else 0.0

    # Top1 Accuracy
    top1_correct = 0
    total_races = 0

    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() > 0:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]

            predicted_winner_idx = np.argmax(race_y_pred)
            actual_winner_idx = np.argmin(race_y_true)

            if predicted_winner_idx == actual_winner_idx:
                top1_correct += 1

            total_races += 1

    metrics['top1_accuracy'] = top1_correct / total_races if total_races > 0 else 0.0

    # Top3 Hit Rate
    top3_correct = 0
    total_races = 0

    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() >= 3:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]

            predicted_top3 = set(np.argsort(race_y_pred)[-3:])
            actual_top3 = set(np.argsort(race_y_true)[:3])

            if len(predicted_top3 & actual_top3) > 0:
                top3_correct += 1

            total_races += 1

    metrics['top3_hit_rate'] = top3_correct / total_races if total_races > 0 else 0.0

    return metrics


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("Phase 2: モデル訓練と2025年データ評価")
    logging.info("="*60)

    # 2025年データを訓練とテストに分割
    test_file = 'data/processed/keibalab_g1_2025_processed.csv'

    if not Path(test_file).exists():
        logging.error(f"テストデータが見つかりません: {test_file}")
        return

    df_2025 = pd.read_csv(test_file, encoding='utf-8-sig')
    logging.info(f"\n2025年データ: {len(df_2025)} records, {df_2025['race_id'].nunique()} races")

    # 特徴量
    feature_columns = [
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight', 'popularity', 'odds_numeric',
        'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size', 'race_avg_odds', 'race_avg_weight',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]

    available_features = [col for col in feature_columns if col in df_2025.columns]
    logging.info(f"使用する特徴量: {len(available_features)}個")
    logging.info(f"特徴量: {available_features}")

    # 2025年データを前半（訓練）と後半（テスト）に分割
    races = sorted(df_2025['expected_date'].unique())
    split_idx = len(races) * 2 // 3
    train_races = races[:split_idx]
    test_races = races[split_idx:]

    train_mask = df_2025['expected_date'].isin(train_races)
    test_mask = df_2025['expected_date'].isin(test_races)

    X_train = df_2025.loc[train_mask, available_features]
    # 回帰モデルのために順位を負の値に（スコアが高いほど順位が良い）
    y_train_rank = df_2025.loc[train_mask, 'finish_position_numeric']
    y_train = -y_train_rank  # 1位=-1, 2位=-2, ...
    race_groups_train = df_2025.loc[train_mask, 'race_id']

    X_test = df_2025.loc[test_mask, available_features]
    y_test_rank = df_2025.loc[test_mask, 'finish_position_numeric']
    y_test = -y_test_rank  # 1位=-1, 2位=-2, ...
    race_groups_test = df_2025.loc[test_mask, 'race_id']

    logging.info(f"\n訓練データ: {len(X_train)} records, {race_groups_train.nunique()} races")
    logging.info(f"テストデータ: {len(X_test)} records, {race_groups_test.nunique()} races")

    # LightGBMデータセット
    train_data = lgb.Dataset(
        X_train,
        label=y_train
    )

    # LightGBMパラメータ（回帰）
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 8,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    logging.info("\nモデル訓練中...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data],
        valid_names=['train']
    )

    # 訓練データでの評価（元の順位データを使用）
    y_pred_train = model.predict(X_train)
    train_metrics = evaluate_ranking_model(y_train_rank, y_pred_train, race_groups_train)

    logging.info("\n訓練データ性能:")
    logging.info(f"  NDCG@3: {train_metrics['ndcg@3']:.4f}")
    logging.info(f"  1着的中率: {train_metrics['top1_accuracy']*100:.2f}%")
    logging.info(f"  3着内的中率: {train_metrics['top3_hit_rate']*100:.2f}%")

    # テストデータでの評価（元の順位データを使用）
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate_ranking_model(y_test_rank, y_pred_test, race_groups_test)

    logging.info("\nテストデータ性能 (2025年後半のG1レース):")
    logging.info(f"  NDCG@3: {test_metrics['ndcg@3']:.4f}")
    logging.info(f"  1着的中率: {test_metrics['top1_accuracy']*100:.2f}%")
    logging.info(f"  3着内的中率: {test_metrics['top3_hit_rate']*100:.2f}%")

    # ランダム予測との比較
    avg_field_size = df_2025.loc[test_mask].groupby('race_id').size().mean()
    random_top1 = (1 / avg_field_size) * 100

    logging.info(f"\nランダム予測（ベースライン）:")
    logging.info(f"  1着的中率: {random_top1:.2f}%")
    logging.info(f"  平均出走頭数: {avg_field_size:.1f}")

    # モデル保存
    output_dir = Path('models/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = output_dir / f'lgbm_2025_model_{timestamp}.txt'
    model.save_model(str(model_file))
    logging.info(f"\nモデル保存: {model_file}")

    # 結果をファイルに保存
    results_file = output_dir / f'phase2_results_{timestamp}.txt'

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Phase 2: 2025年G1データ評価結果\n")
        f.write("="*60 + "\n\n")
        f.write(f"訓練データ: {len(X_train)} records, {race_groups_train.nunique()} races\n")
        f.write(f"テストデータ: {len(X_test)} records, {race_groups_test.nunique()} races\n\n")

        f.write("訓練データ性能:\n")
        f.write(f"  NDCG@3: {train_metrics['ndcg@3']:.4f}\n")
        f.write(f"  1着的中率: {train_metrics['top1_accuracy']*100:.2f}%\n")
        f.write(f"  3着内的中率: {train_metrics['top3_hit_rate']*100:.2f}%\n\n")

        f.write("テストデータ性能 (2025年後半のG1レース):\n")
        f.write(f"  NDCG@3: {test_metrics['ndcg@3']:.4f}\n")
        f.write(f"  1着的中率: {test_metrics['top1_accuracy']*100:.2f}%\n")
        f.write(f"  3着内的中率: {test_metrics['top3_hit_rate']*100:.2f}%\n\n")

        f.write(f"ランダム予測（ベースライン）:\n")
        f.write(f"  1着的中率: {random_top1:.2f}%\n")
        f.write(f"  平均出走頭数: {avg_field_size:.1f}\n\n")

        f.write("分析:\n")
        improvement = test_metrics['top1_accuracy'] * 100 - random_top1
        f.write(f"  ランダム予測との差: {improvement:+.2f}%\n")

        if test_metrics['top1_accuracy'] > (1 / avg_field_size):
            f.write("  ✓ モデルはランダム予測を上回っています\n")
        else:
            f.write("  ✗ モデルはランダム予測を下回っています（過学習の可能性）\n")

    logging.info(f"結果保存: {results_file}")

    logging.info("\n" + "="*60)
    logging.info("Phase 2評価完了!")
    logging.info("="*60)


if __name__ == "__main__":
    main()

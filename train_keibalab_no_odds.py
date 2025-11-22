"""
keibalab G1データでモデル訓練（オッズ除外版）
2000-2024年で訓練、2025年でテスト
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import joblib

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


def train_no_odds_model(train_file_2000_2024: str, test_file_2025: str, output_dir: str):
    """
    オッズを除外してモデル訓練

    Args:
        train_file_2000_2024: 訓練データ（2000-2024年）
        test_file_2025: テストデータ（2025年）
        output_dir: モデル保存ディレクトリ
    """
    logging.info("="*60)
    logging.info("オッズ除外モデル訓練開始")
    logging.info("="*60)

    # データ読み込み
    df_train = pd.read_csv(train_file_2000_2024, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file_2025, encoding='utf-8-sig')

    logging.info(f"\n訓練データ: {len(df_train)} records, {df_train['race_id'].nunique()} races")
    logging.info(f"テストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races")

    # 特徴量（オッズと人気を除外）
    feature_columns = [
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight',
        # 'popularity',  # 除外
        # 'odds_numeric',  # 除外
        'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size',
        # 'race_avg_odds',  # 除外
        'race_avg_weight',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]

    # 存在する特徴量のみ使用
    available_features = [col for col in feature_columns if col in df_train.columns and col in df_test.columns]
    logging.info(f"\n使用する特徴量: {len(available_features)}個")
    logging.info(f"除外した特徴量: popularity, odds_numeric, race_avg_odds")

    X_train = df_train[available_features].copy()
    y_train = df_train['finish_position_numeric'].copy()
    race_groups_train = df_train['race_id'].copy()

    X_test = df_test[available_features].copy()
    y_test = df_test['finish_position_numeric'].copy()
    race_groups_test = df_test['race_id'].copy()

    logging.info(f"\n訓練データ: {len(X_train)} records, {race_groups_train.nunique()} races")
    logging.info(f"テストデータ: {len(X_test)} records, {race_groups_test.nunique()} races")

    # LightGBMデータセット
    train_data = lgb.Dataset(
        X_train, label=y_train,
        group=race_groups_train.value_counts().sort_index().values
    )

    # パラメータ
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }

    # モデル訓練
    logging.info("\nモデル訓練中...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=50)]
    )

    # 予測
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 評価
    train_metrics = evaluate_ranking_model(y_train, y_pred_train, race_groups_train)
    test_metrics = evaluate_ranking_model(y_test, y_pred_test, race_groups_test)

    logging.info("\n訓練データ評価:")
    for metric, value in train_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    logging.info("\nテストデータ評価 (2025年):")
    for metric, value in test_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    # 特徴量重要度
    logging.info("\n特徴量重要度 Top 10:")
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_file = output_path / f'no_odds_model_{timestamp}.pkl'
    joblib.dump(model, model_file)
    logging.info(f"\nモデル保存: {model_file}")

    results_file = output_path / f'no_odds_results_{timestamp}.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("オッズ除外モデル訓練結果\n")
        f.write("="*60 + "\n\n")
        f.write(f"訓練データ: 2000-2024年 ({len(X_train)} records)\n")
        f.write(f"テストデータ: 2025年 ({len(X_test)} records)\n\n")
        f.write("訓練データ評価:\n")
        for metric, value in train_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nテストデータ評価:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

    logging.info(f"評価結果保存: {results_file}")

    logging.info("\n" + "="*60)
    logging.info("訓練完了!")
    logging.info("="*60)
    logging.info(f"\n2025年テストデータ性能:")
    logging.info(f"  1着的中率: {test_metrics['top1_accuracy']:.2%}")
    logging.info(f"  3着内的中率: {test_metrics['top3_hit_rate']:.2%}")
    logging.info(f"  NDCG@3: {test_metrics['ndcg@3']:.4f}")

    return model, test_metrics


def main():
    """メイン実行"""
    # 2025年のデータファイルを探す
    data_dir = Path('data/raw/keibalab')
    g1_2025_files = list(data_dir.glob('g1_2025_*.csv'))

    if not g1_2025_files:
        logging.error("2025年のデータファイルが見つかりません")
        logging.error("先に collect_keibalab_2025_g1.py を実行してください")
        return

    # 最新の2025年データを使用
    test_file_2025 = sorted(g1_2025_files)[-1]

    # 前処理済み2000-2024年データ
    train_file = 'data/processed/keibalab_g1_2000_2024_processed.csv'

    if not Path(train_file).exists():
        logging.error(f"訓練データが見つかりません: {train_file}")
        logging.error("先に preprocess_keibalab_g1.py を実行してください")
        return

    # 2025年データも前処理が必要
    # 簡易的に同じ前処理を適用
    logging.info("2025年データを前処理中...")
    from preprocess_keibalab_g1 import preprocess_keibalab_g1

    processed_2025_file = 'data/processed/keibalab_g1_2025_processed.csv'
    preprocess_keibalab_g1(str(test_file_2025), processed_2025_file)

    # モデル訓練
    train_no_odds_model(train_file, processed_2025_file, 'models/keibalab_g1_no_odds')


if __name__ == "__main__":
    main()

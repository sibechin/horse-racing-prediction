"""
keibalab G1データで機械学習モデルを訓練
LightGBMを使用したランキング予測モデル
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score, roc_auc_score, accuracy_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_ranking_model(y_true, y_pred, race_groups):
    """
    ランキングモデルの評価

    Args:
        y_true: 実際のfinish_position
        y_pred: 予測スコア
        race_groups: レースグループ

    Returns:
        評価指標の辞書
    """
    metrics = {}

    # NDCG Score (レース単位)
    ndcg_scores = []
    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() > 1:  # 少なくとも2頭以上
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]

            # NDCG計算用に順位を相対化（1着=0, 2着=1, ...）
            race_y_true_rel = race_y_true - 1

            try:
                ndcg = ndcg_score([race_y_true_rel], [race_y_pred], k=3)
                ndcg_scores.append(ndcg)
            except:
                pass

    metrics['ndcg@3'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    metrics['ndcg@3_std'] = np.std(ndcg_scores) if ndcg_scores else 0.0

    # Top1 Accuracy (1着的中率)
    top1_correct = 0
    total_races = 0

    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() > 0:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]

            predicted_winner = np.argmax(race_y_pred)
            actual_winner = np.argmin(race_y_true)  # 1着=1が最小

            if predicted_winner == actual_winner:
                top1_correct += 1

            total_races += 1

    metrics['top1_accuracy'] = top1_correct / total_races if total_races > 0 else 0.0

    # Top3 Accuracy (3着内的中率)
    top3_correct = 0

    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() >= 3:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]

            # 予測Top3
            predicted_top3 = np.argsort(race_y_pred)[-3:]

            # 実際のTop3
            actual_top3 = np.argsort(race_y_true)[:3]

            # いずれかが的中しているか
            if len(set(predicted_top3) & set(actual_top3)) > 0:
                top3_correct += 1

            total_races += 1

    metrics['top3_hit_rate'] = top3_correct / total_races if total_races > 0 else 0.0

    return metrics


def train_keibalab_g1_model(input_file: str, output_dir: str):
    """
    keibalab G1データでモデルを訓練

    Args:
        input_file: 前処理済みデータファイル
        output_dir: モデル保存ディレクトリ
    """
    logging.info("="*60)
    logging.info("keibalab G1モデル訓練開始")
    logging.info("="*60)

    # データ読み込み
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    logging.info(f"\nデータ読み込み: {len(df)} records, {df['race_id'].nunique()} races")
    logging.info(f"期間: {df['year'].min()}-{df['year'].max()}年")

    # 特徴量とターゲットの分離
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

    # 存在する特徴量のみ使用
    available_features = [col for col in feature_columns if col in df.columns]
    logging.info(f"\n使用する特徴量: {len(available_features)}個")

    X = df[available_features].copy()
    y = df['finish_position_numeric'].copy()
    race_groups = df['race_id'].copy()

    # 時系列分割（2000-2022年を訓練、2023年を検証、2024年をテスト）
    train_mask = df['year'] <= 2022
    val_mask = df['year'] == 2023
    test_mask = df['year'] == 2024

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    race_groups_train = race_groups[train_mask]
    race_groups_val = race_groups[val_mask]
    race_groups_test = race_groups[test_mask]

    logging.info("\nデータ分割:")
    logging.info(f"  訓練データ (2000-2022): {len(X_train)} records, {race_groups_train.nunique()} races")
    logging.info(f"  検証データ (2023): {len(X_val)} records, {race_groups_val.nunique()} races")
    logging.info(f"  テストデータ (2024): {len(X_test)} records, {race_groups_test.nunique()} races")

    # LightGBMデータセットの作成
    train_data = lgb.Dataset(
        X_train, label=y_train,
        group=race_groups_train.value_counts().sort_index().values
    )

    val_data = lgb.Dataset(
        X_val, label=y_val,
        group=race_groups_val.value_counts().sort_index().values,
        reference=train_data
    )

    # LightGBMパラメータ（ランキングモデル）
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
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )

    logging.info(f"\nBest iteration: {model.best_iteration}")

    # 予測
    logging.info("\n予測実行...")
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    # 評価
    logging.info("\n評価...")

    # 訓練データ
    train_metrics = evaluate_ranking_model(y_train, y_pred_train, race_groups_train)
    logging.info("\n訓練データ評価:")
    for metric, value in train_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    # 検証データ
    val_metrics = evaluate_ranking_model(y_val, y_pred_val, race_groups_val)
    logging.info("\n検証データ評価:")
    for metric, value in val_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    # テストデータ
    test_metrics = evaluate_ranking_model(y_test, y_pred_test, race_groups_test)
    logging.info("\nテストデータ評価:")
    for metric, value in test_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    # 特徴量重要度
    logging.info("\n特徴量重要度 Top 10:")
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10))

    # モデルとメタデータの保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # モデル保存
    model_file = output_path / f'keibalab_g1_model_{timestamp}.txt'
    model.save_model(str(model_file))
    logging.info(f"\nモデル保存: {model_file}")

    # Pickle形式でも保存（予測時に使用）
    pickle_file = output_path / f'keibalab_g1_model_{timestamp}.pkl'
    joblib.dump(model, pickle_file)
    logging.info(f"モデル保存(pkl): {pickle_file}")

    # 特徴量リスト保存
    features_file = output_path / f'keibalab_g1_features_{timestamp}.txt'
    with open(features_file, 'w') as f:
        for feat in available_features:
            f.write(f"{feat}\n")
    logging.info(f"特徴量リスト保存: {features_file}")

    # 特徴量重要度保存
    importance_file = output_path / f'keibalab_g1_importance_{timestamp}.csv'
    feature_importance.to_csv(importance_file, index=False, encoding='utf-8-sig')
    logging.info(f"特徴量重要度保存: {importance_file}")

    # 評価結果保存
    results_file = output_path / f'keibalab_g1_results_{timestamp}.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("keibalab G1モデル訓練結果\n")
        f.write("="*60 + "\n\n")

        f.write(f"データ期間: {df['year'].min()}-{df['year'].max()}年\n")
        f.write(f"総レース数: {df['race_id'].nunique()}\n")
        f.write(f"総データ数: {len(df)}\n\n")

        f.write("データ分割:\n")
        f.write(f"  訓練 (2000-2022): {len(X_train)} records\n")
        f.write(f"  検証 (2023): {len(X_val)} records\n")
        f.write(f"  テスト (2024): {len(X_test)} records\n\n")

        f.write("訓練データ評価:\n")
        for metric, value in train_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write("\n検証データ評価:\n")
        for metric, value in val_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write("\nテストデータ評価:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write("\n特徴量数: {}\n".format(len(available_features)))
        f.write(f"Best iteration: {model.best_iteration}\n")

    logging.info(f"評価結果保存: {results_file}")

    # サマリー
    logging.info("\n" + "="*60)
    logging.info("訓練完了!")
    logging.info("="*60)
    logging.info(f"\nモデル性能（テストデータ 2024年）:")
    logging.info(f"  1着的中率: {test_metrics['top1_accuracy']:.2%}")
    logging.info(f"  3着内的中率: {test_metrics['top3_hit_rate']:.2%}")
    logging.info(f"  NDCG@3: {test_metrics['ndcg@3']:.4f}")

    return model, feature_importance, test_metrics


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("keibalab G1モデル訓練スクリプト")
    logging.info("="*60)

    # ファイルパス（クリーニング済み）
    input_file = 'data/processed/keibalab_g1_2000_2024_cleaned.csv'
    output_dir = 'models/keibalab_g1'

    # モデル訓練
    model, feature_importance, test_metrics = train_keibalab_g1_model(input_file, output_dir)

    logging.info("\n" + "="*60)
    logging.info("完了!")
    logging.info("="*60)


if __name__ == "__main__":
    main()

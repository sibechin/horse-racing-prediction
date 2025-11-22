"""
G1専用モデル: ハイパーパラメータチューニング込み

歴史的G1データ(2016-2024)でゼロから訓練
Optunaでパラメータ最適化
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class G1SpecializedModel:
    """G1データ専用モデル"""

    def __init__(self, output_dir='models/g1_specialized'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_params = None
        self.models = []

    def objective(self, trial, X, y, groups, n_splits=5):
        """Optuna目的関数"""

        # ハイパーパラメータ探索空間
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            'verbose': -1,
            'seed': 42
        }

        # Group K-Fold CV
        gkf = GroupKFold(n_splits=n_splits)
        ndcg_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]
            groups_val = groups[val_idx]

            # レースごとのグループ
            unique_train_groups = pd.Series(groups_train).unique()
            group_indices = [np.where(groups_train == g)[0] for g in unique_train_groups]

            # Dataset
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                group=[len(g) for g in group_indices],
                free_raw_data=False
            )

            # 訓練
            num_boost_round = trial.suggest_int('num_boost_round', 50, 300)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=None,
                callbacks=[lgb.log_evaluation(period=0)]  # ログ抑制
            )

            # Validation評価
            val_pred = model.predict(X_val)
            unique_val_races = pd.Series(groups_val).unique()

            race_ndcg = []
            for race_id in unique_val_races:
                race_mask = (groups_val == race_id)
                race_y = y_val[race_mask]
                race_pred = val_pred[race_mask]

                if len(race_y) > 0:
                    predicted_order = np.argsort(-race_pred)
                    true_winner_idx = race_y.argmin()
                    ndcg_at_1 = 1.0 if predicted_order[0] == true_winner_idx else 0.0
                    race_ndcg.append(ndcg_at_1)

            ndcg_scores.append(np.mean(race_ndcg) if race_ndcg else 0.0)

        return np.mean(ndcg_scores)

    def tune_hyperparameters(self, data_path: str, n_trials=50, n_splits=5):
        """ハイパーパラメータチューニング"""
        logging.info("="*60)
        logging.info("G1専用モデル: ハイパーパラメータチューニング")
        logging.info("="*60)

        # データ読み込み
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        logging.info(f"\nG1データ: {len(df)} records, {df['race_id'].nunique()} races")

        # finish_positionフィルタリング
        df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
        df = df[df['finish_position'].notna()].copy()
        logging.info(f"有効なデータ: {len(df)} records")

        # 特徴量 (29 features)
        feature_names = [
            'distance', 'venue_code', 'horse_number', 'weight', 'popularity', 'odds',
            'time_seconds', 'age', 'year', 'jockey_win_rate', 'jockey_top3_rate',
            'jockey_track_win_rate', 'horse_win_rate', 'horse_top3_rate',
            'horse_race_count', 'horse_dist_win_rate', 'field_size', 'race_avg_odds',
            'popularity_rank', 'month', 'day_of_week', 'track_type_encoded',
            'track_condition_encoded', 'weather_encoded', 'sex_encoded',
            'distance_category_encoded', 'season_encoded', 'track_combined_encoded',
            'time_weight'
        ]

        X = df[feature_names].copy()
        y = df['finish_position'].values
        groups = df['race_id'].values

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X.loc[:, col] = X[col].fillna(X[col].median())

        # Optuna Study
        logging.info(f"\nOptuna最適化開始")
        logging.info(f"  Trials: {n_trials}")
        logging.info(f"  CV Folds: {n_splits}")

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        study.optimize(
            lambda trial: self.objective(trial, X, y, groups, n_splits),
            n_trials=n_trials,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        logging.info("\n" + "="*60)
        logging.info("最適パラメータ発見!")
        logging.info("="*60)
        logging.info(f"Best NDCG@1 (CV): {study.best_value:.4f}")
        logging.info(f"\nBest Parameters:")
        for key, value in self.best_params.items():
            logging.info(f"  {key}: {value}")

        # 最適パラメータで全データ訓練
        logging.info("\n最適パラメータで最終モデル訓練...")
        self.train_final_model(X, y, groups, n_splits)

        return study

    def train_final_model(self, X, y, groups, n_splits=5):
        """最適パラメータで最終モデル訓練"""

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            **self.best_params,
            'verbose': -1,
            'seed': 42
        }

        num_boost_round = self.best_params.pop('num_boost_round', 150)

        # Group K-Fold
        gkf = GroupKFold(n_splits=n_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            logging.info(f"  Fold {fold_idx}/{n_splits} 訓練中...")

            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            groups_train = groups[train_idx]

            unique_train_groups = pd.Series(groups_train).unique()
            group_indices = [np.where(groups_train == g)[0] for g in unique_train_groups]

            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                group=[len(g) for g in group_indices],
                free_raw_data=False
            )

            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=None,
                callbacks=[lgb.log_evaluation(period=50)]
            )

            self.models.append(model)

            # モデル保存
            model_path = self.output_dir / f'g1_specialized_fold{fold_idx}.txt'
            model.save_model(str(model_path))

        logging.info(f"\n{n_splits}個のモデル訓練完了!")


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("G1専用モデル訓練 (ハイパーパラメータチューニング込み)")
    logging.info("="*60)

    # モデル初期化
    g1_model = G1SpecializedModel(output_dir='models/g1_specialized')

    # データパス
    data_path = "data/processed/g1_races_historical_processed.csv"
    logging.info(f"\n使用データ: {data_path}")

    # ハイパーパラメータチューニング + 訓練
    study = g1_model.tune_hyperparameters(
        data_path=data_path,
        n_trials=50,  # 50回試行
        n_splits=5    # 5-Fold CV
    )

    # 結果保存
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'LightGBM LambdaRank (G1 Specialized)',
        'training_data': 'G1 Races 2016-2024 (~1,900 records, 102 races)',
        'n_trials': 50,
        'best_cv_ndcg@1': study.best_value,
        'best_params': study.best_params,
        'n_models': len(g1_model.models)
    }

    results_file = g1_model.output_dir / 'training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n結果保存: {results_file}")
    logging.info("\n✅ G1専用モデル訓練完了!")
    logging.info("\n次のステップ:")
    logging.info("  python test_g1_specialized_model.py")
    logging.info("  (2025年G1データでテスト)")


if __name__ == "__main__":
    main()

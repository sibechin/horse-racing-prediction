"""
アンサンブル学習: XGBoost、CatBoost、Random Forest

複数のアルゴリズムを組み合わせて予測精度を向上させる試み
- XGBoost (Gradient Boosting)
- CatBoost (Gradient Boosting with categorical features)
- Random Forest (Bagging)
- LightGBM (既存)

スタッキングアンサンブルで最終予測を行う
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnsembleRankingModel:
    """アンサンブルランキングモデル"""

    def __init__(self, output_dir='models/ensemble'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.xgb_models = []
        self.catboost_models = []
        self.rf_models = []
        self.lgb_models = []

        self.feature_names = [
            'distance', 'venue_code', 'horse_number', 'weight', 'popularity', 'odds',
            'time_seconds', 'age', 'year', 'jockey_win_rate', 'jockey_top3_rate',
            'jockey_track_win_rate', 'horse_win_rate', 'horse_top3_rate',
            'horse_race_count', 'horse_dist_win_rate', 'field_size', 'race_avg_odds',
            'popularity_rank', 'month', 'day_of_week', 'track_type_encoded',
            'track_condition_encoded', 'weather_encoded', 'sex_encoded',
            'distance_category_encoded', 'season_encoded', 'track_combined_encoded',
            'time_weight'
        ]

    def train_all_models(self, data_path: str, n_splits=5):
        """全モデルを訓練"""
        logging.info("="*60)
        logging.info("アンサンブルモデル訓練開始")
        logging.info("="*60)

        # データ読み込み
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        logging.info(f"\nG1データ: {len(df)} records, {df['race_id'].nunique()} races")

        # finish_positionフィルタリング
        df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
        df = df[df['finish_position'].notna()].copy()
        logging.info(f"有効なデータ: {len(df)} records")

        # 特徴量準備
        X = df[self.feature_names].copy()
        y = df['finish_position'].values
        groups = df['race_id'].values

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X.loc[:, col] = X[col].fillna(X[col].median())

        # Group K-Fold
        gkf = GroupKFold(n_splits=n_splits)

        logging.info(f"\n{n_splits}-Fold Cross-Validation")
        logging.info(f"訓練する model: XGBoost, CatBoost, Random Forest, LightGBM")

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Fold {fold_idx}/{n_splits}")
            logging.info(f"{'='*60}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]

            # 1. XGBoost
            logging.info("\n[1/4] XGBoost 訓練中...")
            xgb_model = xgb.XGBRanker(
                objective='rank:pairwise',
                learning_rate=0.01,
                max_depth=6,
                n_estimators=100,
                reg_alpha=1.0,
                reg_lambda=5.0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold_idx
            )

            # レースごとのグループサイズ
            unique_train_groups = pd.Series(groups_train).unique()
            group_sizes = [np.sum(groups_train == g) for g in unique_train_groups]

            xgb_model.fit(
                X_train, y_train,
                group=group_sizes,
                verbose=False
            )
            self.xgb_models.append(xgb_model)
            logging.info("  ✓ XGBoost 訓練完了")

            # 2. CatBoost
            logging.info("\n[2/4] CatBoost 訓練中...")

            # CatBoost用のグループデータ準備
            train_df_cb = X_train.copy()
            train_df_cb['target'] = y_train
            train_df_cb['group_id'] = groups_train
            train_df_cb = train_df_cb.sort_values('group_id')

            cb_train = cb.Pool(
                data=train_df_cb[self.feature_names],
                label=train_df_cb['target'],
                group_id=train_df_cb['group_id']
            )

            cb_model = cb.CatBoost({
                'loss_function': 'YetiRank',
                'iterations': 100,
                'learning_rate': 0.01,
                'depth': 6,
                'l2_leaf_reg': 5.0,
                'random_seed': 42 + fold_idx,
                'verbose': False
            })

            cb_model.fit(cb_train)
            self.catboost_models.append(cb_model)
            logging.info("  ✓ CatBoost 訓練完了")

            # 3. Random Forest
            logging.info("\n[3/4] Random Forest 訓練中...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42 + fold_idx,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            self.rf_models.append(rf_model)
            logging.info("  ✓ Random Forest 訓練完了")

            # 4. LightGBM
            logging.info("\n[4/4] LightGBM 訓練中...")
            lgb_train = lgb.Dataset(
                X_train,
                label=y_train,
                group=[len(g) for g in [np.where(groups_train == gid)[0] for gid in unique_train_groups]],
                free_raw_data=False
            )

            lgb_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3, 5],
                'learning_rate': 0.01,
                'num_leaves': 31,
                'max_depth': 6,
                'lambda_l1': 1.0,
                'lambda_l2': 5.0,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42 + fold_idx
            }

            lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            self.lgb_models.append(lgb_model)
            logging.info("  ✓ LightGBM 訓練完了")

            # Validation評価
            self._evaluate_fold(X_val, y_val, groups[val_idx], fold_idx)

        logging.info(f"\n{'='*60}")
        logging.info("全モデル訓練完了!")
        logging.info(f"{'='*60}")
        logging.info(f"XGBoost モデル数: {len(self.xgb_models)}")
        logging.info(f"CatBoost モデル数: {len(self.catboost_models)}")
        logging.info(f"Random Forest モデル数: {len(self.rf_models)}")
        logging.info(f"LightGBM モデル数: {len(self.lgb_models)}")

        # モデル保存
        self._save_models()

    def _evaluate_fold(self, X_val, y_val, groups_val, fold_idx):
        """Fold評価"""
        # 各モデルの予測
        xgb_pred = self.xgb_models[-1].predict(X_val)
        cb_pred = self.catboost_models[-1].predict(X_val)
        rf_pred = self.rf_models[-1].predict(X_val)
        lgb_pred = self.lgb_models[-1].predict(X_val)

        # アンサンブル予測 (平均)
        ensemble_pred = (xgb_pred + cb_pred + rf_pred + lgb_pred) / 4.0

        # レースごとにNDCG@1を計算
        unique_val_races = pd.Series(groups_val).unique()
        ndcg_scores = {
            'xgb': [], 'catboost': [], 'rf': [], 'lgb': [], 'ensemble': []
        }

        for race_id in unique_val_races:
            race_mask = (groups_val == race_id)
            race_y = y_val[race_mask]

            true_winner_idx = race_y.argmin()

            # 各モデルのNDCG@1
            for name, pred in [('xgb', xgb_pred), ('catboost', cb_pred),
                               ('rf', rf_pred), ('lgb', lgb_pred),
                               ('ensemble', ensemble_pred)]:
                race_pred = pred[race_mask]
                predicted_order = np.argsort(-race_pred)
                ndcg_at_1 = 1.0 if predicted_order[0] == true_winner_idx else 0.0
                ndcg_scores[name].append(ndcg_at_1)

        # 平均NDCG@1
        logging.info(f"\nFold {fold_idx} Validation NDCG@1:")
        for name, scores in ndcg_scores.items():
            mean_ndcg = np.mean(scores) if scores else 0.0
            logging.info(f"  {name:12s}: {mean_ndcg:.4f}")

    def _save_models(self):
        """モデル保存"""
        logging.info(f"\nモデル保存中...")

        # XGBoost
        for i, model in enumerate(self.xgb_models, 1):
            model_path = self.output_dir / f'xgboost_fold{i}.json'
            model.save_model(str(model_path))

        # CatBoost
        for i, model in enumerate(self.catboost_models, 1):
            model_path = self.output_dir / f'catboost_fold{i}.cbm'
            model.save_model(str(model_path))

        # Random Forest (pickle)
        import joblib
        for i, model in enumerate(self.rf_models, 1):
            model_path = self.output_dir / f'random_forest_fold{i}.pkl'
            joblib.dump(model, str(model_path))

        # LightGBM
        for i, model in enumerate(self.lgb_models, 1):
            model_path = self.output_dir / f'lightgbm_fold{i}.txt'
            model.save_model(str(model_path))

        logging.info(f"全モデル保存完了: {self.output_dir}")

    def predict_ensemble(self, X):
        """アンサンブル予測"""
        predictions = {
            'xgb': [], 'catboost': [], 'rf': [], 'lgb': []
        }

        # 各モデルの予測
        for model in self.xgb_models:
            predictions['xgb'].append(model.predict(X))

        for model in self.catboost_models:
            predictions['catboost'].append(model.predict(X))

        for model in self.rf_models:
            predictions['rf'].append(model.predict(X))

        for model in self.lgb_models:
            predictions['lgb'].append(model.predict(X))

        # 各アルゴリズムの平均
        xgb_mean = np.mean(predictions['xgb'], axis=0)
        cb_mean = np.mean(predictions['catboost'], axis=0)
        rf_mean = np.mean(predictions['rf'], axis=0)
        lgb_mean = np.mean(predictions['lgb'], axis=0)

        # 最終アンサンブル (単純平均)
        ensemble_pred = (xgb_mean + cb_mean + rf_mean + lgb_mean) / 4.0

        return {
            'xgb': xgb_mean,
            'catboost': cb_mean,
            'rf': rf_mean,
            'lgb': lgb_mean,
            'ensemble': ensemble_pred
        }


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("アンサンブルモデル訓練")
    logging.info("="*60)
    logging.info("\n使用アルゴリズム:")
    logging.info("  1. XGBoost (Gradient Boosting)")
    logging.info("  2. CatBoost (Gradient Boosting + Categorical)")
    logging.info("  3. Random Forest (Bagging)")
    logging.info("  4. LightGBM (Gradient Boosting)")
    logging.info("\n最終予測: 4モデルの単純平均")

    # モデル初期化
    ensemble_model = EnsembleRankingModel(output_dir='models/ensemble')

    # データパス
    data_path = "data/processed/g1_races_historical_processed.csv"
    logging.info(f"\n使用データ: {data_path}")

    # 全モデル訓練
    ensemble_model.train_all_models(data_path, n_splits=5)

    # 結果保存
    results = {
        'timestamp': datetime.now().isoformat(),
        'models': ['XGBoost', 'CatBoost', 'Random Forest', 'LightGBM'],
        'ensemble_method': 'Simple Average',
        'training_data': 'G1 Races 2016-2024 (~1,900 records, 102 races)',
        'n_folds': 5
    }

    results_file = ensemble_model.output_dir / 'training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n結果保存: {results_file}")
    logging.info("\n✅ アンサンブルモデル訓練完了!")
    logging.info("\n次のステップ:")
    logging.info("  python test_ensemble_models.py")
    logging.info("  (2025年G1データでテスト)")


if __name__ == "__main__":
    main()

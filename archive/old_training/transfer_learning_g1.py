"""
転移学習: 既存モデルをG1データでファインチューニング

アプローチ:
1. 既存の7-Foldモデル(一般レース学習済み)をベースとして読み込み
2. 歴史的G1データ(2016-2024)でファインチューニング
3. 低学習率・少ないブースティング回数で過学習を防ぐ
4. 2025年G1データで評価
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging
from sklearn.model_selection import GroupKFold
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TransferLearningG1:
    """
    転移学習モデル: 一般レースで学習したモデルをG1データでファインチューニング
    """

    def __init__(self, base_model_dir='models', output_dir='models/transfer_learning'):
        self.base_model_dir = Path(base_model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_models = []
        self.finetuned_models = []
        self.feature_names = []

    def load_base_models(self):
        """既存の5-Foldモデルを読み込み"""
        logging.info("="*60)
        logging.info("既存モデル読み込み (5-Fold Optimized)")
        logging.info("="*60)

        for i in range(1, 6):
            model_path = self.base_model_dir / f'lightgbm_lambdarank_optimized_fold{i}.txt'
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                self.base_models.append(model)
                logging.info(f"  Fold {i} 読み込み完了")

        if not self.base_models:
            logging.error("❌ モデルが見つかりません")
            return False

        # 特徴量リスト (29 features from optimized model)
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

        logging.info(f"\nモデル数: {len(self.base_models)}")
        logging.info(f"特徴量数: {len(self.feature_names)}")

        return True

    def finetune_on_g1_data(self, g1_data_path: str, n_splits=5, max_rounds=150):
        """
        G1データでファインチューニング

        Args:
            g1_data_path: G1データパス (2016-2024)
            n_splits: K-Fold分割数
            max_rounds: ブースティング回数 (少なめに設定)
        """
        logging.info("\n" + "="*60)
        logging.info("G1データでファインチューニング開始")
        logging.info("="*60)

        # データ読み込み
        df = pd.read_csv(g1_data_path, encoding='utf-8-sig')
        logging.info(f"\nG1データ: {len(df)} records, {df['race_id'].nunique()} races")

        # finish_positionを数値に変換 (失格馬などを除外)
        df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
        df = df[df['finish_position'].notna()].copy()
        logging.info(f"有効なデータ: {len(df)} records (非数値を除外)")

        # 特徴量とターゲット準備
        available_features = [f for f in self.feature_names if f in df.columns]
        if len(available_features) < len(self.feature_names):
            logging.warning(f"⚠️ 不足している特徴量: {set(self.feature_names) - set(available_features)}")

        X = df[available_features].copy()
        y = df['finish_position'].values

        # グループ (race_id)
        groups = df['race_id'].values

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X.loc[:, col] = X[col].fillna(median_val)

        # Group K-Fold (レース単位で分割)
        gkf = GroupKFold(n_splits=n_splits)

        logging.info(f"\n{n_splits}-Fold Cross-Validation (Group K-Fold)")
        logging.info(f"ファインチューニング設定:")
        logging.info(f"  Boosting rounds: {max_rounds}")
        logging.info(f"  Learning rate: 0.01 (低学習率で慎重にファインチューニング)")
        logging.info(f"  Lambda L2: 5.0 (強い正則化)")

        # 各Foldでファインチューニング
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            logging.info(f"\n--- Fold {fold_idx}/{n_splits} ---")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]

            # レースごとのグループ情報
            unique_train_groups = pd.Series(groups_train).unique()
            group_indices = [np.where(groups_train == g)[0] for g in unique_train_groups]

            # LightGBM Dataset
            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                group=[len(g) for g in group_indices],
                free_raw_data=False
            )

            # ファインチューニング用パラメータ (低学習率・強正則化)
            finetune_params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3, 5],
                'learning_rate': 0.01,  # 低学習率
                'num_leaves': 31,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'lambda_l1': 1.0,
                'lambda_l2': 5.0,  # 強い正則化
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42 + fold_idx
            }

            # 既存モデルから継続学習 (init_model)
            # Note: LightGBM の init_model は特徴量が完全一致している必要がある
            # ここでは既存モデルをベースに、G1データで追加学習

            init_model = None
            if fold_idx <= len(self.base_models):
                init_model = self.base_models[fold_idx - 1]

            # ファインチューニング
            logging.info(f"  学習開始 (init_model: {'あり' if init_model else 'なし'})")

            finetuned_model = lgb.train(
                finetune_params,
                train_data,
                num_boost_round=max_rounds,
                init_model=init_model,
                valid_sets=None,  # Validation setは使わない (データが少ないため)
                callbacks=[
                    lgb.log_evaluation(period=50)
                ]
            )

            self.finetuned_models.append(finetuned_model)

            # Validation評価
            val_pred = finetuned_model.predict(X_val)
            val_groups = groups[val_idx]

            # レースごとにNDCG@1を計算
            unique_val_races = pd.Series(val_groups).unique()
            ndcg_scores = []

            for race_id in unique_val_races:
                race_mask = (val_groups == race_id)
                race_y = y_val[race_mask]
                race_pred = val_pred[race_mask]

                # NDCG@1 (勝ち馬を1位に予測できたか)
                if len(race_y) > 0:
                    predicted_order = np.argsort(-race_pred)
                    true_winner_idx = race_y.argmin()
                    ndcg_at_1 = 1.0 if predicted_order[0] == true_winner_idx else 0.0
                    ndcg_scores.append(ndcg_at_1)

            mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
            logging.info(f"  Validation NDCG@1: {mean_ndcg:.4f}")

            fold_results.append({
                'fold': fold_idx,
                'ndcg@1': mean_ndcg,
                'val_races': len(unique_val_races)
            })

            # モデル保存
            model_path = self.output_dir / f'g1_finetuned_fold{fold_idx}.txt'
            finetuned_model.save_model(str(model_path))
            logging.info(f"  モデル保存: {model_path}")

        # 全体の統計
        mean_ndcg_all = np.mean([r['ndcg@1'] for r in fold_results])
        logging.info("\n" + "="*60)
        logging.info("ファインチューニング完了")
        logging.info("="*60)
        logging.info(f"平均 Validation NDCG@1: {mean_ndcg_all:.4f}")

        # 結果保存
        results_file = self.output_dir / 'finetuning_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model': 'LightGBM Transfer Learning (G1 Fine-tuned)',
                'base_model': '5-Fold Optimized (General Races)',
                'finetuning_data': f'{len(df)} records, {df["race_id"].nunique()} races',
                'n_splits': n_splits,
                'max_rounds': max_rounds,
                'mean_val_ndcg@1': mean_ndcg_all,
                'fold_results': fold_results
            }, f, indent=2, ensure_ascii=False)

        logging.info(f"\n結果保存: {results_file}")

        return fold_results


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("転移学習: G1データファインチューニング")
    logging.info("="*60)

    # 転移学習モデル初期化
    transfer_model = TransferLearningG1(
        base_model_dir='models',
        output_dir='models/transfer_learning'
    )

    # 既存モデル読み込み
    if not transfer_model.load_base_models():
        logging.error("❌ 既存モデルの読み込みに失敗しました")
        return

    # G1データでファインチューニング
    g1_data_path = "data/processed/g1_races_historical_processed.csv"
    logging.info(f"\n使用するG1データ: {g1_data_path}")

    fold_results = transfer_model.finetune_on_g1_data(
        g1_data_path=g1_data_path,
        n_splits=5,  # 5-Fold (102レース ÷ 5 ≈ 20レース/Fold)
        max_rounds=150  # 少なめのブースティング回数
    )

    logging.info("\n✅ ファインチューニング完了!")
    logging.info("\n次のステップ:")
    logging.info("  python test_finetuned_model_2025.py")
    logging.info("  (ファインチューニングしたモデルで2025年G1データをテスト)")


if __name__ == "__main__":
    main()

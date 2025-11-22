"""
精度最優先の最適化モデル訓練

過学習防止とデータ量最適化を考慮した高度な訓練パイプライン
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import optuna
import json
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class OptimizedModelTrainer:
    """最適化モデル訓練"""

    def __init__(self, data_path: str, config: dict = None):
        self.data_path = data_path
        self.config = config or self._default_config()
        self.data = None
        self.feature_cols = []
        self.best_params = None
        self.models = []
        self.cv_results = []

    def _default_config(self):
        """デフォルト設定"""
        return {
            'n_folds': 7,  # 過学習対策で増加
            'optuna_trials': 250,  # 精度優先
            'early_stopping_rounds': 100,
            'num_boost_round': 5000,
            'random_state': 42,
            'output_dir': 'models/optimized',
            'baseline_dir': 'models/baseline_backup_v1'
        }

    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logging.info("=" * 70)
        logging.info("データ読み込みと前処理")
        logging.info("=" * 70)

        # データ読み込み
        self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
        logging.info(f"データサイズ: {len(self.data)}レコード")
        logging.info(f"レース数: {self.data['race_id'].nunique()}")

        # 日付処理
        self.data['race_date'] = pd.to_datetime(self.data['race_date'])

        # カテゴリカル変数のエンコーディング (重要な特徴量)
        logging.info("カテゴリカル変数のエンコーディング...")

        # track_type: 芝/ダート
        if 'track_type' in self.data.columns and self.data['track_type'].dtype == 'object':
            track_type_map = {'芝': 0, 'ダート': 1, 'dart': 1, 'turf': 0}
            self.data['track_type_encoded'] = self.data['track_type'].map(
                lambda x: track_type_map.get(x, 0) if pd.notna(x) else 0
            )
            logging.info(f"  track_type → track_type_encoded")

        # weather: 天候
        if 'weather' in self.data.columns and self.data['weather'].dtype == 'object':
            weather_map = {'晴': 0, '曇': 1, '雨': 2, '小雨': 2, '小雪': 3, '雪': 3}
            self.data['weather_encoded'] = self.data['weather'].map(
                lambda x: weather_map.get(x, 0) if pd.notna(x) else 0
            )
            logging.info(f"  weather → weather_encoded")

        # track_condition: 馬場状態
        if 'track_condition' in self.data.columns and self.data['track_condition'].dtype == 'object':
            condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
            self.data['track_condition_encoded'] = self.data['track_condition'].map(
                lambda x: condition_map.get(x, 0) if pd.notna(x) else 0
            )
            logging.info(f"  track_condition → track_condition_encoded")

        # sex: 性別
        if 'sex' in self.data.columns and self.data['sex'].dtype == 'object':
            sex_map = {'牡': 0, '牝': 1, '騸': 2, 'セ': 2}
            self.data['sex_encoded'] = self.data['sex'].map(
                lambda x: sex_map.get(x, 0) if pd.notna(x) else 0
            )
            logging.info(f"  sex → sex_encoded")

        # 特徴量選択
        exclude_cols = [
            'race_id', 'race_date', 'horse_name', 'horse_id', 'jockey_id',
            'finish_position', 'venue_name', 'time', 'margin', 'passing_order',
            'post_time', 'race_name', 'jockey_name', 'bracket_number'
        ]

        # 数値型カラムのみ選択 (エンコード済みカラムを含む)
        self.feature_cols = [col for col in self.data.columns
                             if col not in exclude_cols and
                             self.data[col].dtype in ['int64', 'float64', 'bool']]

        logging.info(f"特徴量数: {len(self.feature_cols)}")

        # 未エンコードのobject型カラムをログに記録
        object_cols = [col for col in self.data.columns
                      if col not in exclude_cols and
                      self.data[col].dtype == 'object' and
                      not col.endswith('_encoded')]
        if object_cols:
            logging.info(f"未使用のobject型カラム: {', '.join(object_cols)}")

        # 欠損値処理
        for col in self.feature_cols:
            if self.data[col].dtype in ['float64', 'int64']:
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)

        # データ検証
        logging.info("\nデータ統計:")
        logging.info(f"  期間: {self.data['race_date'].min()} - {self.data['race_date'].max()}")
        logging.info(f"  平均出走頭数: {self.data.groupby('race_id').size().mean():.1f}")

        return self

    def optimize_hyperparameters(self):
        """ハイパーパラメータ最適化"""
        logging.info("\n" + "=" * 70)
        logging.info("ハイパーパラメータ最適化 (Optuna)")
        logging.info("=" * 70)

        # Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config['random_state'])
        )

        # 最適化実行
        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=self.config['optuna_trials'],
            show_progress_bar=True,
            callbacks=[self._optuna_callback]
        )

        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'verbose': -1,
            'random_state': self.config['random_state'],
            'force_col_wise': True
        })

        logging.info(f"\n最良NDCG@1: {study.best_value:.4f}")
        logging.info("\n最良パラメータ:")
        for key, value in self.best_params.items():
            if key not in ['objective', 'metric', 'ndcg_eval_at', 'verbose', 'random_state', 'force_col_wise']:
                logging.info(f"  {key}: {value}")

        return self

    def _objective(self, trial):
        """Optuna目的関数"""
        # パラメータ提案 (過学習対策強化)
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 50.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1],
            'verbose': -1
        }

        # 3-Fold CV (高速化)
        kf = GroupKFold(n_splits=3)
        groups = self.data['race_id']

        scores = []

        for train_idx, val_idx in kf.split(self.data, groups=groups):
            train_data = self.data.iloc[train_idx]
            val_data = self.data.iloc[val_idx]

            # グループ作成
            train_groups = train_data.groupby('race_id').size().values
            val_groups = val_data.groupby('race_id').size().values

            # Dataset作成
            train_set = lgb.Dataset(
                train_data[self.feature_cols],
                label=train_data['finish_position'],
                group=train_groups
            )

            val_set = lgb.Dataset(
                val_data[self.feature_cols],
                label=val_data['finish_position'],
                group=val_groups,
                reference=train_set
            )

            # 訓練
            model = lgb.train(
                params,
                train_set,
                num_boost_round=500,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )

            # 評価
            val_pred = model.predict(val_data[self.feature_cols])

            # NDCG@1計算
            ndcg_scores = []
            for race_id in val_data['race_id'].unique():
                race_mask = val_data['race_id'] == race_id
                if race_mask.sum() > 1:
                    y_true = val_data[race_mask]['finish_position'].values
                    y_pred = val_pred[race_mask]

                    # 降順ソート
                    sorted_indices = np.argsort(y_pred)[::-1]
                    sorted_true = y_true[sorted_indices]

                    # NDCG@1
                    dcg = (2 ** sorted_true[0] - 1) / np.log2(2)
                    ideal = np.sort(y_true)
                    idcg = (2 ** ideal[0] - 1) / np.log2(2)

                    if idcg > 0:
                        ndcg_scores.append(dcg / idcg)

            scores.append(np.mean(ndcg_scores))

        return np.mean(scores)

    def _optuna_callback(self, study, trial):
        """Optunaコールバック"""
        if trial.number % 10 == 0:
            logging.info(f"Trial {trial.number}: NDCG@1 = {trial.value:.4f}")

    def train_final_model(self):
        """最終モデル訓練 (k-Fold CV)"""
        logging.info("\n" + "=" * 70)
        logging.info(f"{self.config['n_folds']}-Fold CV 最終モデル訓練")
        logging.info("=" * 70)

        kf = GroupKFold(n_splits=self.config['n_folds'])
        groups = self.data['race_id']

        self.models = []
        self.cv_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.data, groups=groups), 1):
            logging.info(f"\n{'='*70}")
            logging.info(f"Fold {fold}/{self.config['n_folds']}")
            logging.info(f"{'='*70}")

            train_data = self.data.iloc[train_idx]
            val_data = self.data.iloc[val_idx]

            logging.info(f"Train: {len(train_data)}レコード, {train_data['race_id'].nunique()}レース")
            logging.info(f"Val: {len(val_data)}レコード, {val_data['race_id'].nunique()}レース")

            # グループ作成
            train_groups = train_data.groupby('race_id').size().values
            val_groups = val_data.groupby('race_id').size().values

            # Dataset作成
            train_set = lgb.Dataset(
                train_data[self.feature_cols],
                label=train_data['finish_position'],
                group=train_groups
            )

            val_set = lgb.Dataset(
                val_data[self.feature_cols],
                label=val_data['finish_position'],
                group=val_groups,
                reference=train_set
            )

            # 訓練
            evals_result = {}
            model = lgb.train(
                self.best_params,
                train_set,
                num_boost_round=self.config['num_boost_round'],
                valid_sets=[train_set, val_set],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(self.config['early_stopping_rounds']),
                    lgb.log_evaluation(100),
                    lgb.record_evaluation(evals_result)
                ]
            )

            self.models.append(model)

            # 評価
            train_pred = model.predict(train_data[self.feature_cols])
            val_pred = model.predict(val_data[self.feature_cols])

            # NDCG計算
            train_ndcg = self._calculate_ndcg(train_data, train_pred)
            val_ndcg = self._calculate_ndcg(val_data, val_pred)

            # Top1/Top3精度計算
            train_top1 = self._calculate_top_accuracy(train_data, train_pred, k=1)
            train_top3 = self._calculate_top_accuracy(train_data, train_pred, k=3)
            val_top1 = self._calculate_top_accuracy(val_data, val_pred, k=1)
            val_top3 = self._calculate_top_accuracy(val_data, val_pred, k=3)

            fold_result = {
                'fold': fold,
                'train_ndcg@1': train_ndcg[0],
                'train_ndcg@3': train_ndcg[1],
                'train_ndcg@5': train_ndcg[2],
                'val_ndcg@1': val_ndcg[0],
                'val_ndcg@3': val_ndcg[1],
                'val_ndcg@5': val_ndcg[2],
                'train_top1': train_top1,
                'train_top3': train_top3,
                'val_top1': val_top1,
                'val_top3': val_top3,
                'best_iteration': model.best_iteration
            }

            self.cv_results.append(fold_result)

            logging.info(f"\nFold {fold} 結果:")
            logging.info(f"  Train NDCG@1: {train_ndcg[0]:.4f}, Top1: {train_top1:.2%}")
            logging.info(f"  Val NDCG@1: {val_ndcg[0]:.4f}, Top1: {val_top1:.2%}")
            logging.info(f"  過学習ギャップ: {abs(train_ndcg[0] - val_ndcg[0]):.4f}")

            # 過学習警告
            if abs(train_ndcg[0] - val_ndcg[0]) > 0.10:
                logging.warning("  ⚠️ 過学習の可能性あり (ギャップ > 0.10)")

        return self

    def _calculate_ndcg(self, data, predictions):
        """NDCG計算"""
        ndcg_1, ndcg_3, ndcg_5 = [], [], []

        for race_id in data['race_id'].unique():
            race_mask = data['race_id'] == race_id
            if race_mask.sum() > 1:
                y_true = data[race_mask]['finish_position'].values
                y_pred = predictions[race_mask]

                for k, ndcg_list in [(1, ndcg_1), (3, ndcg_3), (5, ndcg_5)]:
                    try:
                        score = ndcg_score([y_true], [y_pred], k=k)
                        ndcg_list.append(score)
                    except:
                        pass

        return (np.mean(ndcg_1) if ndcg_1 else 0.0,
                np.mean(ndcg_3) if ndcg_3 else 0.0,
                np.mean(ndcg_5) if ndcg_5 else 0.0)

    def _calculate_top_accuracy(self, data, predictions, k=1):
        """Top-k精度計算"""
        correct = 0
        total = 0

        for race_id in data['race_id'].unique():
            race_mask = data['race_id'] == race_id
            if race_mask.sum() > 1:
                y_true = data[race_mask]['finish_position'].values
                y_pred = predictions[race_mask]

                top_k_pred = np.argsort(y_pred)[-k:]
                actual_winner = np.argmin(y_true)

                if actual_winner in top_k_pred:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def save_models(self):
        """モデル保存"""
        logging.info("\n" + "=" * 70)
        logging.info("モデル保存")
        logging.info("=" * 70)

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 各Foldモデル保存
        for i, model in enumerate(self.models, 1):
            model_path = output_dir / f'optimized_model_fold{i}.txt'
            model.save_model(str(model_path))
            logging.info(f"  Fold {i}: {model_path}")

        # 設定・結果保存
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_size': len(self.data),
            'n_folds': self.config['n_folds'],
            'best_params': self.best_params,
            'cv_results': self.cv_results,
            'cv_summary': {
                'mean_val_ndcg@1': np.mean([r['val_ndcg@1'] for r in self.cv_results]),
                'mean_val_top1': np.mean([r['val_top1'] for r in self.cv_results]),
                'mean_val_top3': np.mean([r['val_top3'] for r in self.cv_results]),
                'std_val_ndcg@1': np.std([r['val_ndcg@1'] for r in self.cv_results])
            }
        }

        results_path = output_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logging.info(f"  結果: {results_path}")

        # サマリー表示
        logging.info("\n" + "=" * 70)
        logging.info("最終結果サマリー")
        logging.info("=" * 70)
        logging.info(f"  平均 Val NDCG@1: {results['cv_summary']['mean_val_ndcg@1']:.4f}")
        logging.info(f"  平均 Val Top1: {results['cv_summary']['mean_val_top1']:.2%}")
        logging.info(f"  平均 Val Top3: {results['cv_summary']['mean_val_top3']:.2%}")
        logging.info(f"  NDCG@1 標準偏差: {results['cv_summary']['std_val_ndcg@1']:.4f}")

        return self

    def compare_with_baseline(self):
        """Baseline比較"""
        baseline_path = Path(self.config['baseline_dir']) / 'refined_optimization_results.json'

        if not baseline_path.exists():
            logging.warning("Baselineデータなし")
            return

        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        logging.info("\n" + "=" * 70)
        logging.info("Baseline比較")
        logging.info("=" * 70)

        baseline_ndcg1 = baseline['cv_results']['val_ndcg@1_mean']
        new_ndcg1 = np.mean([r['val_ndcg@1'] for r in self.cv_results])

        logging.info(f"  Baseline NDCG@1: {baseline_ndcg1:.4f}")
        logging.info(f"  新モデル NDCG@1: {new_ndcg1:.4f}")
        logging.info(f"  改善: {((new_ndcg1 - baseline_ndcg1) / baseline_ndcg1 * 100):+.2f}%")


def main():
    """メイン実行"""
    logging.info("=" * 70)
    logging.info("精度最優先 最適化モデル訓練")
    logging.info("=" * 70)

    # データパス
    data_path = "data/processed/cleaned_data_all.csv"

    if not Path(data_path).exists():
        logging.info(f"{data_path} が存在しません。既存データを使用します。")
        data_path = "data/processed/merged_data_all.csv"

    if not Path(data_path).exists():
        logging.error("データファイルが見つかりません")
        return

    # 訓練設定
    config = {
        'n_folds': 7,  # 過学習対策
        'optuna_trials': 250,  # 精度優先
        'early_stopping_rounds': 100,
        'num_boost_round': 5000,
        'random_state': 42,
        'output_dir': 'models/optimized',
        'baseline_dir': 'models/baseline_backup_v1'
    }

    # 訓練パイプライン
    trainer = OptimizedModelTrainer(data_path, config)

    (trainer
     .load_and_prepare_data()
     .optimize_hyperparameters()
     .train_final_model()
     .save_models()
     .compare_with_baseline())

    logging.info("\n" + "=" * 70)
    logging.info("訓練完了!")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()

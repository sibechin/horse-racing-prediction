"""
メタ学習パイプライン

アプローチ2: モデル選択器 - どのモデルを使うべきか学習
アプローチ3: スタッキング - ベースモデルの予測からメタモデルが最終予測
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseModelWrapper:
    """ベースモデルのラッパー"""

    def __init__(self, model_path, model_type):
        """
        Args:
            model_path: モデルファイルパス
            model_type: 'lightgbm' or 'conditional_logit' or 'top3'
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None

    def load(self):
        """モデル読み込み"""
        if not self.model_path.exists():
            logging.warning(f"モデルが見つかりません: {self.model_path}")
            return False

        try:
            if self.model_type == 'lightgbm' or self.model_type == 'top3':
                self.model = lgb.Booster(model_file=str(self.model_path))
            elif self.model_type == 'conditional_logit':
                self.model = joblib.load(self.model_path)

            logging.info(f"✓ {self.model_type} モデル読み込み成功")
            return True
        except Exception as e:
            logging.error(f"モデル読み込みエラー ({self.model_type}): {e}")
            return False

    def predict(self, X, race_groups=None):
        """
        予測実行

        Returns:
            予測スコア（レースごと）
        """
        if self.model is None:
            return None

        try:
            if self.model_type == 'lightgbm' or self.model_type == 'top3':
                # LightGBM形式
                scores = self.model.predict(X)
            elif self.model_type == 'conditional_logit':
                # Conditional Logit形式
                if hasattr(self.model, 'predict_proba'):
                    scores = self.model.predict_proba(X)[:, 1]
                else:
                    scores = self.model.predict(X)

            return scores
        except Exception as e:
            logging.warning(f"予測エラー ({self.model_type}): {e}")
            return None


class ModelSelector:
    """
    アプローチ2: モデル選択器

    レース特徴量から最適なモデルを選択
    """

    def __init__(self):
        self.selector = None
        self.race_feature_cols = [
            'distance', 'field_size', 'track_condition_encoded',
            'surface_encoded', 'race_class_encoded', 'venue_code',
            'month', 'day_of_week'
        ]

    def train(self, df_train, base_models):
        """
        訓練

        Args:
            df_train: 訓練データ
            base_models: {'model_name': BaseModelWrapper}
        """
        logging.info("\n" + "="*60)
        logging.info("アプローチ2: モデル選択器 訓練")
        logging.info("="*60)

        # 各レースで最も良いモデルを特定
        race_features = []
        best_model_labels = []

        for race_id in df_train['race_id'].unique():
            race_data = df_train[df_train['race_id'] == race_id]

            # レース特徴量
            race_feature = self._extract_race_features(race_data)

            # 各モデルの性能評価
            model_scores = {}
            for model_name, model in base_models.items():
                score = self._evaluate_model_on_race(model, race_data)
                if score is not None:
                    model_scores[model_name] = score

            if not model_scores:
                continue

            # 最良モデル
            best_model = max(model_scores, key=model_scores.get)

            race_features.append(race_feature)
            best_model_labels.append(best_model)

        # データフレーム化
        X_train = pd.DataFrame(race_features)
        y_train = pd.Series(best_model_labels)

        # ラベルエンコーディング
        self.label_map = {name: idx for idx, name in enumerate(y_train.unique())}
        self.inv_label_map = {idx: name for name, idx in self.label_map.items()}
        y_train_encoded = y_train.map(self.label_map)

        logging.info(f"\n訓練レース数: {len(X_train)}")
        logging.info(f"モデル分布:")
        for model_name, count in y_train.value_counts().items():
            logging.info(f"  {model_name}: {count}レース ({count/len(y_train)*100:.1f}%)")

        # LightGBM分類器で訓練
        self.selector = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(self.label_map),
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            verbose=-1
        )

        self.selector.fit(X_train, y_train_encoded)

        logging.info("✓ モデル選択器 訓練完了")

        return self

    def _extract_race_features(self, race_data):
        """レース特徴量の抽出"""
        features = {}
        for col in self.race_feature_cols:
            if col in race_data.columns:
                features[col] = race_data[col].iloc[0]
            else:
                features[col] = 0
        return features

    def _evaluate_model_on_race(self, model, race_data):
        """レース単位でのモデル評価"""
        if model.model is None:
            return None

        try:
            # 特徴量準備
            feature_cols = [col for col in race_data.columns
                          if col not in ['race_id', 'horse_id', 'horse_name',
                                       'finish_position', 'finish_position_numeric']]
            X = race_data[feature_cols]

            # 予測
            scores = model.predict(X)
            if scores is None:
                return None

            # 実際の順位
            y_true = race_data['finish_position_numeric'].values

            # Top1精度で評価
            pred_winner_idx = np.argmax(scores)
            actual_winner_idx = np.argmin(y_true)

            return 1.0 if pred_winner_idx == actual_winner_idx else 0.0
        except Exception as e:
            return None

    def select_model(self, race_features):
        """最適モデルを選択"""
        X = pd.DataFrame([race_features])
        pred_encoded = self.selector.predict(X)[0]
        return self.inv_label_map[pred_encoded]

    def save(self, output_path):
        """保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'selector': self.selector,
            'label_map': self.label_map,
            'inv_label_map': self.inv_label_map,
            'race_feature_cols': self.race_feature_cols
        }, output_path)

        logging.info(f"✓ モデル選択器保存: {output_path}")


class StackingEnsemble:
    """
    アプローチ3: スタッキングアンサンブル

    Level 1: ベースモデルの予測
    Level 2: メタモデルが最終予測
    """

    def __init__(self):
        self.meta_model = None
        self.base_feature_names = None

    def train(self, df_train, base_models):
        """
        訓練

        Args:
            df_train: 訓練データ
            base_models: {'model_name': BaseModelWrapper}
        """
        logging.info("\n" + "="*60)
        logging.info("アプローチ3: スタッキングアンサンブル 訓練")
        logging.info("="*60)

        # Cross-validationでベースモデルの予測を生成
        gkf = GroupKFold(n_splits=5)

        # メタ特徴量の初期化
        meta_features = []
        meta_labels = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train['race_id']), 1):
            logging.info(f"\nFold {fold}/5")

            val_data = df_train.iloc[val_idx]

            # 各馬のメタ特徴量を生成
            for race_id in val_data['race_id'].unique():
                race_data = val_data[val_data['race_id'] == race_id]

                # ベースモデルの予測
                base_predictions = {}
                for model_name, model in base_models.items():
                    try:
                        feature_cols = [col for col in race_data.columns
                                      if col not in ['race_id', 'horse_id', 'horse_name',
                                                   'finish_position', 'finish_position_numeric']]
                        X = race_data[feature_cols]
                        scores = model.predict(X)

                        if scores is not None:
                            base_predictions[model_name] = scores
                    except Exception as e:
                        logging.warning(f"予測エラー ({model_name}): {e}")

                if not base_predictions:
                    continue

                # 各馬のメタ特徴量
                for idx, (_, row) in enumerate(race_data.iterrows()):
                    meta_feature = {}

                    # ベースモデルの予測スコア
                    for model_name, scores in base_predictions.items():
                        meta_feature[f'{model_name}_score'] = scores[idx]

                    # レース特徴量
                    for col in ['distance', 'field_size', 'track_condition_encoded',
                               'surface_encoded', 'race_class_encoded']:
                        if col in race_data.columns:
                            meta_feature[col] = row[col]

                    # 馬特徴量
                    for col in ['age', 'weight', 'horse_weight_kg', 'last_3f']:
                        if col in race_data.columns:
                            meta_feature[col] = row[col]

                    meta_features.append(meta_feature)

                    # ラベル（順位）
                    meta_labels.append(row['finish_position_numeric'])

        # DataFrame化
        X_meta = pd.DataFrame(meta_features)
        y_meta = np.array(meta_labels)

        logging.info(f"\nメタ訓練データ:")
        logging.info(f"  サンプル数: {len(X_meta)}")
        logging.info(f"  特徴量数: {len(X_meta.columns)}")

        self.base_feature_names = X_meta.columns.tolist()

        # メタモデル訓練（ランキング）
        self.meta_model = lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            verbose=-1
        )

        # グループ情報（レースごと）
        # 簡易的に、サンプル数 / 平均出走頭数で推定
        avg_horses_per_race = 15
        n_races = len(X_meta) // avg_horses_per_race
        group_sizes = [avg_horses_per_race] * n_races

        self.meta_model.fit(X_meta, y_meta, group=group_sizes)

        logging.info("✓ スタッキングアンサンブル 訓練完了")

        return self

    def predict(self, race_data, base_models):
        """予測"""
        # ベースモデルの予測
        base_predictions = {}
        for model_name, model in base_models.items():
            try:
                feature_cols = [col for col in race_data.columns
                              if col not in ['race_id', 'horse_id', 'horse_name',
                                           'finish_position', 'finish_position_numeric']]
                X = race_data[feature_cols]
                scores = model.predict(X)

                if scores is not None:
                    base_predictions[model_name] = scores
            except:
                pass

        if not base_predictions:
            return None

        # メタ特徴量生成
        meta_features = []
        for idx in range(len(race_data)):
            meta_feature = {}

            # ベースモデルの予測スコア
            for model_name, scores in base_predictions.items():
                meta_feature[f'{model_name}_score'] = scores[idx]

            # レース特徴量
            row = race_data.iloc[idx]
            for col in ['distance', 'field_size', 'track_condition_encoded',
                       'surface_encoded', 'race_class_encoded']:
                if col in race_data.columns:
                    meta_feature[col] = row[col]

            # 馬特徴量
            for col in ['age', 'weight', 'horse_weight_kg', 'last_3f']:
                if col in race_data.columns:
                    meta_feature[col] = row[col]

            meta_features.append(meta_feature)

        X_meta = pd.DataFrame(meta_features)

        # 不足している特徴量を0で補完
        for col in self.base_feature_names:
            if col not in X_meta.columns:
                X_meta[col] = 0

        # 列順序を合わせる
        X_meta = X_meta[self.base_feature_names]

        # メタモデルで予測
        final_scores = self.meta_model.predict(X_meta)

        return final_scores

    def save(self, output_path):
        """保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'meta_model': self.meta_model,
            'base_feature_names': self.base_feature_names
        }, output_path)

        logging.info(f"✓ スタッキングアンサンブル保存: {output_path}")


def evaluate_meta_approach(approach, df_test, base_models, approach_name):
    """メタ学習アプローチの評価"""
    logging.info(f"\n{'='*60}")
    logging.info(f"評価: {approach_name}")
    logging.info(f"{'='*60}")

    metrics = {
        'top1_accuracy': 0,
        'top3_hit_rate': 0,
        'total_races': 0
    }

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]

        try:
            if approach_name == "Model Selector":
                # モデル選択器
                race_features = approach._extract_race_features(race_data)
                selected_model_name = approach.select_model(race_features)
                selected_model = base_models[selected_model_name]

                feature_cols = [col for col in race_data.columns
                              if col not in ['race_id', 'horse_id', 'horse_name',
                                           'finish_position', 'finish_position_numeric']]
                X = race_data[feature_cols]
                scores = selected_model.predict(X)
            else:
                # スタッキング
                scores = approach.predict(race_data, base_models)

            if scores is None:
                continue

            # 実際の順位
            y_true = race_data['finish_position_numeric'].values

            # Top1
            pred_winner_idx = np.argmax(scores)
            actual_winner_idx = np.argmin(y_true)
            if pred_winner_idx == actual_winner_idx:
                metrics['top1_accuracy'] += 1

            # Top3
            if len(scores) >= 3:
                pred_top3 = set(np.argsort(scores)[-3:])
                actual_top3 = set(np.argsort(y_true)[:3])
                if len(pred_top3 & actual_top3) > 0:
                    metrics['top3_hit_rate'] += 1

            metrics['total_races'] += 1
        except Exception as e:
            logging.warning(f"評価エラー (race {race_id}): {e}")

    # パーセンテージ
    if metrics['total_races'] > 0:
        metrics['top1_accuracy'] = metrics['top1_accuracy'] / metrics['total_races']
        metrics['top3_hit_rate'] = metrics['top3_hit_rate'] / metrics['total_races']

    logging.info(f"\n結果:")
    logging.info(f"  Top1精度: {metrics['top1_accuracy']:.2%}")
    logging.info(f"  Top3適中率: {metrics['top3_hit_rate']:.2%}")
    logging.info(f"  評価レース数: {metrics['total_races']}")

    return metrics


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("メタ学習パイプライン")
    logging.info("="*60)

    # データ読み込み
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")

    logging.info(f"\n訓練データ: {train_file}")
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    logging.info(f"  {len(df_train):,}レコード, {df_train['race_id'].nunique()}レース")

    logging.info(f"\nテストデータ: {test_file}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    logging.info(f"  {len(df_test):,}レコード, {df_test['race_id'].nunique()}レース")

    # finish_position_numericの確認・作成
    for df in [df_train, df_test]:
        if 'finish_position_numeric' not in df.columns:
            if 'finish_position' in df.columns:
                df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

    # ベースモデル読み込み
    base_models = {
        'lightgbm': BaseModelWrapper('models/keibalab_g1/keibalab_g1_model_20251121_054321.txt', 'lightgbm'),
        'conditional_logit': BaseModelWrapper('models/statistical/conditional_logit_20251121_054350.pkl', 'conditional_logit'),
        'top3': BaseModelWrapper('models/top3_no_odds/top3_no_odds_model_20251121_053558.txt', 'top3')
    }

    # モデル読み込み
    for name, model in base_models.items():
        model.load()

    # アプローチ2: モデル選択器
    selector = ModelSelector()
    selector.train(df_train, base_models)
    selector.save('models/meta_learning/model_selector.pkl')

    results_selector = evaluate_meta_approach(
        selector, df_test, base_models, "Model Selector"
    )

    # アプローチ3: スタッキング
    stacking = StackingEnsemble()
    stacking.train(df_train, base_models)
    stacking.save('models/meta_learning/stacking_ensemble.pkl')

    results_stacking = evaluate_meta_approach(
        stacking, df_test, base_models, "Stacking Ensemble"
    )

    # 結果比較
    logging.info("\n" + "="*60)
    logging.info("結果比較")
    logging.info("="*60)

    comparison = {
        'Model Selector': results_selector,
        'Stacking Ensemble': results_stacking
    }

    for approach_name, metrics in comparison.items():
        logging.info(f"\n{approach_name}:")
        logging.info(f"  Top1精度: {metrics['top1_accuracy']:.2%}")
        logging.info(f"  Top3適中率: {metrics['top3_hit_rate']:.2%}")

    # 結果保存
    output_dir = Path("models/meta_learning")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"meta_learning_results_{timestamp}.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'model_selector': results_selector,
            'stacking_ensemble': results_stacking
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"\n✓ 結果保存: {results_path}")
    logging.info("\n完了!")


if __name__ == "__main__":
    main()

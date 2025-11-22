"""
動的ハイブリッド融合システム

異なるモデルの強み（当日情報 vs. 事前情報）をコンテキストに応じて
インテリジェントに統合し、安定した高性能な最終ランキングを生成

Components:
- Top3予測モデル (オッズ除外): 事前情報ベース
- Conditional Logit (オッズ使用): 当日情報ベース
- Dynamic Alpha Tuning (DAT): レースごとに最適重みを動的決定
- CombMNZ: 複数モデルの合意度を考慮した融合
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DynamicAlphaTuner:
    """
    Dynamic Alpha Tuning (DAT)

    レース特徴量から最適な重み（alpha）を動的に決定
    """

    def __init__(self):
        self.alpha_model = None
        self.race_feature_cols = [
            'distance', 'field_size', 'track_condition_encoded',
            'surface_encoded', 'race_class_encoded',
            'month', 'day_of_week', 'venue_code'
        ]

    def train(self, df_train, model_no_odds, model_with_odds):
        """
        訓練: 各レースで最適なalphaを学習

        Args:
            df_train: 訓練データ
            model_no_odds: オッズ除外モデル
            model_with_odds: オッズ使用モデル
        """
        logging.info("\n" + "="*60)
        logging.info("Dynamic Alpha Tuning (DAT) 訓練")
        logging.info("="*60)

        race_features = []
        optimal_alphas = []

        # 各レースで最適alphaを探索
        for race_id in df_train['race_id'].unique()[:100]:  # サンプル100レース
            race_data = df_train[df_train['race_id'] == race_id]

            # レース特徴量
            race_feature = self._extract_race_features(race_data)

            # 最適alphaを探索（grid search）
            best_alpha, best_score = self._find_optimal_alpha(
                race_data, model_no_odds, model_with_odds
            )

            if best_alpha is not None:
                race_features.append(race_feature)
                optimal_alphas.append(best_alpha)

        if not race_features:
            logging.warning("訓練データが不足")
            return self

        # DataFrame化
        X = pd.DataFrame(race_features)
        y = np.array(optimal_alphas)

        logging.info(f"\n訓練サンプル数: {len(X)}")
        logging.info(f"Alpha範囲: [{y.min():.2f}, {y.max():.2f}]")
        logging.info(f"Alpha平均: {y.mean():.2f}")

        # LightGBM回帰モデルで学習
        self.alpha_model = lgb.LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            verbose=-1
        )

        self.alpha_model.fit(X, y)

        logging.info("✓ DAT訓練完了")

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

    def _find_optimal_alpha(self, race_data, model_no_odds, model_with_odds):
        """最適alphaの探索"""
        try:
            # 両モデルの予測
            scores_no_odds = self._predict_model(race_data, model_no_odds)
            scores_with_odds = self._predict_model(race_data, model_with_odds)

            if scores_no_odds is None or scores_with_odds is None:
                return None, None

            # 実際の順位
            y_true = race_data['finish_position_numeric'].values

            # Alpha探索
            best_alpha = 0.5
            best_score = 0

            for alpha in np.arange(0, 1.1, 0.1):
                # 融合スコア
                fused_scores = alpha * scores_with_odds + (1 - alpha) * scores_no_odds

                # Top1精度で評価
                pred_winner = np.argmax(fused_scores)
                actual_winner = np.argmin(y_true)

                score = 1.0 if pred_winner == actual_winner else 0.0

                if score > best_score:
                    best_score = score
                    best_alpha = alpha

            return best_alpha, best_score
        except:
            return None, None

    def _predict_model(self, race_data, model):
        """モデルの予測"""
        try:
            if isinstance(model, lgb.Booster):
                # LightGBMモデル（Top3 no-odds）: 19特徴量
                top3_features = [
                    'year', 'month', 'day_of_week', 'venue_code', 'race_number',
                    'sex_encoded', 'age', 'weight', 'last_3f', 'horse_weight_kg',
                    'horse_weight_change', 'field_size', 'race_avg_odds', 'race_avg_weight',
                    'trainer_encoded', 'trainer_frequency', 'jockey_encoded', 'jockey_frequency',
                    'race_type_encoded'
                ]
                available_features = [col for col in top3_features if col in race_data.columns]
                X = race_data[available_features]
                return model.predict(X)

            elif isinstance(model, dict):
                # Conditional Logit形式
                actual_model = model.get('model')
                feature_cols = model.get('feature_cols', [])
                available_cols = [col for col in feature_cols if col in race_data.columns]
                X = race_data[available_cols]

                if actual_model and hasattr(actual_model, 'predict_proba'):
                    return actual_model.predict_proba(X)[:, 1]

            return None
        except:
            return None

    def predict_alpha(self, race_features):
        """レース特徴量からalphaを予測"""
        if self.alpha_model is None:
            return 0.5  # デフォルト

        X = pd.DataFrame([race_features])
        alpha = self.alpha_model.predict(X)[0]

        # 0-1の範囲に制限
        return np.clip(alpha, 0, 1)


class CombMNZFusion:
    """
    CombMNZ正規化による融合

    情報検索分野で実績のある手法
    複数モデルの「合意度」を重視
    """

    @staticmethod
    def fuse(scores_list, weights=None):
        """
        CombMNZ融合

        Args:
            scores_list: [scores1, scores2, ...] (各scoresは正規化済み)
            weights: 各モデルの重み (Noneなら均等)

        Returns:
            融合スコア
        """
        if weights is None:
            weights = np.ones(len(scores_list)) / len(scores_list)

        # 正規化
        normalized_scores = []
        for scores in scores_list:
            scaler = MinMaxScaler()
            norm_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
            normalized_scores.append(norm_scores)

        # CombSUM: 重み付き合計
        combsum = np.zeros_like(normalized_scores[0])
        for scores, weight in zip(normalized_scores, weights):
            combsum += weight * scores

        # 非ゼロスコア数（合意度）
        score_matrix = np.array(normalized_scores).T  # (n_horses, n_models)
        non_zero_count = (score_matrix > 0).sum(axis=1)

        # CombMNZ = CombSUM × 合意度
        combmnz = combsum * non_zero_count

        return combmnz


class DynamicHybridFusion:
    """動的ハイブリッド融合システム"""

    def __init__(self):
        self.model_no_odds = None  # Top3予測（オッズ除外）
        self.model_with_odds = None  # Conditional Logit（オッズ使用）
        self.dat = DynamicAlphaTuner()
        self.combmnz = CombMNZFusion()

        # Top3モデルの訓練時の特徴量（19個）
        self.top3_features = [
            'year', 'month', 'day_of_week', 'venue_code', 'race_number',
            'sex_encoded', 'age', 'weight', 'last_3f', 'horse_weight_kg',
            'horse_weight_change', 'field_size', 'race_avg_odds', 'race_avg_weight',
            'trainer_encoded', 'trainer_frequency', 'jockey_encoded', 'jockey_frequency',
            'race_type_encoded'
        ]

    def load_models(self, path_no_odds, path_with_odds):
        """モデル読み込み"""
        logging.info("モデル読み込み...")

        # Top3予測モデル（オッズ除外）
        self.model_no_odds = lgb.Booster(model_file=str(path_no_odds))
        logging.info(f"✓ Top3予測モデル: {path_no_odds}")

        # Conditional Logit（オッズ使用）
        self.model_with_odds = joblib.load(path_with_odds)
        logging.info(f"✓ Conditional Logit: {path_with_odds}")

    def train_dat(self, df_train):
        """DATを訓練"""
        self.dat.train(df_train, self.model_no_odds, self.model_with_odds)

    def predict(self, race_data, fusion_method='dat'):
        """
        予測

        Args:
            race_data: レースデータ（1レース分）
            fusion_method: 'dat', 'combmnz', 'equal'

        Returns:
            融合された予測スコア
        """
        # 両モデルの予測
        scores_no_odds = self._predict_no_odds(race_data)
        scores_with_odds = self._predict_with_odds(race_data)

        if scores_no_odds is None or scores_with_odds is None:
            # どちらか失敗したら利用可能な方を返す
            return scores_with_odds if scores_no_odds is None else scores_no_odds

        # 融合方法に応じて処理
        if fusion_method == 'dat':
            # Dynamic Alpha Tuning
            race_features = self.dat._extract_race_features(race_data)
            alpha = self.dat.predict_alpha(race_features)

            fused_scores = alpha * scores_with_odds + (1 - alpha) * scores_no_odds

            logging.info(f"  DAT Alpha: {alpha:.2f} (Odds重視度)")

        elif fusion_method == 'combmnz':
            # CombMNZ融合
            fused_scores = self.combmnz.fuse([scores_no_odds, scores_with_odds])

        elif fusion_method == 'equal':
            # 均等重み
            fused_scores = 0.5 * scores_with_odds + 0.5 * scores_no_odds

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        return fused_scores

    def _predict_no_odds(self, race_data):
        """オッズ除外モデルの予測"""
        try:
            # Top3モデルの訓練時の特徴量のみ使用
            available_features = [col for col in self.top3_features if col in race_data.columns]

            if len(available_features) != len(self.top3_features):
                missing = set(self.top3_features) - set(available_features)
                logging.warning(f"オッズ除外モデル: 欠損特徴量 {missing}")

            X = race_data[available_features]
            return self.model_no_odds.predict(X)
        except Exception as e:
            logging.warning(f"オッズ除外モデル予測エラー: {e}")
            return None

    def _predict_with_odds(self, race_data):
        """オッズ使用モデルの予測"""
        try:
            actual_model = self.model_with_odds.get('model')
            feature_cols = self.model_with_odds.get('feature_cols', [])

            # 利用可能な特徴量のみ使用
            available_cols = [col for col in feature_cols if col in race_data.columns]
            X = race_data[available_cols]

            if hasattr(actual_model, 'predict_proba'):
                return actual_model.predict_proba(X)[:, 1]
            elif hasattr(actual_model, 'predict'):
                return actual_model.predict(X)

            return None
        except Exception as e:
            logging.warning(f"オッズ使用モデル予測エラー: {e}")
            return None

    def evaluate(self, df_test, fusion_methods=['dat', 'combmnz', 'equal']):
        """評価"""
        logging.info("\n" + "="*60)
        logging.info("動的ハイブリッド融合 評価")
        logging.info("="*60)

        results = {}

        for method in fusion_methods:
            logging.info(f"\n融合方法: {method.upper()}")

            metrics = {
                'top1_accuracy': 0,
                'top3_hit_rate': 0,
                'total_races': 0
            }

            for race_id in df_test['race_id'].unique():
                race_data = df_test[df_test['race_id'] == race_id]

                try:
                    # 予測
                    scores = self.predict(race_data, fusion_method=method)

                    if scores is None:
                        continue

                    # 実際の順位
                    y_true = race_data['finish_position_numeric'].values

                    # Top1
                    pred_winner = np.argmax(scores)
                    actual_winner = np.argmin(y_true)
                    if pred_winner == actual_winner:
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

            results[method] = metrics

            logging.info(f"  Top1精度: {metrics['top1_accuracy']:.2%}")
            logging.info(f"  Top3適中率: {metrics['top3_hit_rate']:.2%}")
            logging.info(f"  評価レース数: {metrics['total_races']}")

        return results


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("動的ハイブリッド融合システム")
    logging.info("="*60)

    # データ読み込み
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")

    logging.info(f"\n訓練データ: {train_file}")
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')

    logging.info(f"テストデータ: {test_file}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')

    # finish_position_numericの確認
    for df in [df_train, df_test]:
        if 'finish_position_numeric' not in df.columns:
            df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

    # システム初期化
    system = DynamicHybridFusion()

    # モデル読み込み
    system.load_models(
        path_no_odds='models/top3_no_odds/top3_no_odds_model_20251121_053558.txt',
        path_with_odds='models/statistical/conditional_logit_20251121_054350.pkl'
    )

    # DAT訓練
    system.train_dat(df_train)

    # 評価
    results = system.evaluate(df_test, fusion_methods=['dat', 'combmnz', 'equal'])

    # 結果比較
    logging.info("\n" + "="*60)
    logging.info("結果比較")
    logging.info("="*60)

    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = '融合方法'

    print("\n", comparison_df.to_string())

    # 結果保存
    output_dir = Path("models/dynamic_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f"dynamic_fusion_results_{timestamp}.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n✓ 結果保存: {results_path}")
    logging.info("\n完了!")


if __name__ == "__main__":
    main()

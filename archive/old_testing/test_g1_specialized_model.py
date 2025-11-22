"""
G1専用モデルのテスト: 2025年G1データで評価

ベースラインモデルとの比較:
- Original 5-Fold Model: 0% Top1, 0% Top3
- Optimized 7-Fold Model: 0% Top1, 0% Top3
- G1 Specialized Model: [To be determined]
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class G1SpecializedModelTester:
    """G1専用モデルのテスター"""

    def __init__(self, model_dir='models/g1_specialized'):
        self.model_dir = Path(model_dir)
        self.models = []
        self.feature_names = []

    def load_models(self):
        """訓練済みG1専用モデル(5-Fold)を読み込み"""
        logging.info("="*60)
        logging.info("G1専用モデル読み込み")
        logging.info("="*60)

        for i in range(1, 6):
            model_path = self.model_dir / f'g1_specialized_fold{i}.txt'
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                self.models.append(model)
                logging.info(f"  Fold {i} 読み込み完了")
            else:
                logging.warning(f"  ⚠️ {model_path} が見つかりません")

        if not self.models:
            logging.error("❌ モデルが見つかりません")
            return False

        # 特徴量リスト (29 features)
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

        logging.info(f"\nモデル数: {len(self.models)}")
        logging.info(f"特徴量数: {len(self.feature_names)}")

        return True

    def predict(self, X):
        """アンサンブル予測 (5モデルの平均)"""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # 平均予測
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def test_on_2025_data(self, test_data_path: str):
        """2025年G1データでテスト"""
        logging.info("\n" + "="*60)
        logging.info("2025年G1データでテスト開始")
        logging.info("="*60)

        # データ読み込み
        df = pd.read_csv(test_data_path, encoding='utf-8-sig')
        logging.info(f"\nテストデータ: {len(df)} records, {df['race_id'].nunique()} races")

        # finish_positionを数値に変換
        df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
        df = df[df['finish_position'].notna()].copy()
        logging.info(f"有効なデータ: {len(df)} records")

        # 特徴量準備
        available_features = [f for f in self.feature_names if f in df.columns]
        if len(available_features) < len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            logging.warning(f"⚠️ 不足している特徴量: {missing}")

        X = df[available_features].copy()

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X.loc[:, col] = X[col].fillna(median_val)

        # 予測
        logging.info("\n予測実行中...")
        predictions = self.predict(X)

        # レース単位で評価
        race_ids = df['race_id'].unique()
        logging.info(f"\nレース数: {len(race_ids)}")

        results = []
        top1_correct = 0
        top3_correct = 0

        for race_id in race_ids:
            race_mask = (df['race_id'] == race_id)
            race_df = df[race_mask].copy()
            race_pred = predictions[race_mask]

            # 予測順位 (予測値の降順)
            predicted_order = np.argsort(-race_pred)

            # 実際の順位
            actual_positions = race_df['finish_position'].values
            true_winner_idx = actual_positions.argmin()

            # 予測Top3の馬名
            top3_indices = predicted_order[:3]
            predicted_top3 = race_df.iloc[top3_indices]['horse_name'].tolist()

            # 実際のTop3の馬名
            actual_top3_indices = np.argsort(actual_positions)[:3]
            actual_top3 = race_df.iloc[actual_top3_indices]['horse_name'].tolist()

            # Top1評価
            top1_match = (predicted_order[0] == true_winner_idx)
            if top1_match:
                top1_correct += 1

            # Top3評価
            top3_match = true_winner_idx in predicted_order[:3]
            if top3_match:
                top3_correct += 1

            # NDCG@1
            ndcg_at_1 = 1.0 if top1_match else 0.0

            # レース詳細 (JSON serializable types)
            race_info = {
                'race_id': str(race_id),
                'race_name': str(race_df['race_name'].iloc[0]) if 'race_name' in race_df.columns else str(race_id),
                'horses': int(len(race_df)),
                'predicted_winner': str(race_df.iloc[predicted_order[0]]['horse_name']),
                'actual_winner': str(race_df.iloc[true_winner_idx]['horse_name']),
                'top1_correct': bool(top1_match),
                'top3_correct': bool(top3_match),
                'ndcg@1': float(ndcg_at_1),
                'predicted_top3': [str(h) for h in predicted_top3],
                'actual_top3': [str(h) for h in actual_top3]
            }

            results.append(race_info)

            logging.info(f"\n{race_info['race_name']}")
            logging.info(f"  予測1着: {race_info['predicted_winner']}")
            logging.info(f"  実際1着: {race_info['actual_winner']}")
            logging.info(f"  Top1的中: {'✅' if top1_match else '❌'}")
            logging.info(f"  Top3的中: {'✅' if top3_match else '❌'}")

        # 全体の統計
        total_races = len(race_ids)
        top1_accuracy = top1_correct / total_races if total_races > 0 else 0
        top3_accuracy = top3_correct / total_races if total_races > 0 else 0
        mean_ndcg = np.mean([r['ndcg@1'] for r in results])

        logging.info("\n" + "="*60)
        logging.info("テスト結果")
        logging.info("="*60)
        logging.info(f"総レース数: {total_races}")
        logging.info(f"Top1的中数: {top1_correct} / {total_races}")
        logging.info(f"Top3的中数: {top3_correct} / {total_races}")
        logging.info(f"Top1的中率: {top1_accuracy:.1%}")
        logging.info(f"Top3的中率: {top3_accuracy:.1%}")
        logging.info(f"平均NDCG@1: {mean_ndcg:.4f}")

        # 結果保存 (JSON serializable)
        output_results = {
            'timestamp': datetime.now().isoformat(),
            'model': 'LightGBM G1 Specialized (5-Fold Ensemble)',
            'test_period': f'2025 G1 Races ({total_races} races)',
            'summary': {
                'total_races': int(total_races),
                'top1_accuracy': float(top1_accuracy),
                'top3_accuracy': float(top3_accuracy),
                'mean_ndcg@1': float(mean_ndcg),
                'top1_correct_count': int(top1_correct),
                'top3_correct_count': int(top3_correct)
            },
            'details': results
        }

        return output_results


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("G1専用モデルテスト (2025年G1データ)")
    logging.info("="*60)

    # テスター初期化
    tester = G1SpecializedModelTester(model_dir='models/g1_specialized')

    # モデル読み込み
    if not tester.load_models():
        logging.error("❌ モデルの読み込みに失敗しました")
        return

    # 2025年G1データでテスト
    test_data_path = "data/processed/g1_races_2025_processed.csv"
    logging.info(f"\nテストデータ: {test_data_path}")

    results = tester.test_on_2025_data(test_data_path)

    # 結果保存
    output_dir = Path('models/test_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'test_2025_g1_specialized_model_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n結果保存: {results_file}")

    # ベースラインとの比較
    logging.info("\n" + "="*60)
    logging.info("ベースラインとの比較")
    logging.info("="*60)

    # Original 5-Fold Model
    original_results_path = output_dir / 'test_2025_g1_original_model_results.json'
    if original_results_path.exists():
        with open(original_results_path, 'r', encoding='utf-8') as f:
            original_results = json.load(f)

        logging.info("\n【Original 5-Fold Model】")
        logging.info(f"  Top1的中率: {original_results['summary']['top1_accuracy']:.1%}")
        logging.info(f"  Top3的中率: {original_results['summary']['top3_accuracy']:.1%}")

    # Optimized 7-Fold Model (if exists)
    optimized_results_path = output_dir / 'test_2025_g1_optimized_model_results.json'
    if optimized_results_path.exists():
        with open(optimized_results_path, 'r', encoding='utf-8') as f:
            optimized_results = json.load(f)

        logging.info("\n【Optimized 7-Fold Model】")
        logging.info(f"  Top1的中率: {optimized_results['summary']['top1_accuracy']:.1%}")
        logging.info(f"  Top3的中率: {optimized_results['summary']['top3_accuracy']:.1%}")

    logging.info("\n【G1 Specialized Model (NEW)】")
    logging.info(f"  Top1的中率: {results['summary']['top1_accuracy']:.1%}")
    logging.info(f"  Top3的中率: {results['summary']['top3_accuracy']:.1%}")

    # 改善度の計算
    if original_results_path.exists():
        improvement_top1 = results['summary']['top1_accuracy'] - original_results['summary']['top1_accuracy']
        improvement_top3 = results['summary']['top3_accuracy'] - original_results['summary']['top3_accuracy']

        logging.info("\n【改善度】")
        logging.info(f"  Top1: {improvement_top1:+.1%} ポイント")
        logging.info(f"  Top3: {improvement_top3:+.1%} ポイント")

    logging.info("\n✅ テスト完了!")


if __name__ == "__main__":
    main()

"""
LightGBM 7-Foldモデル 2025年G1レーステスト

訓練データ: 2010-2024年5月 (4,852レコード)
テストデータ: 2025年G1レース (5レース、88頭)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LightGBMTester2025:
    """2025年G1レースでLightGBMテスト"""

    def __init__(self, model_dir='models/optimized'):
        self.model_dir = Path(model_dir)
        self.models = []
        self.feature_names = []
        self.load_models()

    def load_models(self):
        """7-Foldモデル読み込み"""
        logging.info("=" * 60)
        logging.info("LightGBM 7-Foldモデル読み込み")
        logging.info("=" * 60)

        for i in range(1, 8):
            model_path = self.model_dir / f'optimized_model_fold{i}.txt'
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                self.models.append(model)
                logging.info(f"  Fold {i} 読み込み完了")

        if self.models:
            self.feature_names = self.models[0].feature_name()
            logging.info(f"\nモデル数: {len(self.models)}")
            logging.info(f"特徴量数: {len(self.feature_names)}")
        else:
            raise ValueError("モデルが見つかりません")

    def prepare_features(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """特徴量準備"""
        available_features = [f for f in self.feature_names if f in race_data.columns]

        if len(available_features) < len(self.feature_names) * 0.3:
            logging.warning(f"⚠️ 特徴量不足: {len(available_features)}/{len(self.feature_names)}")
            logging.info(f"利用可能な特徴量: {available_features[:10]}...")

        X = race_data[available_features].copy()

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X.loc[:, col] = X[col].fillna(median_val)

        return X

    def predict_race(self, race_data: pd.DataFrame):
        """レース予測 (7-Fold Ensemble)"""
        X = self.prepare_features(race_data)

        # 各Foldで予測
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # アンサンブル平均
        ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def evaluate_race(self, race_data: pd.DataFrame, race_name: str):
        """単一レース評価"""
        if len(race_data) < 2:
            logging.warning(f"⚠️ {race_name}: 出走頭数不足 ({len(race_data)}頭)")
            return None

        # 予測
        predictions = self.predict_race(race_data)

        # ランキング
        predicted_order = np.argsort(-predictions)

        # 実際の順位
        true_ranks = race_data['finish_position'].values
        true_winner_idx = true_ranks.argmin()

        # 予測結果
        pred_winner_idx = predicted_order[0]
        pred_winner_name = race_data.iloc[pred_winner_idx]['horse_name']
        true_winner_name = race_data.iloc[true_winner_idx]['horse_name']

        # Top1, Top3精度
        top1_correct = (pred_winner_idx == true_winner_idx)
        top3_correct = true_winner_idx in predicted_order[:3]

        # NDCG@1
        ndcg_at_1 = 1.0 if top1_correct else 0.0

        result = {
            'race_name': race_name,
            'race_id': race_data.iloc[0]['race_id'] if 'race_id' in race_data.columns else 'Unknown',
            'horses': len(race_data),
            'predicted_winner': pred_winner_name,
            'actual_winner': true_winner_name,
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'ndcg@1': ndcg_at_1,
            'predicted_top3': [race_data.iloc[idx]['horse_name'] for idx in predicted_order[:3]],
            'actual_top3': [race_data.iloc[idx]['horse_name'] for idx in np.argsort(true_ranks)[:3]]
        }

        return result

    def test_on_2025_data(self, data_path: str):
        """2025年G1レースでテスト"""
        logging.info("=" * 60)
        logging.info("2025年G1レース テスト開始")
        logging.info("=" * 60)

        # データ読み込み
        try:
            df = pd.read_csv(data_path, encoding='utf-8-sig')
            logging.info(f"\nデータ読み込み: {len(df)}レコード")
            logging.info(f"カラム数: {len(df.columns)}")

            # レース情報
            race_ids = df['race_id'].unique()
            logging.info(f"レース数: {len(race_ids)}")
        except FileNotFoundError:
            logging.error(f"❌ ファイルが見つかりません: {data_path}")
            return None

        # 各レースで評価
        results = []
        race_names = {
            '202505010811': 'フェブラリーS',
            '202505021211': '日本ダービー',
            '202505030211': '安田記念',
            '202506030811': '皐月賞',
            '202509020611': '桜花賞',
        }

        for race_id in race_ids:
            race_data = df[df['race_id'] == race_id].copy()
            race_name = race_names.get(race_id, race_id)

            result = self.evaluate_race(race_data, race_name)
            if result:
                results.append(result)

        # 集計
        if results:
            total_races = len(results)
            top1_correct = sum(r['top1_correct'] for r in results)
            top3_correct = sum(r['top3_correct'] for r in results)
            mean_ndcg = np.mean([r['ndcg@1'] for r in results])

            summary = {
                'total_races': total_races,
                'top1_accuracy': top1_correct / total_races,
                'top3_accuracy': top3_correct / total_races,
                'mean_ndcg@1': mean_ndcg,
                'top1_correct_count': top1_correct,
                'top3_correct_count': top3_correct
            }

            # 結果表示
            logging.info("\n" + "=" * 60)
            logging.info("2025年G1レース テスト結果")
            logging.info("=" * 60)
            logging.info(f"\n総レース数: {total_races}")
            logging.info(f"Top1精度: {summary['top1_accuracy']:.1%} ({top1_correct}/{total_races})")
            logging.info(f"Top3精度: {summary['top3_accuracy']:.1%} ({top3_correct}/{total_races})")
            logging.info(f"NDCG@1: {mean_ndcg:.4f}")

            logging.info("\n個別レース結果:")
            for r in results:
                status = "✅" if r['top1_correct'] else "❌"
                logging.info(f"{status} {r['race_name']}")
                logging.info(f"   予測: {r['predicted_winner']} | 実際: {r['actual_winner']}")

            # 訓練データとの比較
            logging.info("\n" + "=" * 60)
            logging.info("訓練データとの比較")
            logging.info("=" * 60)
            logging.info(f"7-Fold CV NDCG@1: 92.6%")
            logging.info(f"2025年テスト NDCG@1: {mean_ndcg:.1%}")
            diff = mean_ndcg - 0.926
            logging.info(f"差分: {diff:+.1%}")

            return {'summary': summary, 'details': results}
        else:
            logging.warning("⚠️ テスト可能なレースがありませんでした")
            return None


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("LightGBMモデル 2025年G1テスト")
    logging.info("=" * 60)

    # テスター初期化
    tester = LightGBMTester2025(model_dir='models/optimized')

    # 2025年データでテスト（前処理済みデータを使用）
    data_path = "data/processed/g1_races_2025_processed.csv"

    logging.info(f"\n使用するデータ: {data_path}")

    results = tester.test_on_2025_data(data_path)

    if results:
        # 結果保存
        output_dir = Path("models/test_results")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "test_2025_g1_results.json"

        # int64などをPython標準型に変換
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model': 'LightGBM 7-Fold Ensemble',
                'test_period': '2025 G1 Races (5 races)',
                **serializable_results
            }, f, indent=2, ensure_ascii=False)

        logging.info(f"\n結果保存: {output_file}")
    else:
        logging.error("\n❌ テスト失敗")


if __name__ == "__main__":
    main()

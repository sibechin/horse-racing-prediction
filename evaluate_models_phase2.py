"""
Phase 2: 2025年G1データで既存モデルを評価
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from datetime import datetime

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


def test_lightgbm_model(model_file, test_file):
    """LightGBMモデルをテスト"""
    logging.info(f"\nLightGBMモデルのテスト: {Path(model_file).name}")

    # モデル読み込み
    model = lgb.Booster(model_file=model_file)

    # テストデータ読み込み
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')

    # 特徴量（オッズ含む）
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

    available_features = [col for col in feature_columns if col in df_test.columns]
    X_test = df_test[available_features].copy()
    y_test = df_test['finish_position_numeric'].copy()
    race_groups_test = df_test['race_id'].copy()

    # 予測
    y_pred = model.predict(X_test)

    # 評価
    metrics = evaluate_ranking_model(y_test, y_pred, race_groups_test)

    return metrics, y_pred


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("Phase 2: 2025年G1データでモデル評価")
    logging.info("="*60)

    # ファイルパス
    test_file = 'data/processed/keibalab_g1_2025_processed.csv'

    # 利用可能なLightGBMモデル
    lightgbm_models = [
        'models/lightgbm_lambdarank_optimized.txt',
        'models/lightgbm_lambdarank_best.txt',
        'models/g1_specialized/g1_specialized_fold1.txt'
    ]

    # テストデータの確認
    if not Path(test_file).exists():
        logging.error(f"テストデータが見つかりません: {test_file}")
        return

    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    logging.info(f"\nテストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races")
    logging.info(f"レース種類: {df_test['expected_race_name'].nunique()}")

    # 全モデルのテスト結果を保存
    results = {}

    # LightGBMモデルのテスト
    for model_file in lightgbm_models:
        if Path(model_file).exists():
            model_name = Path(model_file).stem
            try:
                metrics, _ = test_lightgbm_model(model_file, test_file)
                results[model_name] = metrics
                logging.info(f"  NDCG@3: {metrics['ndcg@3']:.4f}, Top1: {metrics['top1_accuracy']*100:.2f}%, Top3: {metrics['top3_hit_rate']*100:.2f}%")
            except Exception as e:
                logging.error(f"  エラー: {e}")
        else:
            logging.warning(f"モデルが見つかりません: {model_file}")

    # 結果の比較
    if results:
        logging.info("\n" + "="*60)
        logging.info("2025年G1テストデータ性能比較")
        logging.info("="*60)

        comparison = pd.DataFrame(results).T
        comparison['NDCG@3'] = comparison['ndcg@3']
        comparison['1着的中率(%)'] = comparison['top1_accuracy'] * 100
        comparison['3着内的中率(%)'] = comparison['top3_hit_rate'] * 100
        comparison = comparison[['NDCG@3', '1着的中率(%)', '3着内的中率(%)']]

        print("\n")
        print(comparison.to_string())

        # 最良モデルの特定
        best_ndcg = comparison['NDCG@3'].idxmax()
        best_top1 = comparison['1着的中率(%)'].idxmax()
        best_top3 = comparison['3着内的中率(%)'].idxmax()

        logging.info(f"\n最良モデル:")
        logging.info(f"  NDCG@3: {best_ndcg} ({comparison.loc[best_ndcg, 'NDCG@3']:.4f})")
        logging.info(f"  1着的中率: {best_top1} ({comparison.loc[best_top1, '1着的中率(%)']:.2f}%)")
        logging.info(f"  3着内的中率: {best_top3} ({comparison.loc[best_top3, '3着内的中率(%)']:.2f}%)")

        # ランダム予測との比較
        num_races = df_test['race_id'].nunique()
        avg_field_size = df_test.groupby('race_id').size().mean()
        random_top1 = (1 / avg_field_size) * 100

        logging.info(f"\nランダム予測（ベースライン）:")
        logging.info(f"  1着的中率: {random_top1:.2f}%")
        logging.info(f"  平均出走頭数: {avg_field_size:.1f}")

        # 結果をファイルに保存
        output_dir = Path('models/comparison')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'phase2_results_{timestamp}.txt'

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Phase 2: 2025年G1データ評価結果\n")
            f.write("="*60 + "\n\n")
            f.write(f"テストデータ: {len(df_test)} records, {num_races} races\n")
            f.write(f"平均出走頭数: {avg_field_size:.1f}\n\n")
            f.write("モデル性能比較:\n")
            f.write(comparison.to_string())
            f.write("\n\n")
            f.write(f"最良モデル:\n")
            f.write(f"  NDCG@3: {best_ndcg} ({comparison.loc[best_ndcg, 'NDCG@3']:.4f})\n")
            f.write(f"  1着的中率: {best_top1} ({comparison.loc[best_top1, '1着的中率(%)']:.2f}%)\n")
            f.write(f"  3着内的中率: {best_top3} ({comparison.loc[best_top3, '3着内的中率(%)']:.2f}%)\n\n")
            f.write(f"ランダム予測（ベースライン）:\n")
            f.write(f"  1着的中率: {random_top1:.2f}%\n")

        logging.info(f"\n結果保存: {results_file}")
    else:
        logging.error("テスト可能なモデルが見つかりませんでした")

    logging.info("\n" + "="*60)
    logging.info("Phase 2評価完了!")
    logging.info("="*60)


if __name__ == "__main__":
    main()

"""
三連複・三連単用の評価スクリプト

実用的な馬券購入に役立つ評価指標を出力
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from itertools import permutations
from transformer_top1_v2 import TransformerTop1TrainerV2
from hybrid_transformer_lgbm import HybridTransformerLGBM
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_trifecta(model_name, predict_func, df_test):
    """
    三連複・三連単の評価

    Returns:
        dict: 各種的中率
    """
    results = {
        'top1': 0,           # 1着的中
        'top2_exact': 0,     # 1-2着順通り（馬単用）
        'top2_any': 0,       # 1-2着順不同（馬連用）
        'top3_exact': 0,     # 1-2-3着順通り（三連単）
        'top3_any': 0,       # 1-2-3着順不同（三連複）
        'top3_2hit': 0,      # Top3中2頭的中
        'top3_1hit': 0,      # Top3中1頭的中
        'top5_3hit': 0,      # Top5中に実際Top3が全て含まれる（ワイドBOX用）
    }
    total = 0

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]

        try:
            scores = predict_func(race_data)
            positions = race_data['finish_position_numeric'].values

            # 予測順位（スコア高い順）
            pred_ranking = np.argsort(scores)[::-1]

            # 実際順位（着順低い順）
            actual_ranking = np.argsort(positions)

            # Top1
            if pred_ranking[0] == actual_ranking[0]:
                results['top1'] += 1

            # Top2 (馬単・馬連)
            pred_top2 = pred_ranking[:2]
            actual_top2 = actual_ranking[:2]

            if list(pred_top2) == list(actual_top2):
                results['top2_exact'] += 1

            if set(pred_top2) == set(actual_top2):
                results['top2_any'] += 1

            # Top3 (三連単・三連複)
            pred_top3 = pred_ranking[:3]
            actual_top3 = actual_ranking[:3]

            if list(pred_top3) == list(actual_top3):
                results['top3_exact'] += 1

            if set(pred_top3) == set(actual_top3):
                results['top3_any'] += 1

            # Top3の部分的中
            hits = len(set(pred_top3) & set(actual_top3))
            if hits == 2:
                results['top3_2hit'] += 1
            elif hits == 1:
                results['top3_1hit'] += 1

            # Top5にTop3全て含まれるか
            pred_top5 = set(pred_ranking[:5])
            if set(actual_top3).issubset(pred_top5):
                results['top5_3hit'] += 1

            total += 1

        except Exception as e:
            logging.warning(f"Race {race_id}: {e}")

    return results, total


def print_results(model_name, results, total):
    """結果を整形して表示"""
    print(f"\n{'='*60}")
    print(f" {model_name} ({total}レース)")
    print(f"{'='*60}")
    print(f"\n【単勝・複勝系】")
    print(f"  単勝的中率:     {results['top1']/total:6.1%} ({results['top1']}/{total})")

    print(f"\n【馬連・馬単系】")
    print(f"  馬単的中率:     {results['top2_exact']/total:6.1%} ({results['top2_exact']}/{total})")
    print(f"  馬連的中率:     {results['top2_any']/total:6.1%} ({results['top2_any']}/{total})")

    print(f"\n【三連単・三連複系】")
    print(f"  三連単的中率:   {results['top3_exact']/total:6.1%} ({results['top3_exact']}/{total})")
    print(f"  三連複的中率:   {results['top3_any']/total:6.1%} ({results['top3_any']}/{total})")

    print(f"\n【参考指標】")
    print(f"  Top3中2頭的中:  {results['top3_2hit']/total:6.1%} ({results['top3_2hit']}/{total})")
    print(f"  Top3中1頭的中:  {results['top3_1hit']/total:6.1%} ({results['top3_1hit']}/{total})")
    print(f"  Top5に3着内全含: {results['top5_3hit']/total:6.1%} ({results['top5_3hit']}/{total})")


def main():
    logging.info("="*60)
    logging.info("三連複・三連単評価")
    logging.info("="*60)

    # データ
    df_test = pd.read_csv('data/processed/keibalab_g1_2025_cleaned.csv', encoding='utf-8-sig')

    # モデルパス
    lgbm_path = Path("models/top3_no_odds/top3_no_odds_model_20251121_053558.txt")
    transformer_path = max(Path('models/transformer_top1_v2').glob('*.pth'), key=lambda p: p.stat().st_mtime)

    # 1. LightGBM単体
    lgbm_model = lgb.Booster(model_file=str(lgbm_path))
    feature_cols = [
        'year', 'month', 'day_of_week', 'venue_code', 'race_number',
        'sex_encoded', 'age', 'weight', 'last_3f', 'horse_weight_kg',
        'horse_weight_change', 'field_size', 'race_avg_odds', 'race_avg_weight',
        'trainer_encoded', 'trainer_frequency', 'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]

    def predict_lgbm(race_data):
        X = race_data[[c for c in feature_cols if c in race_data.columns]]
        return lgbm_model.predict(X)

    results_lgbm, total = evaluate_trifecta("LightGBM", predict_lgbm, df_test)
    print_results("LightGBM", results_lgbm, total)

    # 2. Transformer単体
    trainer = TransformerTop1TrainerV2.load(transformer_path)

    def predict_transformer(race_data):
        return trainer.predict_race(race_data)

    results_trans, total = evaluate_trifecta("Transformer", predict_transformer, df_test)
    print_results("Transformer", results_trans, total)

    # 3. ハイブリッド (max方式)
    hybrid = HybridTransformerLGBM()
    hybrid.load_models(lgbm_path, transformer_path)

    def predict_hybrid(race_data):
        return hybrid.predict(race_data, method='max')

    results_hybrid, total = evaluate_trifecta("Hybrid (max)", predict_hybrid, df_test)
    print_results("Hybrid (LightGBM + Transformer)", results_hybrid, total)

    # サマリー
    print(f"\n{'='*60}")
    print(" サマリー比較")
    print(f"{'='*60}")
    print(f"{'モデル':<25} {'単勝':>8} {'馬連':>8} {'三連複':>8} {'三連単':>8}")
    print("-"*60)

    for name, res in [("LightGBM", results_lgbm),
                      ("Transformer", results_trans),
                      ("Hybrid (max)", results_hybrid)]:
        print(f"{name:<25} {res['top1']/total:>7.1%} {res['top2_any']/total:>7.1%} "
              f"{res['top3_any']/total:>7.1%} {res['top3_exact']/total:>7.1%}")


if __name__ == "__main__":
    main()

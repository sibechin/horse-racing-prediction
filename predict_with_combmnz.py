"""
CombMNZ融合方法を使用した予測スクリプト

動的ハイブリッド融合システムのCombMNZ方式（Top3適中率80%）を使用
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dynamic_hybrid_fusion import DynamicHybridFusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_race(system, race_data, race_name, fusion_method='combmnz'):
    """
    レース予測を実行

    Args:
        system: DynamicHybridFusion インスタンス
        race_data: レースデータ（1レース分）
        race_name: レース名
        fusion_method: 融合方法（'combmnz', 'dat', 'equal'）
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"レース: {race_name}")
    logging.info(f"融合方法: {fusion_method.upper()}")
    logging.info(f"{'='*60}")

    # 予測
    scores = system.predict(race_data, fusion_method=fusion_method)

    if scores is None:
        logging.error("予測に失敗しました")
        return None

    # ランキング作成（スコアの降順）
    rankings = np.argsort(scores)[::-1]

    # 結果表示用データ
    results = []

    for rank, idx in enumerate(rankings, 1):
        horse_info = {
            'rank': rank,
            'score': scores[idx],
            'horse_number': int(race_data.iloc[idx].get('horse_number', idx + 1)),
        }

        # 馬名（利用可能な場合）
        if 'horse_name' in race_data.columns:
            horse_info['horse_name'] = race_data.iloc[idx]['horse_name']

        # オッズ（利用可能な場合）
        if 'odds' in race_data.columns:
            horse_info['odds'] = race_data.iloc[idx]['odds']

        # 実際の着順（テストデータの場合）
        if 'finish_position' in race_data.columns:
            horse_info['actual_position'] = race_data.iloc[idx]['finish_position']
        elif 'finish_position_numeric' in race_data.columns:
            horse_info['actual_position'] = int(race_data.iloc[idx]['finish_position_numeric'])

        results.append(horse_info)

    # 結果表示
    df_results = pd.DataFrame(results)

    logging.info(f"\n予測ランキング（Top10）:")
    print("\n" + df_results.head(10).to_string(index=False))

    # 実際の結果との比較（利用可能な場合）
    if 'actual_position' in df_results.columns:
        logging.info(f"\n実際の着順との比較:")

        # Top3の予測精度
        predicted_top3 = set(df_results.head(3)['horse_number'])
        actual_top3 = set(df_results[df_results['actual_position'] <= 3]['horse_number'])

        hit_count = len(predicted_top3 & actual_top3)
        logging.info(f"  予測Top3: {predicted_top3}")
        logging.info(f"  実際Top3: {actual_top3}")
        logging.info(f"  的中数: {hit_count}/3")

        # Top1の予測精度
        predicted_winner = int(df_results.iloc[0]['horse_number'])
        actual_winner = int(df_results[df_results['actual_position'] == 1]['horse_number'].iloc[0])

        if predicted_winner == actual_winner:
            logging.info(f"  ✓ Top1的中！")
        else:
            logging.info(f"  Top1予測: {predicted_winner}, 実際: {actual_winner}")

    return df_results


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("CombMNZ融合予測システム")
    logging.info("="*60)

    # データ読み込み
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")

    if not test_file.exists():
        logging.error(f"データファイルが見つかりません: {test_file}")
        return

    logging.info(f"\nデータ読み込み: {test_file}")
    df = pd.read_csv(test_file, encoding='utf-8-sig')

    # finish_position_numericの確認
    if 'finish_position_numeric' not in df.columns:
        df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

    # システム初期化
    system = DynamicHybridFusion()

    # モデル読み込み
    system.load_models(
        path_no_odds='models/top3_no_odds/top3_no_odds_model_20251121_053558.txt',
        path_with_odds='models/statistical/conditional_logit_20251121_054350.pkl'
    )

    # 訓練データでDATを訓練（CombMNZには不要だが、システムの完全性のため）
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")
    if train_file.exists():
        logging.info(f"DAT訓練用データ読み込み: {train_file}")
        df_train = pd.read_csv(train_file, encoding='utf-8-sig')
        if 'finish_position_numeric' not in df_train.columns:
            df_train['finish_position_numeric'] = pd.to_numeric(df_train['finish_position'], errors='coerce')
        system.train_dat(df_train)

    # 利用可能なレース一覧
    races = df[['race_id', 'expected_race_name', 'expected_date']].drop_duplicates().sort_values('expected_date')

    logging.info(f"\n利用可能なレース数: {len(races)}")
    logging.info("\n最新のレース:")
    print(races.tail(5).to_string(index=False))

    # 最新レース（安田記念）で予測
    latest_race_id = races.iloc[-1]['race_id']
    latest_race_name = races.iloc[-1]['expected_race_name']

    race_data = df[df['race_id'] == latest_race_id]

    # CombMNZ融合で予測
    results = predict_race(system, race_data, latest_race_name, fusion_method='combmnz')

    # 比較のため、他の融合方法も実行
    logging.info(f"\n\n{'='*60}")
    logging.info("比較: DAT融合")
    logging.info(f"{'='*60}")
    predict_race(system, race_data, latest_race_name, fusion_method='dat')

    logging.info(f"\n\n{'='*60}")
    logging.info("比較: 均等重み融合")
    logging.info(f"{'='*60}")
    predict_race(system, race_data, latest_race_name, fusion_method='equal')

    logging.info("\n完了!")


if __name__ == "__main__":
    main()

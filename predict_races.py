"""
レース予測スクリプト
Phase 2で訓練したモデルを使用して、新しいレースの予測を行います
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import lightgbm as lgb
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_path):
    """モデルを読み込む"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    model = lgb.Booster(model_file=model_path)
    logging.info(f"モデル読み込み完了: {model_path}")
    return model


def prepare_features(race_data):
    """
    レースデータから特徴量を準備

    必要なカラム:
    - year, month, day_of_week
    - venue_code, race_number
    - sex_encoded, age
    - weight, popularity, odds_numeric
    - last_3f, horse_weight_kg, horse_weight_change
    - field_size, race_avg_odds, race_avg_weight
    - trainer_encoded, trainer_frequency
    - jockey_encoded, jockey_frequency
    - race_type_encoded
    """
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

    # 使用可能な特徴量のみを選択
    available_features = [col for col in feature_columns if col in race_data.columns]

    if len(available_features) < 10:
        logging.warning(f"特徴量が不足しています。利用可能: {len(available_features)}個")

    return race_data[available_features]


def predict_race(model, race_data, race_info=None):
    """
    単一レースの予測

    Args:
        model: LightGBMモデル
        race_data: レースデータ（DataFrame）
        race_info: レース情報（辞書、オプション）

    Returns:
        予測結果（DataFrame）
    """
    # 特徴量の準備
    X = prepare_features(race_data)

    # 予測
    predictions = model.predict(X)

    # 結果をDataFrameに追加
    result = race_data.copy()
    result['predicted_score'] = predictions
    result['predicted_rank'] = result['predicted_score'].rank(ascending=False, method='dense').astype(int)

    # スコアでソート（降順）
    result = result.sort_values('predicted_score', ascending=False).reset_index(drop=True)

    return result


def format_prediction_output(predictions, race_name=None, race_date=None):
    """予測結果を見やすく整形"""
    output = []

    if race_name:
        output.append(f"\n{'='*60}")
        output.append(f"レース: {race_name}")
    if race_date:
        output.append(f"開催日: {race_date}")
    output.append(f"{'='*60}")

    # 予測順位でソート
    predictions_sorted = predictions.sort_values('predicted_rank')

    output.append(f"\n{'順位':^6} {'馬名':^20} {'予測スコア':^12} {'オッズ':^8} {'人気':^6}")
    output.append("-" * 60)

    for idx, row in predictions_sorted.iterrows():
        horse_name = row.get('horse_name', f'馬{idx+1}')
        pred_rank = int(row['predicted_rank'])
        pred_score = row['predicted_score']
        odds = row.get('odds_numeric', '-')
        popularity = row.get('popularity', '-')

        odds_str = f"{odds:.1f}" if isinstance(odds, (int, float)) else str(odds)
        pop_str = f"{int(popularity)}" if isinstance(popularity, (int, float)) else str(popularity)

        output.append(f"{pred_rank:^6} {horse_name:^20} {pred_score:^12.4f} {odds_str:^8} {pop_str:^6}")

    output.append("\n" + "="*60)

    # Top3を強調表示
    top3 = predictions_sorted.head(3)
    output.append("\n🏆 予想 Top 3:")
    for i, (idx, row) in enumerate(top3.iterrows(), 1):
        horse_name = row.get('horse_name', f'馬{idx+1}')
        output.append(f"  {i}位: {horse_name} (スコア: {row['predicted_score']:.4f})")

    return '\n'.join(output)


def predict_from_csv(model_path, csv_path, output_path=None):
    """
    CSVファイルからレース予測を実行

    Args:
        model_path: モデルファイルのパス
        csv_path: レースデータCSVファイルのパス
        output_path: 出力先（オプション）
    """
    # モデル読み込み
    model = load_model(model_path)

    # データ読み込み
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    logging.info(f"データ読み込み完了: {len(df)} レコード")

    # レースごとにグループ化
    if 'race_id' in df.columns:
        race_groups = df.groupby('race_id')
        logging.info(f"レース数: {len(race_groups)}")
    else:
        # race_idがない場合は、全体を1レースとして扱う
        race_groups = [(0, df)]

    all_results = []

    for race_id, race_data in race_groups:
        logging.info(f"\nレースID {race_id} の予測中...")

        # レース情報
        race_name = race_data.iloc[0].get('expected_race_name', race_data.iloc[0].get('race_name', f'レース{race_id}'))
        race_date = race_data.iloc[0].get('expected_date', race_data.iloc[0].get('race_date', ''))

        # 予測
        predictions = predict_race(model, race_data)
        predictions['race_id'] = race_id

        # 結果を表示
        output_text = format_prediction_output(predictions, race_name, race_date)
        print(output_text)

        all_results.append(predictions)

    # 全結果を結合
    final_results = pd.concat(all_results, ignore_index=True)

    # 結果を保存
    if output_path:
        final_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"\n予測結果を保存: {output_path}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_output = f'predictions/predictions_{timestamp}.csv'
        Path('predictions').mkdir(exist_ok=True)
        final_results.to_csv(default_output, index=False, encoding='utf-8-sig')
        logging.info(f"\n予測結果を保存: {default_output}")

    return final_results


def create_sample_race_data():
    """
    サンプルレースデータを作成（テスト用）
    """
    sample_data = {
        'race_id': ['20251123_tokyo_11'] * 10,
        'race_name': ['ジャパンカップ'] * 10,
        'race_date': ['2025-11-23'] * 10,
        'horse_name': [f'馬{i+1}' for i in range(10)],
        'year': [2025] * 10,
        'month': [11] * 10,
        'day_of_week': [6] * 10,  # 土曜日
        'venue_code': [5] * 10,  # 東京
        'race_number': [11] * 10,
        'sex_encoded': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],  # 牡/牝
        'age': [4, 5, 3, 4, 4, 5, 3, 4, 5, 4],
        'weight': [57.0, 55.0, 56.0, 57.0, 55.0, 57.0, 55.0, 57.0, 57.0, 55.0],
        'popularity': list(range(1, 11)),
        'odds_numeric': [3.5, 5.2, 8.1, 12.3, 15.7, 20.4, 25.8, 35.2, 48.9, 65.4],
        'last_3f': [35.0, 35.5, 34.8, 35.2, 35.8, 36.0, 35.3, 35.9, 36.2, 36.5],
        'horse_weight_kg': [480, 465, 490, 475, 470, 485, 460, 478, 482, 468],
        'horse_weight_change': [0, 2, -3, 5, -1, 0, 3, -2, 1, -4],
        'field_size': [10] * 10,
        'race_avg_odds': [25.0] * 10,
        'race_avg_weight': [56.0] * 10,
        'trainer_encoded': [1, 5, 3, 8, 2, 4, 6, 7, 9, 10],
        'trainer_frequency': [50, 45, 60, 40, 55, 48, 42, 38, 35, 32],
        'jockey_encoded': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'jockey_frequency': [100, 95, 110, 85, 90, 88, 82, 75, 70, 65],
        'race_type_encoded': [0] * 10
    }

    return pd.DataFrame(sample_data)


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='競馬レース予測')
    parser.add_argument('--model', type=str, default='models/phase2/lgbm_2025_model_20251122_161840.txt',
                        help='モデルファイルのパス')
    parser.add_argument('--input', type=str, help='入力CSVファイル（レースデータ）')
    parser.add_argument('--output', type=str, help='出力CSVファイル')
    parser.add_argument('--sample', action='store_true', help='サンプルデータでテスト')

    args = parser.parse_args()

    logging.info("="*60)
    logging.info("競馬レース予測システム")
    logging.info("="*60)

    if args.sample:
        # サンプルデータで予測
        logging.info("\nサンプルデータで予測を実行...")
        sample_data = create_sample_race_data()
        sample_csv = 'predictions/sample_race.csv'
        Path('predictions').mkdir(exist_ok=True)
        sample_data.to_csv(sample_csv, index=False, encoding='utf-8-sig')
        predict_from_csv(args.model, sample_csv, args.output)
    elif args.input:
        # 指定されたCSVファイルで予測
        predict_from_csv(args.model, args.input, args.output)
    else:
        logging.error("--input または --sample を指定してください")
        parser.print_help()


if __name__ == "__main__":
    main()

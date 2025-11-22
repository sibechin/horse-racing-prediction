"""
マイルチャンピオンシップ（マイルCS）予測スクリプト

訓練済みTop3予測モデル（オッズ除外版）を使用して、
マイルCSの出走馬を予測します。
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_trained_model():
    """訓練済みモデルの読み込み"""
    model_dir = Path("models/top3_no_odds")

    # 最新のモデルファイルを取得
    model_files = list(model_dir.glob("top3_no_odds_model_*.txt"))
    if not model_files:
        raise FileNotFoundError("訓練済みモデルが見つかりません")

    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logging.info(f"モデル読み込み: {latest_model}")

    model = lgb.Booster(model_file=str(latest_model))

    # 特徴量リストを取得
    feature_names = model.feature_name()
    logging.info(f"使用特徴量: {len(feature_names)}個")

    return model, feature_names


def fetch_mile_cs_data():
    """
    マイルCS出走馬データの取得

    Returns:
        DataFrame: 出走馬データ
    """
    # TODO: 実際のデータ取得方法に応じて実装
    # オプション1: Webスクレイピング（netkeiba.com, keibalab.jp等）
    # オプション2: CSV入力ファイル
    # オプション3: 手動入力

    logging.info("マイルCS出走馬データを取得中...")

    # 仮実装: CSVファイルから読み込み
    data_file = Path("data/mile_cs_2025.csv")

    if data_file.exists():
        df = pd.read_csv(data_file, encoding='utf-8-sig')
        logging.info(f"データ読み込み完了: {len(df)}頭")
        return df
    else:
        logging.warning(f"{data_file} が見つかりません")
        logging.info("サンプルテンプレートを作成します...")

        # サンプルテンプレート作成
        sample_df = create_sample_template()
        sample_df.to_csv(data_file, index=False, encoding='utf-8-sig')
        logging.info(f"テンプレート作成: {data_file}")
        logging.info("このファイルに出走馬情報を入力して再実行してください")

        return None


def create_sample_template():
    """
    入力用サンプルテンプレートの作成

    Returns:
        DataFrame: サンプルテンプレート
    """
    # Top10重要特徴量ベースのテンプレート
    template_data = {
        'horse_name': ['馬名1', '馬名2', '馬名3'],  # 馬名（予測には不使用）
        'last_3f': [33.5, 34.0, 33.8],  # 最後3ハロン（最重要）
        'age': [4, 5, 3],  # 馬齢
        'jockey_frequency': [150, 200, 100],  # 騎手騎乗回数
        'field_size': [18, 18, 18],  # 出走頭数
        'horse_weight_kg': [480, 470, 490],  # 馬体重
        'weight_change': [0, -2, 4],  # 馬体重増減
        'horse_frequency': [20, 30, 10],  # 馬の出走回数
        'jockey_win_rate': [0.15, 0.20, 0.10],  # 騎手勝率
        'corner_rank_3': [5, 3, 7],  # 3コーナー順位
        'distance': [1600, 1600, 1600],  # 距離（マイルCS固定）
        'corner_rank_4': [4, 2, 6],  # 4コーナー順位
        'trainer_frequency': [100, 150, 80],  # 調教師管理回数
        'track_condition_encoded': [1, 1, 1],  # 馬場状態（1=良）
        'corner_rank_2': [6, 4, 8],  # 2コーナー順位
        'corner_rank_1': [7, 5, 9],  # 1コーナー順位
        'race_class_encoded': [5, 5, 5],  # レースクラス（5=G1）
        'surface_encoded': [1, 1, 1],  # 芝/ダート（1=芝）
        'horse_win_rate': [0.10, 0.15, 0.05],  # 馬勝率
        'trainer_win_rate': [0.12, 0.18, 0.08],  # 調教師勝率
    }

    return pd.DataFrame(template_data)


def prepare_features(df, feature_names):
    """
    特徴量の準備

    Args:
        df: 出走馬データ
        feature_names: モデルが期待する特徴量名リスト

    Returns:
        準備された特徴量DataFrame
    """
    # 必要な特徴量のみ抽出
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]

    if missing_features:
        logging.warning(f"不足している特徴量: {missing_features}")
        logging.info("不足特徴量は0で補完します")

        # 不足特徴量を0で補完
        for feature in missing_features:
            df[feature] = 0

    # 特徴量の順序を合わせる
    X = df[feature_names].copy()

    return X


def predict_top3(model, X, horse_names=None, horse_numbers=None):
    """
    Top3予測

    Args:
        model: 訓練済みモデル
        X: 特徴量
        horse_names: 馬名リスト（オプション）
        horse_numbers: 馬番リスト（オプション）

    Returns:
        予測結果DataFrame
    """
    # 予測スコア
    scores = model.predict(X)

    # Top3インデックス
    top3_indices = np.argsort(scores)[::-1][:3]

    # 結果DataFrame作成
    results = []

    for rank, idx in enumerate(top3_indices, 1):
        horse_name = horse_names[idx] if horse_names is not None else f"Horse {idx+1}"
        horse_num = horse_numbers[idx] if horse_numbers is not None else idx+1
        score = scores[idx]

        results.append({
            '予想順位': rank,
            '馬番': horse_num,
            '馬名': horse_name,
            '予測スコア': f"{score:.4f}",
            '確率': f"{score * 100:.2f}%"
        })

    # 全順位も表示
    all_results = []
    for idx in np.argsort(scores)[::-1]:
        horse_name = horse_names[idx] if horse_names is not None else f"Horse {idx+1}"
        horse_num = horse_numbers[idx] if horse_numbers is not None else idx+1
        score = scores[idx]

        all_results.append({
            '馬番': horse_num,
            '馬名': horse_name,
            '予測スコア': score
        })

    return pd.DataFrame(results), pd.DataFrame(all_results)


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("マイルチャンピオンシップ（マイルCS）予測")
    logging.info("=" * 60)

    # モデル読み込み
    model, feature_names = load_trained_model()

    # データ取得
    df = fetch_mile_cs_data()

    if df is None:
        logging.error("データが取得できませんでした")
        logging.info("data/mile_cs_2025.csv にデータを入力して再実行してください")
        return

    # 馬名と馬番を保存
    horse_names = df['horse_name'].values if 'horse_name' in df.columns else None
    horse_numbers = df['horse_number'].values if 'horse_number' in df.columns else None

    # 特徴量準備
    X = prepare_features(df, feature_names)

    logging.info(f"\n出走頭数: {len(X)}頭")
    logging.info(f"使用特徴量: {len(feature_names)}個")

    # 予測実行
    logging.info("\n予測実行中...")
    top3_results, all_results = predict_top3(model, X, horse_names, horse_numbers)

    # 結果表示
    logging.info("\n" + "=" * 60)
    logging.info("🏆 マイルCS Top3予想結果")
    logging.info("=" * 60)
    print("\n", top3_results.to_string(index=False))

    logging.info("\n" + "=" * 60)
    logging.info("📊 全馬予測スコア（降順）")
    logging.info("=" * 60)
    print("\n", all_results.to_string(index=False))

    # 結果をCSVに保存
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"mile_cs_prediction_{timestamp}.csv"

    all_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"\n✓ 予測結果保存: {output_file}")

    # 使用した特徴量の値も保存
    feature_output = output_dir / f"mile_cs_features_{timestamp}.csv"
    if horse_names is not None:
        X_output = X.copy()
        X_output.insert(0, 'horse_name', horse_names)
        X_output.to_csv(feature_output, index=False, encoding='utf-8-sig')
        logging.info(f"✓ 特徴量保存: {feature_output}")


if __name__ == "__main__":
    main()

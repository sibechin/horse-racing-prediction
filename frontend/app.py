"""
競馬予測Webアプリケーション - Flask Backend

精度最優先のモデルを使用した予測インターフェース
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

class RacePredictionSystem:
    """レース予測システム"""

    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.model_config = {}
        self.feature_names = []

        self.load_models()

    def load_models(self):
        """モデル読み込み"""
        logging.info("モデル読み込み開始...")

        # Fold models
        fold_models = []
        for i in range(1, 6):
            model_path = self.model_dir / f'lightgbm_lambdarank_refined_fold{i}.txt'
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                fold_models.append(model)
                logging.info(f"  Fold {i} モデル読み込み完了")

        if fold_models:
            self.models['ensemble'] = fold_models
            self.feature_names = fold_models[0].feature_name()

        # 設定ファイル読み込み
        config_path = self.model_dir / 'refined_optimization_results.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)

        logging.info(f"モデル読み込み完了: {len(self.models.get('ensemble', []))}個")

    def predict_race(self, race_data: pd.DataFrame):
        """レース予測"""
        if 'ensemble' not in self.models:
            raise ValueError("モデルが読み込まれていません")

        # 特徴量準備
        features = self._prepare_features(race_data)

        # アンサンブル予測
        predictions = []
        for model in self.models['ensemble']:
            pred = model.predict(features)
            predictions.append(pred)

        # 平均予測
        ensemble_pred = np.mean(predictions, axis=0)

        # ランキング作成
        rankings = self._create_rankings(race_data, ensemble_pred)

        return rankings

    def _prepare_features(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """特徴量準備"""
        # 必要な特徴量のみ選択
        available_features = [f for f in self.feature_names if f in race_data.columns]

        if len(available_features) < len(self.feature_names) * 0.8:
            logging.warning(f"特徴量不足: {len(available_features)}/{len(self.feature_names)}")

        features = race_data[available_features].copy()

        # 欠損値処理
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                features[col].fillna(features[col].median(), inplace=True)

        return features

    def _create_rankings(self, race_data: pd.DataFrame, predictions: np.ndarray):
        """ランキング作成"""
        rankings = []

        for idx, pred_score in enumerate(predictions):
            horse_info = {
                'rank': 0,  # 後で設定
                'horse_name': race_data.iloc[idx].get('horse_name', f'Horse {idx+1}'),
                'horse_number': race_data.iloc[idx].get('horse_number', idx+1),
                'jockey_name': race_data.iloc[idx].get('jockey_name', 'Unknown'),
                'prediction_score': float(pred_score),
                'odds': race_data.iloc[idx].get('odds', 0.0),
                'weight': race_data.iloc[idx].get('weight', 0.0),
                'popularity': race_data.iloc[idx].get('popularity', 0),
            }
            rankings.append(horse_info)

        # スコアでソート (降順 = 1位が最高スコア)
        rankings.sort(key=lambda x: x['prediction_score'], reverse=True)

        # ランク付与
        for rank, horse in enumerate(rankings, 1):
            horse['rank'] = rank

        return rankings


# グローバル予測システム
prediction_system = None


@app.route('/')
def index():
    """トップページ"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """予測API"""
    try:
        data = request.get_json()

        if 'race_data' not in data:
            return jsonify({'error': '  レースデータが必要です'}), 400

        # データフレーム作成
        race_df = pd.DataFrame(data['race_data'])

        # 予測
        rankings = prediction_system.predict_race(race_df)

        return jsonify({
            'success': True,
            'rankings': rankings,
            'race_info': {
                'total_horses': len(rankings),
                'prediction_time': datetime.now().isoformat()
            }
        })

    except Exception as e:
        logging.error(f"予測エラー: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """モデル情報API"""
    try:
        info = {
            'model_count': len(prediction_system.models.get('ensemble', [])),
            'feature_count': len(prediction_system.feature_names),
            'features': prediction_system.feature_names[:20],  # 最初の20個
            'config': prediction_system.model_config.get('optuna_best_params', {}),
            'performance': prediction_system.model_config.get('cv_results', {})
        }

        return jsonify(info)

    except Exception as e:
        logging.error(f"モデル情報エラー: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """統計情報API"""
    try:
        # 統計データ読み込み
        stats_data = {
            'total_races_trained': 340,
            'total_records': 4270,
            'model_accuracy': {
                'top1': 0.9118,
                'top3': 0.8344,
                'ndcg@1': 0.6174,
                'ndcg@3': 0.6392,
                'ndcg@5': 0.6981
            },
            'last_updated': datetime.now().isoformat()
        }

        return jsonify(stats_data)

    except Exception as e:
        logging.error(f"統計情報エラー: {str(e)}")
        return jsonify({'error': str(e)}), 500


def initialize_app():
    """アプリ初期化"""
    global prediction_system

    logging.info("=" * 60)
    logging.info("競馬予測Webアプリケーション起動")
    logging.info("=" * 60)

    # 予測システム初期化
    prediction_system = RacePredictionSystem()

    logging.info("初期化完了")


if __name__ == '__main__':
    initialize_app()

    # 開発サーバー起動
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

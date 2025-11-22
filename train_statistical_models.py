"""
統計モデルの訓練
- Conditional Logit Model
- Simplified Elo Rating System
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConditionalLogitPredictor:
    """Conditional Logit Model"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = [
            'odds', 'popularity_rank', 'weight', 'age',
            'jockey_win_rate', 'horse_win_rate',
            'distance', 'track_type_encoded',
            'track_condition_encoded', 'weather_encoded'
        ]

    def fit(self, df: pd.DataFrame):
        """訓練"""
        logging.info("Conditional Logit Model 訓練開始...")

        # 利用可能な特徴量のみ使用
        available_features = [col for col in self.feature_cols if col in df.columns]
        logging.info(f"使用特徴量: {available_features}")

        X = df[available_features].copy()

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        # 標準化
        X_scaled = self.scaler.fit_transform(X)

        # ターゲット: 1着=1, それ以外=0
        if 'finish_position' in df.columns:
            y = (df['finish_position'] == 1).astype(int)
        elif 'finish_position_numeric' in df.columns:
            y = (df['finish_position_numeric'] == 1).astype(int)
        else:
            raise ValueError("finish_position カラムが見つかりません")

        # Logistic Regression
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # 不均衡データ対応
        )
        self.model.fit(X_scaled, y)

        # 訓練精度
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

        # 確率予測
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        logging.info(f"✓ Conditional Logit 訓練完了")
        logging.info(f"  訓練精度: {accuracy:.4f}")
        logging.info(f"  1着予測確率範囲: [{y_proba.min():.4f}, {y_proba.max():.4f}]")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """確率予測"""
        available_features = [col for col in self.feature_cols if col in df.columns]
        X = df[available_features].copy()

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        # 標準化
        X_scaled = self.scaler.transform(X)

        # 確率予測 (1着確率)
        proba = self.model.predict_proba(X_scaled)[:, 1]

        return proba

    def save(self, path: str):
        """モデル保存"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }, path)
        logging.info(f"✓ モデル保存: {path}")

    @classmethod
    def load(cls, path: str):
        """モデル読み込み"""
        data = joblib.load(path)
        predictor = cls()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_cols = data['feature_cols']
        return predictor


class SimplifiedEloRating:
    """簡易Eloレーティングシステム"""

    def __init__(self, k=15, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}

    def get_rating(self, horse_id: str) -> float:
        """馬のレーティング取得"""
        if horse_id not in self.ratings:
            self.ratings[horse_id] = self.initial_rating
        return self.ratings[horse_id]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """期待スコア計算"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, race_results: pd.DataFrame):
        """レース結果からレーティング更新"""
        if 'horse_id' not in race_results.columns:
            logging.warning("horse_id カラムがありません")
            return

        finish_col = 'finish_position' if 'finish_position' in race_results.columns else 'finish_position_numeric'

        if finish_col not in race_results.columns:
            logging.warning("finish_position カラムがありません")
            return

        # レース内の全ペアで更新
        for idx, horse_a in race_results.iterrows():
            horse_a_id = str(horse_a['horse_id'])
            horse_a_pos = horse_a[finish_col]

            for jdx, horse_b in race_results.iterrows():
                if idx == jdx:
                    continue

                horse_b_id = str(horse_b['horse_id'])
                horse_b_pos = horse_b[finish_col]

                # 実際の結果
                if horse_a_pos < horse_b_pos:
                    actual_score = 1.0
                elif horse_a_pos == horse_b_pos:
                    actual_score = 0.5
                else:
                    actual_score = 0.0

                # 期待スコア
                rating_a = self.get_rating(horse_a_id)
                rating_b = self.get_rating(horse_b_id)
                expected = self.expected_score(rating_a, rating_b)

                # レーティング更新
                self.ratings[horse_a_id] = rating_a + self.k * (actual_score - expected)

    def fit(self, df: pd.DataFrame):
        """訓練（時系列順にレーティング更新）"""
        logging.info("Elo Rating System 初期化開始...")

        # レースを時系列順にソート
        if 'race_date' in df.columns:
            df = df.sort_values('race_date')
        elif 'year' in df.columns and 'month' in df.columns:
            df = df.sort_values(['year', 'month'])

        # レースごとに更新
        race_col = 'race_id'
        if race_col not in df.columns:
            logging.error("race_id カラムがありません")
            return self

        total_races = df[race_col].nunique()
        logging.info(f"総レース数: {total_races}")

        for i, race_id in enumerate(df[race_col].unique(), 1):
            race_data = df[df[race_col] == race_id]
            self.update_ratings(race_data)

            if i % 500 == 0:
                logging.info(f"  処理済み: {i}/{total_races} レース")

        logging.info(f"✓ Elo Rating 初期化完了")
        logging.info(f"  登録馬数: {len(self.ratings)}")

        if self.ratings:
            ratings_array = np.array(list(self.ratings.values()))
            logging.info(f"  レーティング範囲: [{ratings_array.min():.0f}, {ratings_array.max():.0f}]")
            logging.info(f"  平均レーティング: {ratings_array.mean():.0f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """レース内の勝率予測"""
        if 'horse_id' not in df.columns:
            return np.ones(len(df)) / len(df)

        ratings = np.array([self.get_rating(str(hid)) for hid in df['horse_id']])

        # Softmaxで確率化
        exp_ratings = np.exp(ratings / 400)  # スケーリング
        probas = exp_ratings / exp_ratings.sum()

        return probas

    def save(self, path: str):
        """モデル保存"""
        joblib.dump({
            'ratings': self.ratings,
            'k': self.k,
            'initial_rating': self.initial_rating
        }, path)
        logging.info(f"✓ モデル保存: {path}")

    @classmethod
    def load(cls, path: str):
        """モデル読み込み"""
        data = joblib.load(path)
        elo = cls(k=data['k'], initial_rating=data['initial_rating'])
        elo.ratings = data['ratings']
        return elo


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("統計モデル訓練スクリプト")
    logging.info("="*60)

    # データ読み込み（クリーニング済み）
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")

    if not train_file.exists():
        logging.error(f"訓練データが見つかりません: {train_file}")
        return

    logging.info(f"\nデータ読み込み: {train_file}")
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    logging.info(f"  レコード数: {len(df)}")
    logging.info(f"  レース数: {df['race_id'].nunique() if 'race_id' in df.columns else 'N/A'}")
    logging.info(f"  カラム数: {len(df.columns)}")

    # 出力ディレクトリ
    output_dir = Path("models/statistical")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Conditional Logit Model
    logging.info("\n" + "="*60)
    logging.info("1. Conditional Logit Model")
    logging.info("="*60)

    logit_model = ConditionalLogitPredictor()
    logit_model.fit(df)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logit_path = output_dir / f"conditional_logit_{timestamp}.pkl"
    logit_model.save(logit_path)

    # 2. Elo Rating System
    logging.info("\n" + "="*60)
    logging.info("2. Elo Rating System")
    logging.info("="*60)

    elo_model = SimplifiedEloRating(k=15, initial_rating=1500)
    elo_model.fit(df)

    elo_path = output_dir / f"elo_rating_{timestamp}.pkl"
    elo_model.save(elo_path)

    # メタデータ保存
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "training_data": str(train_file),
        "num_records": len(df),
        "num_races": int(df['race_id'].nunique()) if 'race_id' in df.columns else 0,
        "models": {
            "conditional_logit": {
                "path": str(logit_path),
                "features": logit_model.feature_cols
            },
            "elo_rating": {
                "path": str(elo_path),
                "k": elo_model.k,
                "initial_rating": elo_model.initial_rating,
                "num_horses": len(elo_model.ratings)
            }
        }
    }

    metadata_path = output_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info("\n" + "="*60)
    logging.info("訓練完了！")
    logging.info("="*60)
    logging.info(f"Conditional Logit: {logit_path}")
    logging.info(f"Elo Rating: {elo_path}")
    logging.info(f"メタデータ: {metadata_path}")


if __name__ == "__main__":
    main()

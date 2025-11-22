"""
特徴量エンジニアリング
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
import category_encoders as ce

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """競馬データの特徴量エンジニアリング"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.label_encoders = {}
        self.target_encoder = None
        self.scaler = StandardScaler()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を作成

        Args:
            df: クリーニング済みのDataFrame

        Returns:
            特徴量が追加されたDataFrame
        """
        logger.info("Starting feature engineering")

        df_feat = df.copy()

        # 1. 馬の特徴量
        df_feat = self._create_horse_features(df_feat)

        # 2. 騎手の特徴量
        df_feat = self._create_jockey_features(df_feat)

        # 3. 調教師の特徴量
        df_feat = self._create_trainer_features(df_feat)

        # 4. レースの特徴量
        df_feat = self._create_race_features(df_feat)

        # 5. 過去成績の特徴量
        df_feat = self._create_historical_features(df_feat)

        # 6. 相互作用の特徴量
        df_feat = self._create_interaction_features(df_feat)

        logger.info(f"Feature engineering completed. Final columns: {df_feat.shape[1]}")

        return df_feat

    def _create_horse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬の特徴量を作成"""
        logger.info("Creating horse features")

        # 年齢（既に存在）

        # 性別のエンコーディング
        if 'sex' in df.columns:
            sex_map = {'牡': 0, '牝': 1, 'セ': 2}
            df['sex_encoded'] = df['sex'].map(sex_map)

        # 馬体重の変化率
        if 'horse_weight' in df.columns and 'weight_change' in df.columns:
            df['weight_change_ratio'] = df['weight_change'] / df['horse_weight']

        # 斤量と馬体重の比率
        if 'weight' in df.columns and 'horse_weight' in df.columns:
            df['weight_burden_ratio'] = df['weight'] / df['horse_weight']

        return df

    def _create_jockey_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """騎手の特徴量を作成"""
        logger.info("Creating jockey features")

        # 騎手ごとの統計情報を計算
        if 'jockey_id' in df.columns and 'finish_position' in df.columns:
            # 騎手の勝率
            jockey_stats = df.groupby('jockey_id').agg({
                'finish_position': ['mean', 'std', lambda x: (x == 1).sum()],
                'race_id': 'count'
            }).reset_index()

            jockey_stats.columns = ['jockey_id', 'jockey_avg_position',
                                   'jockey_position_std', 'jockey_wins', 'jockey_races']

            # 勝率・連対率・複勝率
            jockey_stats['jockey_win_rate'] = jockey_stats['jockey_wins'] / jockey_stats['jockey_races']

            # マージ
            df = df.merge(jockey_stats[['jockey_id', 'jockey_avg_position',
                                       'jockey_win_rate']], on='jockey_id', how='left')

        return df

    def _create_trainer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """調教師の特徴量を作成"""
        logger.info("Creating trainer features")

        # 調教師ごとの統計情報を計算
        if 'trainer_id' in df.columns and 'finish_position' in df.columns:
            trainer_stats = df.groupby('trainer_id').agg({
                'finish_position': ['mean', 'std', lambda x: (x == 1).sum()],
                'race_id': 'count'
            }).reset_index()

            trainer_stats.columns = ['trainer_id', 'trainer_avg_position',
                                    'trainer_position_std', 'trainer_wins', 'trainer_races']

            # 勝率
            trainer_stats['trainer_win_rate'] = trainer_stats['trainer_wins'] / trainer_stats['trainer_races']

            # マージ
            df = df.merge(trainer_stats[['trainer_id', 'trainer_avg_position',
                                        'trainer_win_rate']], on='trainer_id', how='left')

        return df

    def _create_race_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """レースの特徴量を作成"""
        logger.info("Creating race features")

        # 距離のカテゴリ化
        if 'distance' in df.columns:
            df['distance_category'] = pd.cut(
                df['distance'],
                bins=[0, 1400, 1800, 2200, 5000],
                labels=['短距離', 'マイル', '中距離', '長距離']
            )

        # 馬場タイプのエンコーディング
        if 'track_type' in df.columns:
            track_type_map = {'芝': 0, 'ダート': 1}
            df['track_type_encoded'] = df['track_type'].map(track_type_map)

        # 馬場状態のエンコーディング
        if 'track_condition' in df.columns:
            condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
            df['track_condition_encoded'] = df['track_condition'].map(condition_map)

        # 天候のエンコーディング
        if 'weather' in df.columns:
            weather_map = {'晴': 0, '曇': 1, '雨': 2, '雪': 3}
            df['weather_encoded'] = df['weather'].map(weather_map)

        # 出走頭数
        if 'race_id' in df.columns:
            field_size = df.groupby('race_id').size().reset_index(name='field_size')
            df = df.merge(field_size, on='race_id', how='left')

        # 人気と実力の関係
        if 'popularity' in df.columns and 'odds' in df.columns:
            df['popularity_odds_ratio'] = df['popularity'] / (df['odds'] + 1)

        return df

    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """過去成績の特徴量を作成"""
        logger.info("Creating historical features")

        # レース日付でソート
        df = df.sort_values(['horse_id', 'race_date'])

        # 各馬の過去N戦の成績
        window_sizes = [3, 5, 10]

        for window in window_sizes:
            if 'finish_position' in df.columns:
                # 過去N戦の平均着順
                df[f'past_{window}_avg_position'] = df.groupby('horse_id')['finish_position'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )

                # 過去N戦の最高着順
                df[f'past_{window}_best_position'] = df.groupby('horse_id')['finish_position'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min().shift(1)
                )

                # 過去N戦の勝利数
                df[f'past_{window}_wins'] = df.groupby('horse_id')['finish_position'].transform(
                    lambda x: (x == 1).rolling(window=window, min_periods=1).sum().shift(1)
                )

        # 同距離・同馬場での過去成績
        if 'distance' in df.columns and 'track_type' in df.columns:
            # 同距離での平均着順
            df['same_distance_avg'] = df.groupby(['horse_id', 'distance'])['finish_position'].transform(
                lambda x: x.expanding().mean().shift(1)
            )

            # 同馬場での平均着順
            df['same_track_avg'] = df.groupby(['horse_id', 'track_type'])['finish_position'].transform(
                lambda x: x.expanding().mean().shift(1)
            )

        # レース間隔
        if 'race_date' in df.columns:
            df['days_since_last_race'] = df.groupby('horse_id')['race_date'].diff().dt.days

        # 連続出走回数
        df['consecutive_races'] = df.groupby('horse_id').cumcount()

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """相互作用の特徴量を作成"""
        logger.info("Creating interaction features")

        # 騎手×調教師のコンビネーション
        if 'jockey_id' in df.columns and 'trainer_id' in df.columns:
            df['jockey_trainer_combo'] = df['jockey_id'].astype(str) + '_' + df['trainer_id'].astype(str)

            # このコンビネーションの過去成績
            combo_stats = df.groupby('jockey_trainer_combo').agg({
                'finish_position': 'mean',
                'race_id': 'count'
            }).reset_index()
            combo_stats.columns = ['jockey_trainer_combo', 'combo_avg_position', 'combo_races']

            df = df.merge(combo_stats, on='jockey_trainer_combo', how='left')

        # 距離×馬場状態
        if 'distance' in df.columns and 'track_condition' in df.columns:
            df['distance_condition_interaction'] = df['distance'] * df['track_condition_encoded']

        # 人気×斤量
        if 'popularity' in df.columns and 'weight' in df.columns:
            df['popularity_weight_interaction'] = df['popularity'] * df['weight']

        return df

    def encode_categorical_features(self, train_df: pd.DataFrame,
                                    val_df: pd.DataFrame = None,
                                    test_df: pd.DataFrame = None) -> Tuple:
        """
        カテゴリカル特徴量のエンコーディング

        Args:
            train_df: 訓練データ
            val_df: 検証データ
            test_df: テストデータ

        Returns:
            エンコーディング済みのデータフレームのタプル
        """
        logger.info("Encoding categorical features")

        categorical_cols = ['course_name', 'distance_category']
        existing_cats = [col for col in categorical_cols if col in train_df.columns]

        if not existing_cats:
            return (train_df, val_df, test_df) if test_df is not None else (train_df, val_df)

        # Target Encodingを使用（ターゲット変数を考慮したエンコーディング）
        self.target_encoder = ce.TargetEncoder(cols=existing_cats)

        # 訓練データでフィット
        train_encoded = train_df.copy()
        train_encoded[existing_cats] = self.target_encoder.fit_transform(
            train_df[existing_cats],
            train_df['finish_position']
        )

        # 検証・テストデータに適用
        results = [train_encoded]

        if val_df is not None:
            val_encoded = val_df.copy()
            val_encoded[existing_cats] = self.target_encoder.transform(val_df[existing_cats])
            results.append(val_encoded)

        if test_df is not None:
            test_encoded = test_df.copy()
            test_encoded[existing_cats] = self.target_encoder.transform(test_df[existing_cats])
            results.append(test_encoded)

        return tuple(results)

    def scale_features(self, train_df: pd.DataFrame,
                      val_df: pd.DataFrame = None,
                      test_df: pd.DataFrame = None,
                      exclude_cols: List[str] = None) -> Tuple:
        """
        数値特徴量のスケーリング

        Args:
            train_df: 訓練データ
            val_df: 検証データ
            test_df: テストデータ
            exclude_cols: スケーリングから除外するカラム

        Returns:
            スケーリング済みのデータフレームのタプル
        """
        logger.info("Scaling numerical features")

        if exclude_cols is None:
            exclude_cols = ['race_id', 'horse_id', 'jockey_id', 'trainer_id',
                           'finish_position', 'race_date']

        # スケーリング対象のカラムを選択
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        # 訓練データでフィット
        train_scaled = train_df.copy()
        train_scaled[cols_to_scale] = self.scaler.fit_transform(train_df[cols_to_scale])

        results = [train_scaled]

        # 検証・テストデータに適用
        if val_df is not None:
            val_scaled = val_df.copy()
            val_scaled[cols_to_scale] = self.scaler.transform(val_df[cols_to_scale])
            results.append(val_scaled)

        if test_df is not None:
            test_scaled = test_df.copy()
            test_scaled[cols_to_scale] = self.scaler.transform(test_df[cols_to_scale])
            results.append(test_scaled)

        return tuple(results)

    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        モデルに使用する特徴量を選択

        Args:
            df: DataFrame

        Returns:
            選択された特徴量のリスト
        """
        # IDや日付などのメタデータを除外
        exclude = ['race_id', 'horse_id', 'jockey_id', 'trainer_id',
                  'race_date', 'horse_name', 'jockey_name', 'trainer_name',
                  'finish_position', 'race_name', 'sex_age', 'time',
                  'margin', 'jockey_trainer_combo']

        feature_cols = [col for col in df.columns if col not in exclude]

        # 数値型のカラムのみを選択
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Selected {len(numeric_features)} features for modeling")

        return numeric_features


if __name__ == "__main__":
    # 使用例
    engineer = FeatureEngineer()

    # データ読み込み
    train = pd.read_csv("../../data/processed/train.csv")
    val = pd.read_csv("../../data/processed/val.csv")
    test = pd.read_csv("../../data/processed/test.csv")

    # 特徴量作成
    train_feat = engineer.create_features(train)
    val_feat = engineer.create_features(val)
    test_feat = engineer.create_features(test)

    # エンコーディング
    train_enc, val_enc, test_enc = engineer.encode_categorical_features(
        train_feat, val_feat, test_feat
    )

    # スケーリング
    train_scaled, val_scaled, test_scaled = engineer.scale_features(
        train_enc, val_enc, test_enc
    )

    # 保存
    train_scaled.to_csv("../../data/processed/train_features.csv", index=False)
    val_scaled.to_csv("../../data/processed/val_features.csv", index=False)
    test_scaled.to_csv("../../data/processed/test_features.csv", index=False)

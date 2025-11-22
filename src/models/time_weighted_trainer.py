"""
時間加重学習
新しいデータほど重視して学習するモジュール
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeWeightedTrainer:
    """
    時間加重学習トレーナー

    戦略:
    - 過去10年のデータを使用
    - 最新4年間を特に重視（重み2-3倍）
    - 古いデータも保持（概念ドリフトの理解のため）
    """

    def __init__(self, current_year: int = None):
        """
        Args:
            current_year: 現在の年（Noneの場合は自動取得）
        """
        self.current_year = current_year or datetime.now().year
        self.lookback_years = 10  # 過去10年
        self.focus_years = 4      # 最新4年を重視

        logger.info(f"Current year: {self.current_year}")
        logger.info(f"Using data from {self.current_year - self.lookback_years} to {self.current_year}")
        logger.info(f"Focusing on {self.current_year - self.focus_years} to {self.current_year}")

    def calculate_time_weights(self, dates: pd.Series) -> np.ndarray:
        """
        各レースの時間加重を計算

        重み付け戦略:
        - 最新4年: 指数関数的に増加（1.0 → 3.0）
        - 過去6年: 線形に減少（0.3 → 1.0）

        Args:
            dates: レース日付のSeries

        Returns:
            重みの配列
        """
        # 日付をdatetimeに変換
        dates = pd.to_datetime(dates)

        # 年を抽出
        years = dates.dt.year

        # 重みを計算
        weights = np.zeros(len(years), dtype=float)

        focus_start_year = self.current_year - self.focus_years + 1
        lookback_start_year = self.current_year - self.lookback_years + 1

        for i, year in enumerate(years):
            if year >= focus_start_year:
                # 最新4年: 指数関数的な重み
                years_from_focus = year - focus_start_year
                # 2021: 1.0, 2022: 1.5, 2023: 2.2, 2024: 3.0
                weight = 1.0 * (1.5 ** years_from_focus)
            elif year >= lookback_start_year:
                # 過去6年: 線形な重み
                years_from_start = year - lookback_start_year
                # 2015: 0.3, 2016: 0.4, ..., 2020: 0.9
                weight = 0.3 + (years_from_start * 0.1)
            else:
                # それ以前: 最小重み
                weight = 0.1

            weights[i] = weight

        # 正規化（合計が元のサンプル数になるように）
        weights = weights * (len(weights) / weights.sum())

        logger.info(f"Weight statistics:")
        logger.info(f"  Mean: {weights.mean():.3f}")
        logger.info(f"  Median: {np.median(weights):.3f}")
        logger.info(f"  Min: {weights.min():.3f}")
        logger.info(f"  Max: {weights.max():.3f}")

        # 年ごとの平均重み
        for year in range(lookback_start_year, self.current_year + 1):
            year_mask = years == year
            if year_mask.sum() > 0:
                avg_weight = weights[year_mask].mean()
                logger.info(f"  {year}: {avg_weight:.3f} (n={year_mask.sum()})")

        return weights

    def filter_data_by_period(self, df: pd.DataFrame,
                              date_column: str = 'race_date') -> pd.DataFrame:
        """
        指定期間のデータのみを抽出

        Args:
            df: DataFrame
            date_column: 日付カラム名

        Returns:
            フィルタリングされたDataFrame
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        start_year = self.current_year - self.lookback_years + 1
        start_date = pd.Timestamp(f"{start_year}-01-01")

        filtered = df[df[date_column] >= start_date]

        logger.info(f"Filtered data: {len(df)} -> {len(filtered)} samples")
        logger.info(f"Period: {start_date.date()} to {df[date_column].max().date()}")

        return filtered

    def get_sample_weights_for_lightgbm(self, df: pd.DataFrame,
                                        date_column: str = 'race_date') -> np.ndarray:
        """
        LightGBM用のサンプル重みを取得

        Args:
            df: DataFrame
            date_column: 日付カラム名

        Returns:
            サンプル重みの配列
        """
        weights = self.calculate_time_weights(df[date_column])
        return weights

    def apply_decay_to_features(self, df: pd.DataFrame,
                                date_column: str = 'race_date',
                                feature_columns: List[str] = None,
                                decay_rate: float = 0.1) -> pd.DataFrame:
        """
        過去成績に時間減衰を適用

        例: 過去5戦の平均着順
        - 最近のレース: フル重み
        - 古いレース: 減衰重み

        Args:
            df: DataFrame
            date_column: 日付カラム名
            feature_columns: 減衰を適用する特徴量
            decay_rate: 減衰率（0-1、大きいほど急速に減衰）

        Returns:
            減衰適用後のDataFrame
        """
        if feature_columns is None:
            # 過去成績関連の特徴量に適用
            feature_columns = [col for col in df.columns
                             if 'past' in col.lower() or 'avg' in col.lower()]

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # 最新の日付を基準に
        max_date = df[date_column].max()

        # 日数の差を計算
        days_diff = (max_date - df[date_column]).dt.days

        # 減衰係数を計算（指数減衰）
        # 365日前 = 0.9倍、730日前 = 0.8倍...
        decay_factor = np.exp(-decay_rate * days_diff / 365)

        # 特徴量に減衰を適用
        for col in feature_columns:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[f'{col}_decayed'] = df[col] * decay_factor
                logger.debug(f"Applied decay to {col}")

        logger.info(f"Applied time decay to {len(feature_columns)} features")

        return df


class AdaptiveSampler:
    """
    適応的サンプリング
    最近のデータを多くサンプリング
    """

    @staticmethod
    def time_stratified_sample(df: pd.DataFrame,
                              date_column: str = 'race_date',
                              n_samples: int = None,
                              recent_ratio: float = 0.6) -> pd.DataFrame:
        """
        時間層化サンプリング

        Args:
            df: DataFrame
            date_column: 日付カラム名
            n_samples: サンプル数（Noneの場合は全データ）
            recent_ratio: 最近のデータの割合（0-1）

        Returns:
            サンプリングされたDataFrame
        """
        if n_samples is None or n_samples >= len(df):
            return df

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # データを新旧で分割
        median_date = df[date_column].median()
        recent_df = df[df[date_column] >= median_date]
        old_df = df[df[date_column] < median_date]

        # サンプル数を配分
        n_recent = int(n_samples * recent_ratio)
        n_old = n_samples - n_recent

        # サンプリング
        recent_sample = recent_df.sample(
            n=min(n_recent, len(recent_df)),
            random_state=42
        )
        old_sample = old_df.sample(
            n=min(n_old, len(old_df)),
            random_state=42
        )

        # 結合
        sampled = pd.concat([recent_sample, old_sample])

        logger.info(f"Sampled: {len(df)} -> {len(sampled)}")
        logger.info(f"  Recent: {len(recent_sample)}")
        logger.info(f"  Old: {len(old_sample)}")

        return sampled


# 使用例
if __name__ == "__main__":
    # ダミーデータ生成
    np.random.seed(42)

    # 2015年から2024年のデータ
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')
    n_samples = len(dates)

    df = pd.DataFrame({
        'race_date': np.random.choice(dates, n_samples),
        'horse_id': np.arange(n_samples),
        'finish_position': np.random.randint(1, 11, n_samples),
        'past_5_avg': np.random.randn(n_samples) + 5,
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })

    # 時間加重トレーナー
    trainer = TimeWeightedTrainer(current_year=2024)

    # 1. データのフィルタリング（過去10年）
    filtered_df = trainer.filter_data_by_period(df)

    # 2. 時間重みの計算
    weights = trainer.get_sample_weights_for_lightgbm(filtered_df)

    print("\n=== Sample Weights ===")
    print(f"Shape: {weights.shape}")
    print(f"Range: [{weights.min():.3f}, {weights.max():.3f}]")

    # 3. 特徴量への減衰適用
    decayed_df = trainer.apply_decay_to_features(
        filtered_df,
        feature_columns=['past_5_avg']
    )

    print("\n=== Decayed Features ===")
    print(decayed_df[['race_date', 'past_5_avg', 'past_5_avg_decayed']].head())

    # 4. 適応的サンプリング
    sampler = AdaptiveSampler()
    sampled_df = sampler.time_stratified_sample(
        filtered_df,
        n_samples=10000,
        recent_ratio=0.6
    )

    print("\n=== Year Distribution (Original) ===")
    print(filtered_df['race_date'].dt.year.value_counts().sort_index())

    print("\n=== Year Distribution (Sampled) ===")
    print(sampled_df['race_date'].dt.year.value_counts().sort_index())

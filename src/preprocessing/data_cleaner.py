"""
データクリーニングと前処理
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """レースデータのクリーニングと前処理"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.missing_threshold = self.config['preprocessing']['missing_value_threshold']
        self.random_seed = self.config['preprocessing']['random_seed']

    def clean_race_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レースデータのクリーニング

        Args:
            df: 生データのDataFrame

        Returns:
            クリーニング済みのDataFrame
        """
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")

        df_clean = df.copy()

        # 1. 重複行の削除
        df_clean = self._remove_duplicates(df_clean)

        # 2. 欠損値の処理
        df_clean = self._handle_missing_values(df_clean)

        # 3. データ型の変換
        df_clean = self._convert_data_types(df_clean)

        # 4. 異常値の処理
        df_clean = self._handle_outliers(df_clean)

        # 5. 不正なデータの除去
        df_clean = self._remove_invalid_data(df_clean)

        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")

        return df_clean

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """重複行を削除"""
        initial_rows = len(df)
        df_dedup = df.drop_duplicates(subset=['race_id', 'horse_id'], keep='first')
        removed = initial_rows - len(df_dedup)

        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")

        return df_dedup

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値の処理"""
        # 欠損率が閾値を超えるカラムを削除
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()

        if cols_to_drop:
            logger.info(f"Dropping columns with high missing values: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # 残りの欠損値を適切に処理
        # 数値カラムは中央値で補完
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.debug(f"Filled missing values in {col} with median: {median_value}")

        # カテゴリカルカラムは最頻値で補完
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                logger.debug(f"Filled missing values in {col} with mode: {mode_value}")

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ型の変換"""
        # 着順を数値に変換（除外・中止などは欠損値として扱う）
        if 'finish_position' in df.columns:
            df['finish_position'] = pd.to_numeric(
                df['finish_position'],
                errors='coerce'
            )

        # 人気を数値に変換
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(
                df['popularity'],
                errors='coerce'
            )

        # オッズを数値に変換
        if 'odds' in df.columns:
            df['odds'] = pd.to_numeric(
                df['odds'],
                errors='coerce'
            )

        # 斤量を数値に変換
        if 'weight' in df.columns:
            df['weight'] = pd.to_numeric(
                df['weight'],
                errors='coerce'
            )

        # 性齢を分割
        if 'sex_age' in df.columns:
            df['sex'] = df['sex_age'].str[0]  # 最初の文字（牡/牝/セ）
            df['age'] = pd.to_numeric(
                df['sex_age'].str[1:],
                errors='coerce'
            )

        # タイムを秒数に変換
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(self._convert_time_to_seconds)

        # レース日付を日付型に変換
        if 'race_date' in df.columns:
            df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')

        return df

    @staticmethod
    def _convert_time_to_seconds(time_str: str) -> Optional[float]:
        """
        タイム文字列を秒数に変換

        Args:
            time_str: タイム文字列 (例: "1:23.4")

        Returns:
            秒数（float）
        """
        if pd.isna(time_str) or time_str == '':
            return None

        try:
            # "1:23.4" -> 83.4秒
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except (ValueError, AttributeError):
            return None

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """異常値の処理"""
        outlier_method = self.config['preprocessing']['outlier_method']

        numeric_cols = ['odds', 'weight', 'horse_weight', 'time_seconds']
        existing_cols = [col for col in numeric_cols if col in df.columns]

        for col in existing_cols:
            if outlier_method == 'iqr':
                df = self._remove_outliers_iqr(df, col)
            elif outlier_method == 'zscore':
                df = self._remove_outliers_zscore(df, col)

        return df

    @staticmethod
    def _remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
        """
        IQR法で異常値を除去

        Args:
            df: DataFrame
            column: カラム名
            factor: IQR倍数

        Returns:
            異常値を除去したDataFrame
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        initial_count = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = initial_count - len(df)

        if removed > 0:
            logger.debug(f"Removed {removed} outliers from {column} using IQR method")

        return df

    @staticmethod
    def _remove_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Z-score法で異常値を除去

        Args:
            df: DataFrame
            column: カラム名
            threshold: Z-scoreの閾値

        Returns:
            異常値を除去したDataFrame
        """
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        initial_count = len(df)
        df = df[z_scores < threshold]
        removed = initial_count - len(df)

        if removed > 0:
            logger.debug(f"Removed {removed} outliers from {column} using Z-score method")

        return df

    def _remove_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """不正なデータの除去"""
        initial_count = len(df)

        # 着順が欠損しているレコードを除去
        if 'finish_position' in df.columns:
            df = df.dropna(subset=['finish_position'])

        # 年齢が異常な値のレコードを除去
        if 'age' in df.columns:
            df = df[(df['age'] >= 2) & (df['age'] <= 12)]

        # 距離が異常な値のレコードを除去
        if 'distance' in df.columns:
            df = df[(df['distance'] >= 1000) & (df['distance'] <= 4000)]

        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} invalid records")

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        データを訓練・検証・テストセットに分割

        Args:
            df: クリーニング済みのDataFrame

        Returns:
            (train_df, val_df, test_df)のタプル
        """
        # レース日付でソート
        df = df.sort_values('race_date')

        # 時系列で分割（古い順に訓練、検証、テストデータ）
        test_split = self.config['preprocessing']['train_test_split']
        val_split = self.config['preprocessing']['validation_split']

        n = len(df)
        test_idx = int(n * (1 - test_split))
        val_idx = int(test_idx * (1 - val_split))

        train_df = df.iloc[:val_idx]
        val_df = df.iloc[val_idx:test_idx]
        test_df = df.iloc[test_idx:]

        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def save_cleaned_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                         test_df: pd.DataFrame, output_dir: str):
        """
        クリーニング済みデータを保存

        Args:
            train_df: 訓練データ
            val_df: 検証データ
            test_df: テストデータ
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_path / 'train.csv', index=False, encoding='utf-8-sig')
        val_df.to_csv(output_path / 'val.csv', index=False, encoding='utf-8-sig')
        test_df.to_csv(output_path / 'test.csv', index=False, encoding='utf-8-sig')

        logger.info(f"Saved cleaned data to {output_dir}")


if __name__ == "__main__":
    # 使用例
    cleaner = DataCleaner()

    # 生データを読み込み
    raw_data = pd.read_csv("../../data/raw/all_races.csv")

    # クリーニング
    clean_data = cleaner.clean_race_data(raw_data)

    # 分割
    train, val, test = cleaner.split_data(clean_data)

    # 保存
    cleaner.save_cleaned_data(train, val, test, "../../data/processed")

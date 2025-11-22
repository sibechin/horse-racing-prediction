"""
JRA-VAN Data Lab からのデータ読み込み
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JRAVANLoader:
    """
    JRA-VAN Data Labから提供されるCSVデータを読み込むクラス

    JRA-VAN Data Lab: https://jra-van.jp/dlb/
    - 月額2,090円（無料トライアル期間あり）
    - WindowsアプリでCSVエクスポート可能
    """

    def __init__(self, data_dir: str = "data/raw/jravan"):
        """
        Args:
            data_dir: JRA-VANからダウンロードしたCSVファイルのディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_race_results(self, filepath: str) -> pd.DataFrame:
        """
        レース結果データを読み込み

        Args:
            filepath: CSVファイルのパス

        Returns:
            レース結果のDataFrame
        """
        logger.info(f"Loading race results from {filepath}")

        # JRA-VANのCSVは通常Shift-JISエンコーディング
        df = pd.read_csv(filepath, encoding='shift-jis')

        logger.info(f"Loaded {len(df)} records")

        return df

    def load_horse_master(self, filepath: str) -> pd.DataFrame:
        """
        馬マスタデータを読み込み

        Args:
            filepath: CSVファイルのパス

        Returns:
            馬マスタのDataFrame
        """
        logger.info(f"Loading horse master from {filepath}")

        df = pd.read_csv(filepath, encoding='shift-jis')

        logger.info(f"Loaded {len(df)} horses")

        return df

    def load_jockey_master(self, filepath: str) -> pd.DataFrame:
        """
        騎手マスタデータを読み込み

        Args:
            filepath: CSVファイルのパス

        Returns:
            騎手マスタのDataFrame
        """
        logger.info(f"Loading jockey master from {filepath}")

        df = pd.read_csv(filepath, encoding='shift-jis')

        logger.info(f"Loaded {len(df)} jockeys")

        return df

    def load_trainer_master(self, filepath: str) -> pd.DataFrame:
        """
        調教師マスタデータを読み込み

        Args:
            filepath: CSVファイルのパス

        Returns:
            調教師マスタのDataFrame
        """
        logger.info(f"Loading trainer master from {filepath}")

        df = pd.read_csv(filepath, encoding='shift-jis')

        logger.info(f"Loaded {len(df)} trainers")

        return df

    def merge_all_data(self, race_results: pd.DataFrame,
                      horse_master: pd.DataFrame = None,
                      jockey_master: pd.DataFrame = None,
                      trainer_master: pd.DataFrame = None) -> pd.DataFrame:
        """
        すべてのデータをマージ

        Args:
            race_results: レース結果
            horse_master: 馬マスタ
            jockey_master: 騎手マスタ
            trainer_master: 調教師マスタ

        Returns:
            マージされたDataFrame
        """
        logger.info("Merging all data")

        merged = race_results.copy()

        # 馬マスタとマージ
        if horse_master is not None:
            merged = merged.merge(
                horse_master,
                on='horse_id',  # カラム名は実際のデータに合わせて調整
                how='left',
                suffixes=('', '_horse')
            )

        # 騎手マスタとマージ
        if jockey_master is not None:
            merged = merged.merge(
                jockey_master,
                on='jockey_id',
                how='left',
                suffixes=('', '_jockey')
            )

        # 調教師マスタとマージ
        if trainer_master is not None:
            merged = merged.merge(
                trainer_master,
                on='trainer_id',
                how='left',
                suffixes=('', '_trainer')
            )

        logger.info(f"Merged data shape: {merged.shape}")

        return merged


class JRAPublicDataLoader:
    """
    JRA公式サイトの公開データを読み込むクラス

    注意: JRA公式サイトのデータは主にPDF形式のため、
    実用的にはJRA-VANの使用を推奨
    """

    def __init__(self):
        self.base_url = "https://www.jra.go.jp/datafile/seiseki/"

    def download_pdf(self, year: int, output_dir: str = "data/raw/jra_pdf"):
        """
        年度別成績PDFをダウンロード（参考実装）

        Args:
            year: 年度
            output_dir: 出力ディレクトリ
        """
        import requests

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        url = f"{self.base_url}report/{year}.pdf"

        logger.info(f"Downloading {year} report from {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            filepath = output_path / f"{year}_report.pdf"
            with open(filepath, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded to {filepath}")

        except requests.RequestException as e:
            logger.error(f"Failed to download: {e}")


class DataConverter:
    """
    様々な形式のデータを統一フォーマットに変換
    """

    @staticmethod
    def convert_jravan_to_standard(df: pd.DataFrame) -> pd.DataFrame:
        """
        JRA-VAN形式のデータを標準形式に変換

        Args:
            df: JRA-VAN形式のDataFrame

        Returns:
            標準形式のDataFrame

        注意: カラム名はJRA-VANの実際の出力に合わせて調整が必要
        """
        # カラム名のマッピング（実際のJRA-VANデータに合わせて調整）
        column_mapping = {
            # レース情報
            'レースID': 'race_id',
            '開催年月日': 'race_date',
            '競馬場コード': 'venue_code',
            '距離': 'distance',
            '芝ダ障害コード': 'track_type_code',
            '馬場状態コード': 'track_condition_code',
            '天候コード': 'weather_code',

            # 馬情報
            '馬番': 'horse_number',
            '血統登録番号': 'horse_id',
            '馬名': 'horse_name',
            '性別コード': 'sex_code',
            '馬齢': 'age',
            '負担重量': 'weight',
            '馬体重': 'horse_weight',
            '馬体重増減': 'weight_change',

            # 騎手情報
            '騎手コード': 'jockey_id',
            '騎手名': 'jockey_name',

            # 調教師情報
            '調教師コード': 'trainer_id',
            '調教師名': 'trainer_name',

            # 結果
            '着順': 'finish_position',
            'タイム': 'time',
            '着差': 'margin',
            '人気': 'popularity',
            '単勝オッズ': 'odds',
        }

        # カラム名を変換（存在するカラムのみ）
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_converted = df.rename(columns=rename_dict)

        # データ型の変換
        if 'race_date' in df_converted.columns:
            df_converted['race_date'] = pd.to_datetime(
                df_converted['race_date'],
                errors='coerce'
            )

        # 数値型に変換
        numeric_cols = [
            'distance', 'horse_number', 'age', 'weight',
            'horse_weight', 'weight_change', 'finish_position',
            'popularity', 'odds'
        ]
        for col in numeric_cols:
            if col in df_converted.columns:
                df_converted[col] = pd.to_numeric(
                    df_converted[col],
                    errors='coerce'
                )

        # コード値のマッピング
        if 'track_type_code' in df_converted.columns:
            track_type_map = {
                '1': '芝',
                '2': 'ダート',
                '3': '障害'
            }
            df_converted['track_type'] = df_converted['track_type_code'].astype(str).map(track_type_map)

        if 'track_condition_code' in df_converted.columns:
            condition_map = {
                '1': '良',
                '2': '稍重',
                '3': '重',
                '4': '不良'
            }
            df_converted['track_condition'] = df_converted['track_condition_code'].astype(str).map(condition_map)

        if 'weather_code' in df_converted.columns:
            weather_map = {
                '1': '晴',
                '2': '曇',
                '3': '雨',
                '4': '雪'
            }
            df_converted['weather'] = df_converted['weather_code'].astype(str).map(weather_map)

        if 'sex_code' in df_converted.columns:
            sex_map = {
                '1': '牡',
                '2': '牝',
                '3': 'セ'
            }
            df_converted['sex'] = df_converted['sex_code'].astype(str).map(sex_map)

        logger.info("Data conversion completed")

        return df_converted


# 使用例
if __name__ == "__main__":
    """
    JRA-VAN Data Lab からデータをエクスポートする手順:

    1. JRA-VAN Data Lab に登録（無料トライアル可）
       https://jra-van.jp/dlb/

    2. Windowsアプリをインストール

    3. データベースから必要なデータをCSVエクスポート:
       - レース結果データ
       - 馬マスタ
       - 騎手マスタ
       - 調教師マスタ

    4. エクスポートしたCSVを data/raw/jravan/ に配置

    5. このスクリプトで読み込み・統合
    """

    # JRA-VANデータの読み込み例
    loader = JRAVANLoader(data_dir="data/raw/jravan")

    # 各データファイルを読み込み（ファイル名は実際のものに置き換え）
    try:
        race_results = loader.load_race_results("data/raw/jravan/race_results.csv")
        horse_master = loader.load_horse_master("data/raw/jravan/horse_master.csv")
        jockey_master = loader.load_jockey_master("data/raw/jravan/jockey_master.csv")
        trainer_master = loader.load_trainer_master("data/raw/jravan/trainer_master.csv")

        # データを統合
        merged_data = loader.merge_all_data(
            race_results,
            horse_master,
            jockey_master,
            trainer_master
        )

        # 標準形式に変換
        converter = DataConverter()
        standard_data = converter.convert_jravan_to_standard(merged_data)

        # 保存
        output_file = "data/raw/all_races_jravan.csv"
        standard_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Saved to {output_file}")

    except FileNotFoundError as e:
        logger.warning(f"Data file not found: {e}")
        logger.info("\nJRA-VAN Data Labからデータをエクスポートしてください:")
        logger.info("1. https://jra-van.jp/dlb/ に登録")
        logger.info("2. Windowsアプリでデータをエクスポート")
        logger.info("3. CSVファイルを data/raw/jravan/ に配置")

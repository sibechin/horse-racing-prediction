"""
netkeiba.comからのデータ収集スクレイパー
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pathlib import Path
import yaml
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetkeibaScaper:
    """netkeiba.comからレースデータを収集するスクレイパー"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config['data_collection']['base_url']
        self.delay = self.config['data_collection']['delay_between_requests']
        self.max_retries = self.config['data_collection']['max_retries']
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_race_calendar(self, year: int) -> List[Dict]:
        """
        指定年のレースカレンダーを取得

        Args:
            year: 年

        Returns:
            レース情報のリスト
        """
        logger.info(f"Fetching race calendar for {year}")
        race_list = []

        # ここでは例として、レースIDのリストを生成
        # 実際にはnetkeiba.comのカレンダーページをスクレイピング
        # レースIDの形式: YYYYPPRRDD (年/競馬場/レース回/日)

        # TODO: 実際のカレンダーページからレースIDを取得する実装
        # 暫定的にダミーデータを返す
        logger.warning("Using dummy race IDs - implement actual calendar scraping")

        return race_list

    def scrape_race_result(self, race_id: str) -> Optional[pd.DataFrame]:
        """
        レース結果ページをスクレイピング

        Args:
            race_id: レースID

        Returns:
            レース結果のDataFrame
        """
        url = f"{self.base_url}/race/{race_id}/"

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # レース情報を取得
                race_info = self._parse_race_info(soup, race_id)

                # 結果テーブルを取得
                result_table = soup.find('table', class_='race_table_01')
                if not result_table:
                    logger.warning(f"No result table found for race {race_id}")
                    return None

                # テーブルをDataFrameに変換
                df = self._parse_result_table(result_table, race_info)

                logger.info(f"Successfully scraped race {race_id}")
                return df

            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed for race {race_id}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to scrape race {race_id} after {self.max_retries} attempts")
                    return None
                time.sleep(self.delay * (attempt + 1))

        return None

    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> Dict:
        """
        レース情報を解析

        Args:
            soup: BeautifulSoupオブジェクト
            race_id: レースID

        Returns:
            レース情報の辞書
        """
        race_info = {'race_id': race_id}

        # レース名
        title = soup.find('h1', class_='race_name')
        if title:
            race_info['race_name'] = title.text.strip()

        # レース条件（距離、馬場、天候など）
        condition = soup.find('div', class_='race_data')
        if condition:
            condition_text = condition.text.strip()

            # 距離を抽出
            distance_match = re.search(r'(\d+)m', condition_text)
            if distance_match:
                race_info['distance'] = int(distance_match.group(1))

            # 馬場状態
            if '芝' in condition_text:
                race_info['track_type'] = '芝'
            elif 'ダート' in condition_text:
                race_info['track_type'] = 'ダート'

            # 馬場コンディション
            for cond in ['良', '稍重', '重', '不良']:
                if cond in condition_text:
                    race_info['track_condition'] = cond
                    break

            # 天候
            for weather in ['晴', '曇', '雨', '雪']:
                if weather in condition_text:
                    race_info['weather'] = weather
                    break

        # 開催日
        date_elem = soup.find('p', class_='race_date')
        if date_elem:
            race_info['race_date'] = date_elem.text.strip()

        return race_info

    def _parse_result_table(self, table, race_info: Dict) -> pd.DataFrame:
        """
        結果テーブルを解析してDataFrameに変換

        Args:
            table: BeautifulSoupのtableオブジェクト
            race_info: レース情報

        Returns:
            結果のDataFrame
        """
        rows = []

        # ヘッダー行をスキップしてデータ行を処理
        for tr in table.find_all('tr')[1:]:
            cells = tr.find_all('td')
            if len(cells) < 10:
                continue

            row = race_info.copy()

            # 着順
            row['finish_position'] = cells[0].text.strip()

            # 枠番
            row['post_position'] = cells[1].text.strip()

            # 馬番
            row['horse_number'] = cells[2].text.strip()

            # 馬名とID
            horse_link = cells[3].find('a')
            if horse_link:
                row['horse_name'] = horse_link.text.strip()
                row['horse_id'] = horse_link.get('href', '').split('/')[-2]

            # 性齢
            row['sex_age'] = cells[4].text.strip()

            # 斤量
            row['weight'] = cells[5].text.strip()

            # 騎手
            jockey_link = cells[6].find('a')
            if jockey_link:
                row['jockey_name'] = jockey_link.text.strip()
                row['jockey_id'] = jockey_link.get('href', '').split('/')[-2]

            # タイム
            row['time'] = cells[7].text.strip()

            # 着差
            row['margin'] = cells[8].text.strip()

            # 人気
            row['popularity'] = cells[9].text.strip()

            # オッズ
            if len(cells) > 10:
                row['odds'] = cells[10].text.strip()

            # 馬体重
            if len(cells) > 11:
                weight_text = cells[11].text.strip()
                weight_match = re.match(r'(\d+)\(([+-]?\d+)\)', weight_text)
                if weight_match:
                    row['horse_weight'] = int(weight_match.group(1))
                    row['weight_change'] = int(weight_match.group(2))

            # 調教師
            if len(cells) > 12:
                trainer_link = cells[12].find('a')
                if trainer_link:
                    row['trainer_name'] = trainer_link.text.strip()
                    row['trainer_id'] = trainer_link.get('href', '').split('/')[-2]

            rows.append(row)

        return pd.DataFrame(rows)

    def scrape_horse_profile(self, horse_id: str) -> Optional[Dict]:
        """
        馬のプロフィールページをスクレイピング

        Args:
            horse_id: 馬ID

        Returns:
            馬のプロフィール情報
        """
        url = f"{self.base_url}/horse/{horse_id}/"

        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            profile = {'horse_id': horse_id}

            # TODO: 馬のプロフィール情報を解析
            # - 生年月日
            # - 父馬・母馬
            # - 生産者
            # - 馬主
            # - 過去成績

            return profile

        except requests.RequestException as e:
            logger.error(f"Failed to scrape horse profile {horse_id}: {e}")
            return None

    def scrape_jockey_profile(self, jockey_id: str) -> Optional[Dict]:
        """
        騎手のプロフィールページをスクレイピング

        Args:
            jockey_id: 騎手ID

        Returns:
            騎手のプロフィール情報
        """
        url = f"{self.base_url}/jockey/{jockey_id}/"

        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            profile = {'jockey_id': jockey_id}

            # TODO: 騎手のプロフィール情報を解析
            # - 所属
            # - 勝率
            # - 連対率
            # - 複勝率

            return profile

        except requests.RequestException as e:
            logger.error(f"Failed to scrape jockey profile {jockey_id}: {e}")
            return None

    def collect_historical_data(self, start_year: int, end_year: int, output_dir: str):
        """
        指定期間の過去データを収集

        Args:
            start_year: 開始年
            end_year: 終了年
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_races = []

        for year in range(start_year, end_year + 1):
            logger.info(f"Collecting data for year {year}")

            # レースカレンダーを取得
            race_calendar = self.get_race_calendar(year)

            year_races = []
            for race_info in race_calendar:
                race_id = race_info['race_id']
                df = self.scrape_race_result(race_id)

                if df is not None:
                    year_races.append(df)

            if year_races:
                year_df = pd.concat(year_races, ignore_index=True)

                # 年ごとにCSVファイルとして保存
                output_file = output_path / f"races_{year}.csv"
                year_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"Saved {len(year_df)} race results to {output_file}")

                all_races.append(year_df)

        # 全データを統合して保存
        if all_races:
            combined_df = pd.concat(all_races, ignore_index=True)
            combined_file = output_path / "all_races.csv"
            combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig')
            logger.info(f"Saved combined data to {combined_file}")


if __name__ == "__main__":
    # 使用例
    scraper = NetkeibaScaper()

    # 過去データの収集
    scraper.collect_historical_data(
        start_year=2020,
        end_year=2024,
        output_dir="../data/raw"
    )

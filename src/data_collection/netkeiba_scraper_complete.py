"""
netkeiba.com 完全版スクレイパー
実際に動作するバージョン
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetkeibaScraperComplete:
    """netkeiba.comからレースデータを収集する完全版スクレイパー"""

    def __init__(self, delay: float = 2.0, max_retries: int = 3):
        """
        Args:
            delay: リクエスト間隔（秒）
            max_retries: 最大リトライ回数
        """
        self.base_url = "https://db.netkeiba.com"
        self.delay = delay
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def get_race_list_by_date(self, date: str) -> List[str]:
        """
        指定日のレースID一覧を取得

        Args:
            date: 日付 (YYYYMMDD形式)

        Returns:
            レースIDのリスト
        """
        # カレンダーページから取得
        # 実装の簡略化のため、レースIDを直接指定する方法を推奨
        logger.warning("カレンダーからの自動取得は未実装です")
        return []

    def scrape_race_result(self, race_id: str) -> Optional[pd.DataFrame]:
        """
        レース結果をスクレイピング

        Args:
            race_id: レースID (例: 202406030811)

        Returns:
            レース結果のDataFrame
        """
        url = f"{self.base_url}/race/{race_id}/"

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Scraping race {race_id} (attempt {attempt + 1}/{self.max_retries})")

                time.sleep(self.delay)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # レース情報を取得
                race_info = self._parse_race_info(soup, race_id)

                # 結果テーブルを取得
                results = self._parse_race_results(soup, race_info)

                if results is None or len(results) == 0:
                    logger.warning(f"No results found for race {race_id}")
                    return None

                logger.info(f"Successfully scraped race {race_id}: {len(results)} horses")
                return results

            except requests.RequestException as e:
                logger.error(f"Request failed for race {race_id}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.delay * (attempt + 2))

            except Exception as e:
                logger.error(f"Unexpected error for race {race_id}: {e}")
                return None

        return None

    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> Dict:
        """レース情報を解析"""
        race_info = {'race_id': race_id}

        try:
            # レース名 (h1タグから)
            title_elem = soup.find('h1')
            if title_elem:
                race_info['race_name'] = title_elem.text.strip()

            # レース条件（複数の可能性があるクラス名を試す）
            # パターン1: div.racedata
            race_data = soup.find('div', class_='racedata')
            if not race_data:
                # パターン2: div.race_data
                race_data = soup.find('div', class_='race_data')
            if not race_data:
                # パターン3: テキストから抽出
                race_data = soup.find('div', class_='data_intro')

            if race_data:
                data_text = race_data.text

                # 距離
                distance_match = re.search(r'(\d{4})m', data_text)
                if distance_match:
                    race_info['distance'] = int(distance_match.group(1))

                # 馬場タイプ
                if '芝' in data_text:
                    race_info['track_type'] = '芝'
                elif 'ダート' in data_text or 'ダ' in data_text:
                    race_info['track_type'] = 'ダート'
                elif '障害' in data_text:
                    race_info['track_type'] = '障害'

                # 馬場状態
                for condition in ['良', '稍重', '重', '不良']:
                    if condition in data_text:
                        race_info['track_condition'] = condition
                        break

                # 天候
                for weather in ['晴', '曇', '雨', '雪', '小雨', '小雪']:
                    if weather in data_text:
                        race_info['weather'] = weather
                        break

                # 発走時刻
                time_match = re.search(r'(\d{1,2}):(\d{2})', data_text)
                if time_match:
                    race_info['post_time'] = f"{time_match.group(1)}:{time_match.group(2)}"

            # 開催日（URLから推測）
            # race_id: YYYYMMDDRRNN
            # YYYY: 年, MM: 月, DD: 日, RR: 競馬場, NN: レース番号
            if len(race_id) >= 8:
                year = race_id[:4]
                month = race_id[4:6]
                day = race_id[6:8]
                race_info['race_date'] = f"{year}-{month}-{day}"

            # 競馬場コード（URLから）
            if len(race_id) >= 10:
                venue_code = race_id[8:10]
                race_info['venue_code'] = venue_code

                # 競馬場名のマッピング
                venue_map = {
                    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
                    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
                    '09': '阪神', '10': '小倉'
                }
                race_info['venue_name'] = venue_map.get(venue_code, '不明')

        except Exception as e:
            logger.error(f"Error parsing race info: {e}")

        return race_info

    def _parse_race_results(self, soup: BeautifulSoup, race_info: Dict) -> Optional[pd.DataFrame]:
        """レース結果テーブルを解析"""
        try:
            # 結果テーブルを探す（複数のパターンを試す）
            result_table = None

            # パターン1: table.race_table_01
            result_table = soup.find('table', class_='race_table_01')

            if not result_table:
                # パターン2: table.nk_tb_common
                result_table = soup.find('table', class_='nk_tb_common')

            if not result_table:
                # パターン3: id指定
                result_table = soup.find('table', {'summary': 'レース結果'})

            if not result_table:
                logger.error("Result table not found")
                return None

            rows = []

            # tbody内のtrを取得
            tbody = result_table.find('tbody')
            if tbody:
                tr_list = tbody.find_all('tr')
            else:
                tr_list = result_table.find_all('tr')

            # ヘッダー行をスキップ
            for tr in tr_list:
                # ヘッダー行をスキップ
                if tr.find('th'):
                    continue

                cells = tr.find_all('td')
                if len(cells) < 10:
                    continue

                row = race_info.copy()

                try:
                    # 着順
                    row['finish_position'] = self._extract_text(cells[0])

                    # 枠番
                    row['bracket_number'] = self._extract_text(cells[1])

                    # 馬番
                    row['horse_number'] = self._extract_text(cells[2])

                    # 馬名とID
                    horse_link = cells[3].find('a')
                    if horse_link:
                        row['horse_name'] = horse_link.text.strip()
                        href = horse_link.get('href', '')
                        horse_id_match = re.search(r'/horse/(\w+)/', href)
                        if horse_id_match:
                            row['horse_id'] = horse_id_match.group(1)

                    # 性齢
                    row['sex_age'] = self._extract_text(cells[4])

                    # 斤量
                    row['weight'] = self._extract_text(cells[5])

                    # 騎手
                    jockey_link = cells[6].find('a')
                    if jockey_link:
                        row['jockey_name'] = jockey_link.text.strip()
                        href = jockey_link.get('href', '')
                        jockey_id_match = re.search(r'/jockey/(\w+)/', href)
                        if jockey_id_match:
                            row['jockey_id'] = jockey_id_match.group(1)

                    # タイム
                    row['time'] = self._extract_text(cells[7])

                    # 着差
                    row['margin'] = self._extract_text(cells[8])

                    # 通過順位（オプション）
                    if len(cells) > 9:
                        row['passing_order'] = self._extract_text(cells[9])

                    # 上がり3F（オプション）
                    if len(cells) > 10:
                        row['last_3f'] = self._extract_text(cells[10])

                    # 人気
                    if len(cells) > 11:
                        row['popularity'] = self._extract_text(cells[11])

                    # オッズ
                    if len(cells) > 12:
                        row['odds'] = self._extract_text(cells[12])

                    # 馬体重
                    if len(cells) > 13:
                        weight_text = self._extract_text(cells[13])
                        weight_match = re.match(r'(\d+)\(([+-]?\d+)\)', weight_text)
                        if weight_match:
                            row['horse_weight'] = int(weight_match.group(1))
                            row['weight_change'] = int(weight_match.group(2))

                    # 調教師（オプション）
                    if len(cells) > 14:
                        trainer_link = cells[14].find('a')
                        if trainer_link:
                            row['trainer_name'] = trainer_link.text.strip()
                            href = trainer_link.get('href', '')
                            trainer_id_match = re.search(r'/trainer/(\w+)/', href)
                            if trainer_id_match:
                                row['trainer_id'] = trainer_id_match.group(1)

                    rows.append(row)

                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue

            if not rows:
                return None

            return pd.DataFrame(rows)

        except Exception as e:
            logger.error(f"Error parsing results table: {e}")
            return None

    @staticmethod
    def _extract_text(element) -> str:
        """要素からテキストを抽出"""
        if element:
            return element.text.strip()
        return ''

    def scrape_multiple_races(self, race_ids: List[str], output_dir: str = "data/raw") -> pd.DataFrame:
        """
        複数のレースをスクレイピング

        Args:
            race_ids: レースIDのリスト
            output_dir: 出力ディレクトリ

        Returns:
            全レース結果のDataFrame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_results = []
        failed_races = []

        for i, race_id in enumerate(race_ids, 1):
            logger.info(f"Progress: {i}/{len(race_ids)} races")

            result = self.scrape_race_result(race_id)

            if result is not None:
                all_results.append(result)
            else:
                failed_races.append(race_id)

            # 進捗を保存（10レースごと）
            if i % 10 == 0 and all_results:
                temp_df = pd.concat(all_results, ignore_index=True)
                temp_file = output_path / f"temp_races_{i}.csv"
                temp_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
                logger.info(f"Saved temporary results: {len(temp_df)} records")

        # 最終結果
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)

            # 保存
            output_file = output_path / f"netkeiba_races_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            logger.info(f"✅ Scraping completed!")
            logger.info(f"  Total races: {len(race_ids)}")
            logger.info(f"  Successful: {len(all_results)}")
            logger.info(f"  Failed: {len(failed_races)}")
            logger.info(f"  Total records: {len(final_df)}")
            logger.info(f"  Saved to: {output_file}")

            if failed_races:
                logger.warning(f"Failed race IDs: {failed_races}")

            return final_df
        else:
            logger.error("No data collected")
            return pd.DataFrame()


def generate_race_ids(year: int, venues: List[str] = None) -> List[str]:
    """
    レースIDを生成（サンプル）

    Args:
        year: 年
        venues: 競馬場コードのリスト（例: ['05', '06']）

    Returns:
        レースIDのリスト
    """
    if venues is None:
        # デフォルト: 東京(05)、中山(06)
        venues = ['05', '06']

    race_ids = []

    # 各月の主要な開催日をサンプリング
    # 実際にはカレンダーから取得するのが理想
    sample_dates = [
        '0107', '0114', '0121', '0128',  # 1月
        '0204', '0211', '0218', '0225',  # 2月
        # ... 他の月
    ]

    for venue in venues:
        for date in sample_dates:
            # レース番号1-12
            for race_num in range(1, 13):
                race_id = f"{year}{date}{venue}{race_num:02d}"
                race_ids.append(race_id)

    return race_ids


# 使用例
if __name__ == "__main__":
    scraper = NetkeibaScraperComplete(delay=2.0)

    # 方法1: 特定のレースIDを指定
    logger.info("=== 方法1: 特定レースのスクレイピング ===")

    # 例: 2024年の皐月賞
    race_id = "202406030811"
    result = scraper.scrape_race_result(race_id)

    if result is not None:
        print("\n=== レース結果 ===")
        print(result[['horse_name', 'jockey_name', 'finish_position', 'time', 'popularity']].head(10))
        print(f"\nTotal horses: {len(result)}")

        # 保存
        result.to_csv("data/raw/test_race.csv", index=False, encoding='utf-8-sig')
        print("\nSaved to: data/raw/test_race.csv")

    # 方法2: 複数レースのスクレイピング
    logger.info("\n=== 方法2: 複数レースのスクレイピング ===")

    # サンプルレースID（実際のレースIDに置き換えてください）
    sample_race_ids = [
        "202405051211",  # 2024年東京 (例)
        "202405061207",  # 2024年中山 (例)
        # ... 他のレースID
    ]

    # 実行
    # results = scraper.scrape_multiple_races(sample_race_ids)

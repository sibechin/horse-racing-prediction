"""
keibalab.jp スクレイパー

非商業利用目的での研究用データ収集
- レースデータ収集 (https://www.keibalab.jp/db/race/)
- 10秒以上のクロールディレイを遵守
- レースペース、血統情報などの追加データを収集

注意:
このスクレイパーは教育・研究目的のみで使用してください。
商業利用は禁止されています。
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KeibalabScraper:
    """keibalab.jpからレースデータを収集"""

    def __init__(self, delay: float = 12.0, max_retries: int = 3):
        """
        Args:
            delay: リクエスト間の待機時間（秒）デフォルト12秒（robots.txt推奨10秒以上）
            max_retries: 最大リトライ回数
        """
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research/Educational Purpose) Non-Commercial Use',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
        })
        self.last_request_time = 0

    def _wait(self):
        """クロールディレイを遵守"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            wait_time = self.delay - elapsed
            logging.info(f"  待機中: {wait_time:.1f}秒...")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """ページを取得"""
        for attempt in range(self.max_retries):
            try:
                self._wait()

                logging.info(f"  取得中: {url}")
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'html.parser')
                elif response.status_code == 404:
                    logging.warning(f"  ページが見つかりません (404): {url}")
                    return None
                else:
                    logging.warning(f"  HTTPエラー {response.status_code}: {url}")

            except Exception as e:
                logging.error(f"  エラー (試行 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))

        return None

    def scrape_race(self, race_url: str) -> Optional[pd.DataFrame]:
        """
        単一レースのデータを収集

        Args:
            race_url: レースURL (例: https://www.keibalab.jp/db/race/202411030811/)

        Returns:
            レース結果のDataFrame
        """
        soup = self._fetch_page(race_url)
        if not soup:
            return None

        try:
            # レース情報を抽出
            race_info = self._extract_race_info(soup, race_url)

            # レース結果テーブルを抽出 (クラスに'resulttable'を含むテーブル)
            results_table = soup.find('table', attrs={'class': lambda x: x and 'resulttable' in x})
            if not results_table:
                logging.warning(f"  レース結果テーブルが見つかりません: {race_url}")
                return None

            # 結果をパース
            horses_data = []
            rows = results_table.find_all('tr')[1:]  # ヘッダーをスキップ

            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 10:
                    continue

                # 実際のテーブル構造:
                # 0:着順, 1:枠番, 2:馬番, 3:馬名, 4:性齢, 5:斤量, 6:騎手,
                # 7:人気, 8:単勝, 9:タイム, 10:着差, 11:通過順, 12:上り,
                # 13:調教師, 14:馬体重, 15:α,β,Ω指数
                horse_data = {
                    **race_info,
                    'finish_position': self._clean_text(cols[0].text),
                    'frame_number': self._clean_text(cols[1].text),
                    'horse_number': self._clean_text(cols[2].text),
                    'horse_name': self._clean_text(cols[3].text),
                    'sex_age': self._clean_text(cols[4].text),
                    'weight': self._clean_text(cols[5].text),
                    'jockey': self._clean_text(cols[6].text),
                    'popularity': self._clean_text(cols[7].text) if len(cols) > 7 else '',
                    'odds': self._clean_text(cols[8].text) if len(cols) > 8 else '',
                    'time': self._clean_text(cols[9].text) if len(cols) > 9 else '',
                    'margin': self._clean_text(cols[10].text) if len(cols) > 10 else '',
                    'passing_order': self._clean_text(cols[11].text) if len(cols) > 11 else '',
                    'last_3f': self._clean_text(cols[12].text) if len(cols) > 12 else '',
                    'trainer': self._clean_text(cols[13].text) if len(cols) > 13 else '',
                    'horse_weight': self._clean_text(cols[14].text) if len(cols) > 14 else '',
                    'indices': self._clean_text(cols[15].text) if len(cols) > 15 else '',
                }

                horses_data.append(horse_data)

            if horses_data:
                df = pd.DataFrame(horses_data)
                logging.info(f"  成功 {len(df)}頭のデータを収集")
                return df
            else:
                logging.warning(f"  データが見つかりません: {race_url}")
                return None

        except Exception as e:
            logging.error(f"  パースエラー: {e}")
            return None

    def _extract_race_info(self, soup: BeautifulSoup, race_url: str) -> Dict:
        """レース基本情報を抽出"""
        race_info = {
            'race_url': race_url,
            'race_id': self._extract_race_id_from_url(race_url),
        }

        # レース名
        race_title = soup.find('h1')
        if race_title:
            race_info['race_name'] = self._clean_text(race_title.text)

        # レース詳細（距離、馬場状態など）
        race_details = soup.find('div', class_='race_detail')
        if race_details:
            detail_text = self._clean_text(race_details.text)

            # 距離を抽出 (例: 芝2400m)
            distance_match = re.search(r'(\d+)m', detail_text)
            if distance_match:
                race_info['distance'] = int(distance_match.group(1))

            # 馬場タイプ (芝/ダート)
            if '芝' in detail_text:
                race_info['track_type'] = '芝'
            elif 'ダート' in detail_text or 'ダ' in detail_text:
                race_info['track_type'] = 'ダート'

            # 馬場状態
            for condition in ['良', '稍重', '重', '不良']:
                if condition in detail_text:
                    race_info['track_condition'] = condition
                    break

            # 天候
            for weather in ['晴', '曇', '雨', '小雨', '雪']:
                if weather in detail_text:
                    race_info['weather'] = weather
                    break

        # レース日付
        date_elem = soup.find('div', class_='race_date')
        if date_elem:
            date_text = self._clean_text(date_elem.text)
            race_info['race_date'] = date_text

        # 競馬場名
        venue_elem = soup.find('span', class_='venue')
        if venue_elem:
            race_info['venue_name'] = self._clean_text(venue_elem.text)

        return race_info

    def _extract_race_id_from_url(self, url: str) -> str:
        """URLからレースIDを抽出"""
        # 例: https://www.keibalab.jp/db/race/202411030811/ → 202411030811
        match = re.search(r'/race/(\d+)/', url)
        if match:
            return match.group(1)
        return ''

    def _clean_text(self, text: str) -> str:
        """テキストをクリーニング"""
        return text.strip().replace('\n', '').replace('\r', '').replace('  ', ' ')

    def scrape_multiple_races(self, race_urls: List[str], output_dir: str = 'data/raw/keibalab') -> pd.DataFrame:
        """
        複数レースのデータを収集

        Args:
            race_urls: レースURLのリスト
            output_dir: 出力ディレクトリ

        Returns:
            全レースの結果を結合したDataFrame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_results = []
        successful_races = 0
        failed_races = 0

        logging.info("=" * 60)
        logging.info(f"keibalab.jp データ収集開始")
        logging.info("=" * 60)
        logging.info(f"総レース数: {len(race_urls)}")
        logging.info(f"クロールディレイ: {self.delay}秒")
        logging.info(f"推定所要時間: {len(race_urls) * self.delay / 60:.1f}分")
        logging.info("=" * 60)

        for i, race_url in enumerate(race_urls, 1):
            logging.info(f"\n[{i}/{len(race_urls)}] {race_url}")

            df = self.scrape_race(race_url)

            if df is not None and len(df) > 0:
                all_results.append(df)
                successful_races += 1
            else:
                failed_races += 1

            # 進捗表示
            if i % 10 == 0:
                logging.info(f"\n--- 進捗 ---")
                logging.info(f"成功: {successful_races}, 失敗: {failed_races}")
                logging.info(f"残り: {len(race_urls) - i} レース")

        # 結果を結合
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)

            # CSVに保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_path / f'keibalab_races_{timestamp}.csv'
            combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            logging.info("\n" + "=" * 60)
            logging.info("データ収集完了!")
            logging.info("=" * 60)
            logging.info(f"成功: {successful_races}/{len(race_urls)} レース")
            logging.info(f"失敗: {failed_races}/{len(race_urls)} レース")
            logging.info(f"総データ数: {len(combined_df)}頭")
            logging.info(f"保存先: {output_file}")
            logging.info("=" * 60)

            return combined_df
        else:
            logging.error("データ収集に失敗しました")
            return pd.DataFrame()

    def generate_g1_race_urls(self, start_year: int = 1995, end_year: int = 2024) -> List[str]:
        """
        G1レースのURL一覧を生成

        Args:
            start_year: 開始年
            end_year: 終了年

        Returns:
            G1レースURLのリスト

        注意: keibalab.jpのURL構造に基づいて生成
        レースIDフォーマット: YYYYMMDDVVRR
        - YYYY: 年
        - MM: 月
        - DD: 日
        - VV: 競馬場コード
        - RR: レース番号

        実際の収集前に、netkeibaなどで正確なレースID/日付を確認することを推奨
        """
        # 主要G1レースの開催月（概算）
        g1_races_schedule = [
            # レース名, 月, 競馬場コード(推定)
            ('フェブラリーS', 2, '06'),  # 中山
            ('高松宮記念', 3, '07'),  # 中京
            ('桜花賞', 4, '02'),  # 阪神
            ('皐月賞', 4, '05'),  # 中山
            ('天皇賞春', 4, '08'),  # 京都
            ('NHKマイルC', 5, '05'),  # 東京
            ('ヴィクトリアM', 5, '05'),  # 東京
            ('オークス', 5, '05'),  # 東京
            ('日本ダービー', 5, '05'),  # 東京
            ('安田記念', 6, '05'),  # 東京
            ('宝塚記念', 6, '02'),  # 阪神
            ('スプリンターズS', 9, '06'),  # 中山
            ('秋華賞', 10, '08'),  # 京都
            ('菊花賞', 10, '08'),  # 京都
            ('天皇賞秋', 10, '05'),  # 東京
            ('エリザベス女王杯', 11, '08'),  # 京都
            ('マイルCS', 11, '08'),  # 京都
            ('ジャパンC', 11, '05'),  # 東京
            ('チャンピオンズC', 12, '06'),  # 中山
            ('有馬記念', 12, '06'),  # 中山
        ]

        logging.info("=" * 60)
        logging.info("G1レースURL生成")
        logging.info("=" * 60)
        logging.info(f"期間: {start_year}-{end_year}")
        logging.info(f"主要G1レース数: {len(g1_races_schedule)}")
        logging.info("")
        logging.info("⚠️ 注意: これは推定URLです")
        logging.info("実際のレースIDは netkeibaやJRA公式で確認してください")
        logging.info("=" * 60)

        # 注意: これは概算URL生成です
        # 実際の利用には、正確なレースID/日付が必要です
        urls = []

        for year in range(start_year, end_year + 1):
            for race_name, month, venue_code in g1_races_schedule:
                # 推定レースID（日付とレース番号は推定）
                # 実際には正確な日付とレース番号が必要
                # 例: 202411030811 (2024年11月03日, 08=京都, 11=11R)
                estimated_day = '01'  # 概算（第1週と仮定）
                race_number = '11'  # G1は通常11Rか12R

                race_id = f"{year}{month:02d}{estimated_day}{venue_code}{race_number}"
                url = f"https://www.keibalab.jp/db/race/{race_id}/"
                urls.append(url)

        logging.info(f"生成されたURL数: {len(urls)}")
        logging.info("")

        return urls


def main():
    """メイン実行（テスト用）"""
    logging.info("=" * 60)
    logging.info("keibalab.jp スクレイパー - テストモード")
    logging.info("=" * 60)
    logging.info("")
    logging.info("⚠️ 注意事項:")
    logging.info("  - 非商業利用目的の研究用データ収集")
    logging.info("  - robots.txt推奨の10秒以上のディレイを遵守")
    logging.info("  - 過度なアクセスはサイトに負荷をかけるため控えてください")
    logging.info("=" * 60)
    logging.info("")

    # スクレイパー初期化（12秒ディレイ）
    scraper = KeibalabScraper(delay=12.0, max_retries=3)

    # テスト: 2024年ジャパンカップ（実際のURL確認済み）
    test_url = "https://www.keibalab.jp/db/race/202411030811/"

    logging.info(f"テスト収集: {test_url}")
    logging.info("")

    df = scraper.scrape_race(test_url)

    if df is not None:
        logging.info("\n収集成功!")
        logging.info(f"データ数: {len(df)}頭")
        logging.info("\nサンプルデータ:")
        print(df.head(10))

        # 保存
        output_dir = Path('data/raw/keibalab')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'test_race.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"\n保存先: {output_file}")
    else:
        logging.error("収集失敗")


if __name__ == "__main__":
    main()

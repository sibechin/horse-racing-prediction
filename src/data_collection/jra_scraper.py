"""
JRA公式サイトスクレイパー
https://www.jra.go.jp/datafile/seiseki/ からG1/G2/G3レースデータを収集

重複チェック機能付き
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class JRAScraper:
    """JRA公式サイトスクレイパー"""

    def __init__(self, delay: float = 1.0):
        """
        初期化

        Args:
            delay: リクエスト間の待機時間（秒）
        """
        self.base_url = "https://www.jra.go.jp/datafile/seiseki"
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # G1レース一覧（2010-2024年の主要レース）
        self.g1_races = {
            'feb': 'フェブラリーステークス',
            'takarazuka': '宝塚記念',
            'tenno_sho_spring': '天皇賞（春）',
            'tenno_sho_autumn': '天皇賞（秋）',
            'arima_kinen': '有馬記念',
            'japan_cup': 'ジャパンカップ',
            'mile_cs': 'マイルチャンピオンシップ',
            'yasuda_kinen': '安田記念',
            'sprinters_stakes': 'スプリンターズステークス',
            'shuka_sho': '秋華賞',
            'kikuka_sho': '菊花賞',
            'satsuki_sho': '皐月賞',
            'oaks': '優駿牝馬（オークス）',
            'derby': '東京優駿（日本ダービー）',
            'hopeful_stakes': 'ホープフルステークス',
            'hanshin_juvenile_fillies': '阪神ジュベナイルフィリーズ',
            'asahi_hai_futurity': '朝日杯フューチュリティステークス',
        }

    def load_existing_data(self, csv_path: str) -> Set[tuple]:
        """
        既存データを読み込んで重複チェック用のセットを作成

        Args:
            csv_path: 既存データのCSVパス

        Returns:
            (race_date, venue_name, distance, track_type) のセット
        """
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            # race_date, venue_name, distance, track_typeの組み合わせで識別
            existing = set()
            for _, row in df.iterrows():
                key = (
                    str(row['race_date']),
                    str(row['venue_name']),
                    int(row['distance']),
                    str(row['track_type'])
                )
                existing.add(key)
            logging.info(f"既存データ読み込み: {len(existing)}件のユニークレース")
            return existing
        except FileNotFoundError:
            logging.info("既存データなし - 新規収集を開始")
            return set()
        except Exception as e:
            logging.error(f"既存データ読み込みエラー: {e}")
            return set()

    def scrape_race_result(self, race_code: str, year: int) -> Optional[pd.DataFrame]:
        """
        単一レースの結果をスクレイピング

        Args:
            race_code: レースコード（例: 'feb', 'derby'）
            year: 年度

        Returns:
            レース結果のDataFrame（失敗時はNone）
        """
        url = f"{self.base_url}/g1/{race_code}/result/{race_code}{year}.html"

        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            response.encoding = response.apparent_encoding

            if response.status_code != 200:
                logging.debug(f"レースなし: {race_code}{year} (status={response.status_code})")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # テーブルを探す
            table = soup.find('table')
            if not table:
                logging.debug(f"テーブルなし: {race_code}{year}")
                return None

            # レース情報を抽出（ページから）
            race_info = self._extract_race_info(soup, race_code, year)

            # 結果テーブルをパース
            rows = table.find_all('tr')[1:]  # ヘッダーをスキップ
            race_data = []

            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) < 11:
                    continue

                try:
                    horse_data = self._parse_horse_row(cols, race_info)
                    if horse_data:
                        race_data.append(horse_data)
                except Exception as e:
                    logging.debug(f"行パースエラー: {e}")
                    continue

            if race_data:
                df = pd.DataFrame(race_data)
                logging.info(f"収集成功: {race_code}{year} ({len(df)}頭)")
                return df
            else:
                logging.debug(f"データなし: {race_code}{year}")
                return None

        except requests.RequestException as e:
            logging.error(f"リクエストエラー: {race_code}{year} - {e}")
            return None
        except Exception as e:
            logging.error(f"予期しないエラー: {race_code}{year} - {e}")
            return None

    def _extract_race_info(self, soup: BeautifulSoup, race_code: str, year: int) -> Dict:
        """
        レース情報を抽出

        Args:
            soup: BeautifulSoupオブジェクト
            race_code: レースコード
            year: 年度

        Returns:
            レース情報の辞書
        """
        # デフォルト情報
        info = {
            'race_name': self.g1_races.get(race_code, race_code),
            'year': year,
            'grade': 'G1',
            'venue_name': '不明',
            'track_type': '不明',
            'track_condition': '不明',
            'weather': '不明',
            'distance': 0,
            'race_date': f'{year}-01-01'  # デフォルト
        }

        # ページから情報を抽出（JRAのHTMLは構造が統一されていないため、柔軟に対応）
        try:
            # レース情報のテキストを探す
            text = soup.get_text()

            # 距離を探す
            distance_match = re.search(r'(\d{4})m|(\d{4})メートル', text)
            if distance_match:
                info['distance'] = int(distance_match.group(1) or distance_match.group(2))

            # 競馬場を探す
            venues = ['東京', '中山', '京都', '阪神', '中京', '新潟', '小倉', '福島', '函館', '札幌']
            for venue in venues:
                if venue in text:
                    info['venue_name'] = venue
                    break

            # 馬場タイプを探す
            if '芝' in text:
                info['track_type'] = '芝'
            elif 'ダート' in text or 'ダ' in text:
                info['track_type'] = 'ダート'

            # 馬場状態を探す
            conditions = ['良', '稍重', '重', '不良']
            for cond in conditions:
                if cond in text:
                    info['track_condition'] = cond
                    break

            # 天候を探す
            weathers = ['晴', '曇', '雨', '雪']
            for weather in weathers:
                if weather in text:
                    info['weather'] = weather
                    break

            # 日付を探す（より正確に）
            date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
            if date_match:
                y, m, d = date_match.groups()
                info['race_date'] = f'{y}-{int(m):02d}-{int(d):02d}'
            else:
                # レースコードから推定
                race_month = self._estimate_race_month(race_code)
                info['race_date'] = f'{year}-{race_month:02d}-01'

        except Exception as e:
            logging.debug(f"レース情報抽出エラー: {e}")

        return info

    def _estimate_race_month(self, race_code: str) -> int:
        """レースコードから開催月を推定"""
        month_map = {
            'feb': 2,
            'satsuki_sho': 4,
            'tenno_sho_spring': 4,
            'derby': 5,
            'oaks': 5,
            'yasuda_kinen': 6,
            'takarazuka': 6,
            'sprinters_stakes': 10,
            'tenno_sho_autumn': 10,
            'shuka_sho': 10,
            'kikuka_sho': 10,
            'mile_cs': 11,
            'japan_cup': 11,
            'hanshin_juvenile_fillies': 12,
            'asahi_hai_futurity': 12,
            'hopeful_stakes': 12,
            'arima_kinen': 12,
        }
        return month_map.get(race_code, 6)

    def _parse_horse_row(self, cols: List, race_info: Dict) -> Optional[Dict]:
        """
        馬の行データをパース

        Args:
            cols: テーブルのカラムリスト
            race_info: レース情報

        Returns:
            馬データの辞書
        """
        try:
            # 着順
            finish_text = cols[0].text.strip()
            if not finish_text or not finish_text.isdigit():
                return None

            finish_position = int(finish_text)

            # 枠番・馬番
            frame_number = int(cols[1].text.strip()) if cols[1].text.strip() else 0
            horse_number = int(cols[2].text.strip()) if cols[2].text.strip() else 0

            # 馬名
            horse_name = cols[3].text.strip()

            # 年齢・性別
            age_sex = cols[4].text.strip()
            age_match = re.match(r'(\d+)', age_sex)
            age = int(age_match.group(1)) if age_match else 4
            sex = 'M' if '牡' in age_sex else ('F' if '牝' in age_sex else 'G')

            # 斤量
            weight_kg = float(cols[5].text.strip()) if cols[5].text.strip() else 55.0

            # 騎手名
            jockey_name = cols[6].text.strip()

            # タイム
            time_str = cols[7].text.strip()
            time_match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
            if time_match:
                minutes = int(time_match.group(1))
                seconds = int(time_match.group(2))
                milliseconds = int(time_match.group(3))
                time_seconds = minutes * 60 + seconds + milliseconds / 10
            else:
                time_seconds = 0

            # オッズ
            odds_str = cols[10].text.strip() if len(cols) > 10 else '0'
            try:
                odds = float(odds_str)
            except:
                odds = 0.0

            # 一意のrace_idを生成
            race_id = f"jra_{race_info['year']}_{race_info['race_name'].replace('（', '').replace('）', '')}_{race_info['venue_name']}"

            return {
                'race_id': race_id,
                'race_date': race_info['race_date'],
                'venue_name': race_info['venue_name'],
                'track_type': race_info['track_type'],
                'track_condition': race_info['track_condition'],
                'weather': race_info['weather'],
                'distance': race_info['distance'],
                'grade': race_info['grade'],
                'race_name': race_info['race_name'],
                'finish_position': finish_position,
                'frame_number': frame_number,
                'horse_number': horse_number,
                'horse_name': horse_name,
                'horse_id': f'jra_{horse_name}_{race_info["year"]}',
                'age': age,
                'sex': sex,
                'weight': weight_kg,
                'jockey_name': jockey_name,
                'time_seconds': time_seconds,
                'odds': odds
            }

        except Exception as e:
            logging.debug(f"行パースエラー: {e}")
            return None

    def scrape_multiple_years(
        self,
        race_codes: List[str],
        start_year: int = 2010,
        end_year: int = 2024,
        existing_data_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        複数年のレース結果を収集

        Args:
            race_codes: レースコードのリスト
            start_year: 開始年
            end_year: 終了年
            existing_data_path: 既存データのCSVパス（重複チェック用）

        Returns:
            全レース結果のDataFrame
        """
        # 重複チェック用の既存データ読み込み
        existing_races = set()
        if existing_data_path:
            existing_races = self.load_existing_data(existing_data_path)

        all_data = []
        total_races = len(race_codes) * (end_year - start_year + 1)
        processed = 0
        skipped = 0

        logging.info(f"収集開始: {len(race_codes)}レース × {end_year - start_year + 1}年 = {total_races}件")

        for race_code in race_codes:
            for year in range(start_year, end_year + 1):
                processed += 1

                # プログレス表示
                if processed % 10 == 0:
                    logging.info(f"進捗: {processed}/{total_races} ({processed/total_races*100:.1f}%)")

                # データ収集
                df = self.scrape_race_result(race_code, year)

                if df is not None and len(df) > 0:
                    # 重複チェック
                    if existing_races:
                        sample = df.iloc[0]
                        key = (
                            str(sample['race_date']),
                            str(sample['venue_name']),
                            int(sample['distance']),
                            str(sample['track_type'])
                        )
                        if key in existing_races:
                            logging.info(f"スキップ（重複）: {race_code}{year}")
                            skipped += 1
                            continue

                    all_data.append(df)

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            logging.info(f"\n収集完了!")
            logging.info(f"  総レース数: {result_df['race_id'].nunique()}")
            logging.info(f"  総データ数: {len(result_df)}頭")
            logging.info(f"  スキップ（重複）: {skipped}件")
            return result_df
        else:
            logging.warning("データが収集されませんでした")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """
        DataFrameをCSVに保存

        Args:
            df: 保存するDataFrame
            output_path: 出力ファイルパス
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"保存完了: {output_path}")


def main():
    """メイン実行関数"""
    print("="*60)
    print("JRA公式サイト G1レースデータ収集")
    print("="*60)
    print()

    # スクレイパー初期化
    scraper = JRAScraper(delay=1.5)

    # G1レース一覧
    g1_race_codes = list(scraper.g1_races.keys())

    print(f"収集対象: {len(g1_race_codes)}種類のG1レース")
    print(f"対象年度: 2010-2024年")
    print(f"推定レース数: {len(g1_race_codes) * 15}レース")
    print()

    # 既存データのパス
    existing_data = "data/processed/features_engineered.csv"

    # データ収集
    df = scraper.scrape_multiple_years(
        race_codes=g1_race_codes,
        start_year=2010,
        end_year=2024,
        existing_data_path=existing_data
    )

    if len(df) > 0:
        print()
        print("="*60)
        print("収集結果")
        print("="*60)
        print(f"レース数: {df['race_id'].nunique()}")
        print(f"総データ数: {len(df)}頭")
        print(f"期間: {df['race_date'].min()} - {df['race_date'].max()}")
        print()

        print("競馬場別:")
        print(df['venue_name'].value_counts())
        print()

        print("馬場タイプ別:")
        print(df['track_type'].value_counts())
        print()

        # 保存
        output_path = "data/jra/g1_races_2010_2024.csv"
        scraper.save_to_csv(df, output_path)

        print()
        print(f"データ保存完了: {output_path}")
    else:
        print("データ収集失敗")


if __name__ == "__main__":
    main()

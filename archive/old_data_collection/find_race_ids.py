"""
netkeiba.comから実際のレースIDを取得するスクリプト
"""
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RaceIDFinder:
    """netkeiba.comから実際のレースIDを見つける"""

    def __init__(self, delay: float = 2.0):
        self.base_url = "https://race.netkeiba.com"
        self.db_url = "https://db.netkeiba.com"
        self.delay = delay

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def find_race_ids_by_date(self, date_str: str) -> list:
        """
        指定日のレースIDを取得

        Args:
            date_str: 日付 (YYYY-MM-DD形式)

        Returns:
            レースIDのリスト
        """
        try:
            # カレンダーページにアクセス
            url = f"{self.base_url}/top/race_list.html?kaisai_date={date_str.replace('-', '')}"

            logger.info(f"Fetching calendar: {date_str}")
            time.sleep(self.delay)

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # レースへのリンクを探す
            race_ids = []

            # パターン1: href="/race/YYYYMMDDVVRR/"
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                match = re.search(r'/race/(\d{12})/', href)
                if match:
                    race_id = match.group(1)
                    if race_id not in race_ids:
                        race_ids.append(race_id)

            logger.info(f"Found {len(race_ids)} races on {date_str}")
            return race_ids

        except Exception as e:
            logger.error(f"Error finding races for {date_str}: {e}")
            return []

    def find_race_ids_for_period(self, start_date: str, end_date: str) -> list:
        """
        期間内のレースIDを取得

        Args:
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)

        Returns:
            レースIDのリスト
        """
        all_race_ids = []

        # 日付範囲を生成（土日のみ）
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        current = start
        while current <= end:
            # 土曜日(5)または日曜日(6)のみ
            if current.weekday() in [5, 6]:
                date_str = current.strftime("%Y-%m-%d")
                race_ids = self.find_race_ids_by_date(date_str)
                all_race_ids.extend(race_ids)

            current += timedelta(days=1)

        logger.info(f"Total races found: {len(all_race_ids)}")
        return all_race_ids

    def find_recent_races(self, weeks: int = 4) -> list:
        """
        最近N週間のレースIDを取得

        Args:
            weeks: 何週間前まで

        Returns:
            レースIDのリスト
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)

        return self.find_race_ids_for_period(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )


def save_race_ids(race_ids: list, filename: str = "race_ids.txt"):
    """レースIDをファイルに保存"""
    with open(filename, 'w') as f:
        for race_id in race_ids:
            f.write(f"{race_id}\n")
    logger.info(f"Saved {len(race_ids)} race IDs to {filename}")


if __name__ == "__main__":
    print("="*60)
    print("実際のレースID取得ツール")
    print("="*60)
    print()

    finder = RaceIDFinder(delay=2.0)

    print("取得方法を選択してください:")
    print("1. 最近4週間のレースを取得")
    print("2. 指定期間のレースを取得")
    print("3. 特定の日付のレースを取得")

    choice = input("\n選択 (1/2/3): ")

    race_ids = []

    if choice == '1':
        print("\n最近4週間のレースを取得中...")
        race_ids = finder.find_recent_races(weeks=4)

    elif choice == '2':
        start = input("開始日 (YYYY-MM-DD): ")
        end = input("終了日 (YYYY-MM-DD): ")
        print(f"\n{start}から{end}までのレースを取得中...")
        race_ids = finder.find_race_ids_for_period(start, end)

    elif choice == '3':
        date = input("日付 (YYYY-MM-DD): ")
        print(f"\n{date}のレースを取得中...")
        race_ids = finder.find_race_ids_by_date(date)

    else:
        print("無効な選択です")
        exit(1)

    if race_ids:
        print(f"\n{len(race_ids)}件のレースIDを取得しました")
        print("\n最初の10件:")
        for race_id in race_ids[:10]:
            print(f"  {race_id}")

        save = input("\nrace_ids.txtに保存しますか? (y/n): ")
        if save.lower() == 'y':
            save_race_ids(race_ids)
            print("\n保存完了！")
            print("次に以下のコマンドで収集を実行してください:")
            print("  python collect_data.py")
            print("  → 3. カスタム収集を選択")
    else:
        print("\nレースIDが見つかりませんでした")

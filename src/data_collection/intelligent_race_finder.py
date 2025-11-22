"""
netkeiba.comから実際のレースIDを自動発見するインテリジェントスクレイパー
"""
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
from typing import List, Set
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntelligentRaceFinder:
    """
    netkeibaから実際のレースIDを自動的に発見する

    戦略:
    1. 月別カレンダーページから開催日を取得
    2. 各開催日のレース一覧ページにアクセス
    3. 実際に存在するレースIDを抽出
    """

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        })

    def find_race_ids_from_monthly_calendar(self, year: int, month: int) -> Set[str]:
        """
        月別カレンダーから全レースIDを取得

        Args:
            year: 年
            month: 月

        Returns:
            レースIDのセット
        """
        race_ids = set()

        try:
            # 月別カレンダーURL
            url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month:02d}"

            logger.info(f"Accessing calendar: {year}/{month:02d}")
            time.sleep(self.delay)

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # レースリンクを全て探す
            # パターン1: /race/YYYYMMDDVVRR/
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                match = re.search(r'/race/(\d{12})/', href)
                if match:
                    race_id = match.group(1)
                    race_ids.add(race_id)

            logger.info(f"Found {len(race_ids)} races in {year}/{month:02d}")

        except Exception as e:
            logger.error(f"Error accessing calendar {year}/{month:02d}: {e}")

        return race_ids

    def find_race_ids_from_date_list(self, year: int, month: int) -> Set[str]:
        """
        日付一覧ページから全レースIDを取得（代替手法）

        Args:
            year: 年
            month: 月

        Returns:
            レースIDのセット
        """
        race_ids = set()

        # その月の全ての日をチェック（土日を中心に）
        start_date = datetime(year, month, 1)

        # 次の月の1日を計算
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        current = start_date

        while current < end_date:
            # 土曜日(5)、日曜日(6)、祝日を優先的にチェック
            if current.weekday() in [5, 6]:  # 土日
                date_race_ids = self._find_race_ids_for_date(current)
                race_ids.update(date_race_ids)

            current += timedelta(days=1)

        return race_ids

    def _find_race_ids_for_date(self, date: datetime) -> Set[str]:
        """
        特定日のレースIDを取得

        Args:
            date: 日付

        Returns:
            レースIDのセット
        """
        race_ids = set()

        try:
            date_str = date.strftime("%Y%m%d")

            # race.netkeiba.comのトップページから開催情報を取得
            url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"

            logger.info(f"Checking date: {date.strftime('%Y-%m-%d')}")
            time.sleep(self.delay)

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # レースリンクを探す
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                match = re.search(r'/race/(\d{12})/', href)
                if match:
                    race_id = match.group(1)
                    race_ids.add(race_id)

            if race_ids:
                logger.info(f"  Found {len(race_ids)} races on {date.strftime('%Y-%m-%d')}")

        except Exception as e:
            logger.debug(f"No races on {date.strftime('%Y-%m-%d')}: {e}")

        return race_ids

    def find_race_ids_by_venue_and_period(
        self,
        year: int,
        start_month: int,
        end_month: int,
        max_races: int = 500
    ) -> List[str]:
        """
        期間を指定して全レースIDを取得

        Args:
            year: 年
            start_month: 開始月
            end_month: 終了月
            max_races: 最大レース数

        Returns:
            レースIDのリスト
        """
        all_race_ids = set()

        for month in range(start_month, end_month + 1):
            # 方法1: カレンダーから取得
            race_ids_calendar = self.find_race_ids_from_monthly_calendar(year, month)
            all_race_ids.update(race_ids_calendar)

            # 十分なレースが見つかったら終了
            if len(all_race_ids) >= max_races:
                break

            # 方法1で見つからない場合、方法2を試す
            if len(race_ids_calendar) == 0:
                logger.info(f"Trying alternative method for {year}/{month:02d}")
                race_ids_date = self.find_race_ids_from_date_list(year, month)
                all_race_ids.update(race_ids_date)

        race_id_list = sorted(list(all_race_ids))

        logger.info(f"\nTotal unique races found: {len(race_id_list)}")

        # 最大数に制限
        if len(race_id_list) > max_races:
            race_id_list = race_id_list[:max_races]
            logger.info(f"Limited to {max_races} races")

        return race_id_list

    def find_recent_races(self, months_back: int = 3, max_races: int = 100) -> List[str]:
        """
        最近N ヶ月のレースを取得

        Args:
            months_back: 何ヶ月前まで
            max_races: 最大レース数

        Returns:
            レースIDのリスト
        """
        all_race_ids = set()

        # 現在の日付から遡る
        end_date = datetime.now()

        for i in range(months_back):
            target_date = end_date - timedelta(days=30 * i)
            year = target_date.year
            month = target_date.month

            race_ids = self.find_race_ids_from_monthly_calendar(year, month)
            all_race_ids.update(race_ids)

            if len(all_race_ids) >= max_races:
                break

        race_id_list = sorted(list(all_race_ids), reverse=True)  # 新しい順

        if len(race_id_list) > max_races:
            race_id_list = race_id_list[:max_races]

        return race_id_list


def save_race_ids_to_file(race_ids: List[str], filename: str = "discovered_race_ids.txt"):
    """レースIDをファイルに保存"""
    with open(filename, 'w') as f:
        for race_id in race_ids:
            f.write(f"{race_id}\n")
    logger.info(f"Saved {len(race_ids)} race IDs to {filename}")


if __name__ == "__main__":
    print("="*60)
    print("インテリジェント レースID 発見ツール")
    print("="*60)
    print()

    finder = IntelligentRaceFinder(delay=2.0)

    print("取得方法を選択してください:")
    print("1. 最近3ヶ月のレースを取得（推奨）")
    print("2. 指定期間のレースを取得")
    print("3. 2024年全体のレースを取得")

    choice = input("\n選択 (1/2/3): ")

    race_ids = []

    if choice == '1':
        print("\n最近3ヶ月のレースを取得中...")
        race_ids = finder.find_recent_races(months_back=3, max_races=100)

    elif choice == '2':
        year = int(input("年 (例: 2024): "))
        start_month = int(input("開始月 (1-12): "))
        end_month = int(input("終了月 (1-12): "))
        max_races = int(input("最大レース数 (例: 200): "))

        print(f"\n{year}年{start_month}月〜{end_month}月のレースを取得中...")
        race_ids = finder.find_race_ids_by_venue_and_period(
            year, start_month, end_month, max_races
        )

    elif choice == '3':
        print("\n2024年全体のレースを取得中...")
        race_ids = finder.find_race_ids_by_venue_and_period(
            2024, 1, 12, max_races=500
        )

    else:
        print("無効な選択です")
        exit(1)

    if race_ids:
        print(f"\n{len(race_ids)}件のレースIDを取得しました")
        print("\n最初の10件:")
        for i, race_id in enumerate(race_ids[:10], 1):
            print(f"  {i}. {race_id}")

        if len(race_ids) > 10:
            print(f"  ... (残り{len(race_ids) - 10}件)")

        save = input("\ndiscovered_race_ids.txtに保存しますか? (y/n): ")
        if save.lower() == 'y':
            save_race_ids_to_file(race_ids)
            print("\n保存完了！")
            print("\n次に以下のコマンドでデータ収集を実行してください:")
            print("  python collect_data.py")
            print("  → 3. カスタム収集")
            print("  → race_ids.txtの代わりにdiscovered_race_ids.txtを指定")
    else:
        print("\nレースIDが見つかりませんでした")

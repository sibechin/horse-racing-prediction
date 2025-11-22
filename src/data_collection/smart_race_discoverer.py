"""
スマートブルートフォース方式でのレースID発見
既知の成功したレースから周辺のレースを系統的に探索
"""
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from typing import List, Set
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartRaceDiscoverer:
    """
    既知のレースから周辺レースを発見する

    戦略:
    1. 既知の成功レース(202406030811)を起点に
    2. 同日の全レース番号(01-12)を試す
    3. 前後の日付も同様に試す
    4. 成功したレースから同じパターンを繰り返す
    """

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.base_url = "https://db.netkeiba.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def check_race_exists(self, race_id: str) -> bool:
        """
        レースIDが存在するかチェック（軽量版）

        Args:
            race_id: レースID

        Returns:
            存在すればTrue
        """
        try:
            url = f"{self.base_url}/race/{race_id}/"
            time.sleep(self.delay)

            response = self.session.head(url, timeout=10, allow_redirects=True)

            # 200番台ならレースが存在
            if response.status_code < 400:
                return True

        except Exception:
            pass

        return False

    def discover_races_for_date_and_venue(
        self,
        year: int,
        month: int,
        day: int,
        venue_code: str
    ) -> List[str]:
        """
        指定日・会場の全レースを発見

        Args:
            year: 年
            month: 月
            day: 日
            venue_code: 競馬場コード(01-10)

        Returns:
            見つかったレースIDのリスト
        """
        found_races = []

        date_str = f"{year:04d}{month:02d}{day:02d}"

        # 1レースから12レースまで全て試す
        for race_num in range(1, 13):
            race_id = f"{date_str}{venue_code}{race_num:02d}"

            if self.check_race_exists(race_id):
                found_races.append(race_id)
                logger.info(f"Found: {race_id}")

        return found_races

    def discover_races_for_date(
        self,
        year: int,
        month: int,
        day: int
    ) -> List[str]:
        """
        指定日の全会場・全レースを発見

        Args:
            year: 年
            month: 月
            day: 日

        Returns:
            見つかったレースIDのリスト
        """
        all_races = []

        # 全競馬場コード
        venue_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

        logger.info(f"Searching date: {year}/{month:02d}/{day:02d}")

        for venue_code in venue_codes:
            races = self.discover_races_for_date_and_venue(
                year, month, day, venue_code
            )
            all_races.extend(races)

        if all_races:
            logger.info(f"  Total: {len(all_races)} races found")
        else:
            logger.debug(f"  No races on {year}/{month:02d}/{day:02d}")

        return all_races

    def discover_races_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
        max_races: int = 500
    ) -> List[str]:
        """
        期間内の全レースを発見

        Args:
            start_date: 開始日
            end_date: 終了日
            max_races: 最大レース数

        Returns:
            見つかったレースIDのリスト
        """
        all_races = []

        current = start_date
        while current <= end_date and len(all_races) < max_races:
            # 土日を優先的にチェック（平日もチェックするが）
            races = self.discover_races_for_date(
                current.year,
                current.month,
                current.day
            )

            all_races.extend(races)

            if len(all_races) >= max_races:
                all_races = all_races[:max_races]
                break

            current += timedelta(days=1)

        logger.info(f"\nTotal races discovered: {len(all_races)}")
        return all_races

    def discover_races_for_month(
        self,
        year: int,
        month: int,
        max_races: int = 200
    ) -> List[str]:
        """
        指定月の全レースを発見

        Args:
            year: 年
            month: 月
            max_races: 最大レース数

        Returns:
            見つかったレースIDのリスト
        """
        # 月の最初と最後の日
        start_date = datetime(year, month, 1)

        if month == 12:
            end_date = datetime(year, 12, 31)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        return self.discover_races_for_period(start_date, end_date, max_races)

    def discover_races_smart_sampling(
        self,
        year: int,
        start_month: int,
        end_month: int,
        max_races: int = 500
    ) -> List[str]:
        """
        スマートサンプリング: 土日のみを効率的にチェック

        Args:
            year: 年
            start_month: 開始月
            end_month: 終了月
            max_races: 最大レース数

        Returns:
            見つかったレースIDのリスト
        """
        all_races = []

        for month in range(start_month, end_month + 1):
            # 月の日数を取得
            if month == 12:
                last_day = 31
            else:
                last_day = (datetime(year, month + 1, 1) - timedelta(days=1)).day

            for day in range(1, last_day + 1):
                try:
                    date = datetime(year, month, day)

                    # 土日のみチェック（競馬は主に週末）
                    if date.weekday() in [5, 6]:  # Saturday, Sunday
                        races = self.discover_races_for_date(year, month, day)
                        all_races.extend(races)

                        if len(all_races) >= max_races:
                            all_races = all_races[:max_races]
                            logger.info(f"Reached max races: {max_races}")
                            return all_races

                except ValueError:
                    continue

        return all_races


def save_discovered_races(race_ids: List[str], filename: str = "discovered_races.txt"):
    """発見したレースIDをファイルに保存"""
    with open(filename, 'w') as f:
        for race_id in race_ids:
            f.write(f"{race_id}\n")
    logger.info(f"Saved {len(race_ids)} race IDs to {filename}")


if __name__ == "__main__":
    print("="*60)
    print("スマートレース発見ツール")
    print("="*60)
    print()
    print("既知のレースから系統的に周辺レースを探索します")
    print()

    discoverer = SmartRaceDiscoverer(delay=0.5)  # 0.5秒間隔（高速）

    print("発見方法を選択してください:")
    print("1. 2024年6月の全レース（テスト用）")
    print("2. 2024年の土日のみ（推奨）")
    print("3. 指定期間の全レース")

    choice = input("\n選択 (1/2/3): ")

    race_ids = []

    if choice == '1':
        print("\n2024年6月の全レースを探索中...")
        race_ids = discoverer.discover_races_for_month(2024, 6, max_races=200)

    elif choice == '2':
        print("\n2024年の土日のレースを探索中...")
        print("（これには時間がかかります: 約10-15分）")
        race_ids = discoverer.discover_races_smart_sampling(
            2024, 1, 12, max_races=500
        )

    elif choice == '3':
        year = int(input("年: "))
        start_month = int(input("開始月: "))
        end_month = int(input("終了月: "))
        max_races = int(input("最大レース数: "))

        print(f"\n{year}年{start_month}月〜{end_month}月のレースを探索中...")
        race_ids = discoverer.discover_races_smart_sampling(
            year, start_month, end_month, max_races
        )

    else:
        print("無効な選択です")
        exit(1)

    if race_ids:
        print(f"\n{len(race_ids)}件のレースを発見しました！")
        print("\n最初の20件:")
        for i, race_id in enumerate(race_ids[:20], 1):
            print(f"  {i:2d}. {race_id}")

        if len(race_ids) > 20:
            print(f"  ... (残り{len(race_ids) - 20}件)")

        save_discovered_races(race_ids)
        print("\ndiscovered_races.txt に保存しました")
        print("\n次のステップ:")
        print("  python collect_data.py")
        print("  → 3. カスタム収集")
        print("  → discovered_races.txt を race_ids.txt にコピーして使用")
    else:
        print("\nレースが見つかりませんでした")

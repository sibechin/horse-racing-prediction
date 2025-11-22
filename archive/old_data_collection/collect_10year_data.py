"""
過去10年分（2014-2024）のレースデータを収集
直近4年（2021-2024）を重点的に収集
"""
import sys
sys.path.append('.')

from src.data_collection.smart_race_discoverer import SmartRaceDiscoverer
from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_races_by_year_range(
    start_year: int,
    end_year: int,
    max_races_per_year: int = 100
) -> list:
    """
    指定年範囲のレースIDを発見

    Args:
        start_year: 開始年
        end_year: 終了年
        max_races_per_year: 年あたりの最大レース数

    Returns:
        レースIDのリスト
    """
    discoverer = SmartRaceDiscoverer(delay=0.3)  # 高速化
    all_race_ids = []

    # 新しい年から順に探索（最新データを優先）
    for year in range(end_year, start_year - 1, -1):
        logger.info(f"\n{'='*60}")
        logger.info(f"{year}年のレース探索開始")
        logger.info(f"{'='*60}")

        # 土日のみを効率的にチェック
        year_races = discoverer.discover_races_smart_sampling(
            year=year,
            start_month=1,
            end_month=12,
            max_races=max_races_per_year
        )

        all_race_ids.extend(year_races)

        logger.info(f"{year}年: {len(year_races)}レース発見")
        logger.info(f"累計: {len(all_race_ids)}レース")

    return all_race_ids


def main():
    print("="*60)
    print("過去10年分のデータ収集")
    print("="*60)
    print()
    print("戦略:")
    print("  - 直近4年（2021-2024）: 各年100レース = 400レース")
    print("  - 過去6年（2014-2020）: 各年50レース = 350レース")
    print("  - 合計目標: 750レース")
    print()

    input("Enterキーで開始...")

    # フェーズ1: 直近4年（2021-2024）を重点収集
    print("\n" + "="*60)
    print("フェーズ1: 直近4年（2021-2024）のレースID発見")
    print("="*60)

    recent_races = discover_races_by_year_range(
        start_year=2021,
        end_year=2024,
        max_races_per_year=100
    )

    print(f"\n直近4年のレース数: {len(recent_races)}")

    # 保存
    with open('race_ids_recent_4years.txt', 'w') as f:
        for race_id in recent_races:
            f.write(f"{race_id}\n")
    print("保存: race_ids_recent_4years.txt")

    # フェーズ2: 過去6年（2014-2020）を収集
    print("\n" + "="*60)
    print("フェーズ2: 過去6年（2014-2020）のレースID発見")
    print("="*60)

    past_races = discover_races_by_year_range(
        start_year=2014,
        end_year=2020,
        max_races_per_year=50
    )

    print(f"\n過去6年のレース数: {len(past_races)}")

    # 保存
    with open('race_ids_past_6years.txt', 'w') as f:
        for race_id in past_races:
            f.write(f"{race_id}\n")
    print("保存: race_ids_past_6years.txt")

    # 統合
    all_races = recent_races + past_races

    with open('race_ids_10years.txt', 'w') as f:
        for race_id in all_races:
            f.write(f"{race_id}\n")

    print("\n" + "="*60)
    print("レースID発見完了")
    print("="*60)
    print(f"直近4年: {len(recent_races)}レース")
    print(f"過去6年: {len(past_races)}レース")
    print(f"合計: {len(all_races)}レース")
    print("\n保存先: race_ids_10years.txt")

    # フェーズ3: データ収集
    print("\n" + "="*60)
    print("フェーズ3: 実データ収集を開始しますか？")
    print("="*60)
    print(f"収集予定: {len(all_races)}レース")
    print(f"推定時間: {len(all_races) * 2 / 60:.1f}分 ({len(all_races) * 2 / 3600:.1f}時間)")

    proceed = input("\n収集を開始しますか？ (y/n): ")

    if proceed.lower() == 'y':
        print("\nデータ収集開始...")

        scraper = NetkeibaScraperComplete(delay=1.5, max_retries=3)

        results = scraper.scrape_multiple_races(
            all_races,
            output_dir='data/raw'
        )

        if len(results) > 0:
            print("\n" + "="*60)
            print("収集完了!")
            print("="*60)
            print(f"成功レース数: {results['race_id'].nunique()}")
            print(f"総データ数: {len(results)}頭")
            print(f"期間: {results['race_date'].min()} - {results['race_date'].max()}")
            print("\n年別分布:")

            # 年別集計
            results['year'] = results['race_date'].str[:4]
            year_counts = results.groupby('year')['race_id'].nunique()
            for year, count in sorted(year_counts.items()):
                print(f"  {year}年: {count}レース")
        else:
            print("\nデータ収集失敗")
    else:
        print("\nデータ収集はスキップされました")
        print("後で以下のコマンドで収集できます:")
        print("  python collect_data.py")
        print("  → 3. カスタム収集")
        print("  → race_ids_10years.txt を指定")


if __name__ == "__main__":
    main()

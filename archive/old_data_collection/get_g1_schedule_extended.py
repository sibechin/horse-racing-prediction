"""
G1レース開催日取得スクリプト（拡張版）

2000年代以降の主要G1レースの推定開催日を生成
"""
import sys
from pathlib import Path
sys.path.append('.')

import pandas as pd
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 主要G1レースの開催パターン（月・週・競馬場）
# 競馬場コード: 東京=05, 中山=06, 中京=07, 京都=08, 阪神=09
G1_RACE_PATTERNS = {
    'フェブラリーS': {'month': 2, 'week': 3, 'venue': '東京/中山', 'venue_codes': ['05', '06']},
    '高松宮記念': {'month': 3, 'week': 4, 'venue': '中京', 'venue_codes': ['07']},
    '大阪杯': {'month': 3, 'week': 5, 'venue': '阪神', 'venue_codes': ['09']},  # 2017年から
    '桜花賞': {'month': 4, 'week': 2, 'venue': '阪神', 'venue_codes': ['09']},
    '皐月賞': {'month': 4, 'week': 3, 'venue': '中山', 'venue_codes': ['06']},
    '天皇賞(春)': {'month': 4, 'week': 4, 'venue': '京都', 'venue_codes': ['08']},
    'NHKマイルC': {'month': 5, 'week': 2, 'venue': '東京', 'venue_codes': ['05']},
    'ヴィクトリアM': {'month': 5, 'week': 3, 'venue': '東京', 'venue_codes': ['05']},
    'オークス': {'month': 5, 'week': 4, 'venue': '東京', 'venue_codes': ['05']},
    '日本ダービー': {'month': 5, 'week': 4, 'venue': '東京', 'venue_codes': ['05']},
    '安田記念': {'month': 6, 'week': 1, 'venue': '東京', 'venue_codes': ['05']},
    '宝塚記念': {'month': 6, 'week': 4, 'venue': '阪神/京都', 'venue_codes': ['09', '08']},
    'スプリンターズS': {'month': 9, 'week': 4, 'venue': '中山', 'venue_codes': ['06']},
    '秋華賞': {'month': 10, 'week': 3, 'venue': '京都', 'venue_codes': ['08']},
    '菊花賞': {'month': 10, 'week': 4, 'venue': '京都', 'venue_codes': ['08']},
    '天皇賞(秋)': {'month': 10, 'week': 4, 'venue': '東京', 'venue_codes': ['05']},
    'エリザベス女王杯': {'month': 11, 'week': 2, 'venue': '京都', 'venue_codes': ['08']},
    'マイルCS': {'month': 11, 'week': 3, 'venue': '京都', 'venue_codes': ['08']},
    'ジャパンC': {'month': 11, 'week': 4, 'venue': '東京', 'venue_codes': ['05']},
    'チャンピオンズC': {'month': 12, 'week': 1, 'venue': '中山', 'venue_codes': ['06']},
    '有馬記念': {'month': 12, 'week': 4, 'venue': '中山', 'venue_codes': ['06']},
}


def generate_race_urls_for_year(year: int, race_name: str, pattern: dict) -> list:
    """
    特定年の特定レースのURL候補を生成

    Args:
        year: 年
        race_name: レース名
        pattern: レースパターン情報

    Returns:
        (race_id, date, name, url, venue)のリスト
    """
    month = pattern['month']
    week = pattern['week']
    venue_codes = pattern['venue_codes']
    venue_name = pattern['venue']

    urls = []

    # その月の第X週の推定日付範囲
    start_day = (week - 1) * 7 + 1

    # 土日を含む2週間分を探索（±7日）
    for day_offset in range(-3, 14):
        day = start_day + day_offset
        if day < 1 or day > 31:
            continue

        # 各競馬場コードを試す
        for venue_code in venue_codes:
            # 11R, 12Rを試す
            for race_num in [11, 12]:
                race_id = f"{year}{month:02d}{day:02d}{venue_code}{race_num:02d}"
                url = f"https://www.keibalab.jp/db/race/{race_id}/"
                date_str = f"{year}-{month:02d}-{day:02d}"
                urls.append((race_id, date_str, race_name, url, venue_name))

    return urls


def generate_extended_schedule(start_year: int = 2000, end_year: int = 2024) -> dict:
    """
    複数年にわたるG1スケジュールを生成

    Args:
        start_year: 開始年
        end_year: 終了年

    Returns:
        年ごとのレース情報の辞書
    """
    logging.info(f"{start_year}-{end_year}年のG1スケジュールを生成中...")

    all_schedules = {}

    for year in range(start_year, end_year + 1):
        year_urls = []

        for race_name, pattern in G1_RACE_PATTERNS.items():
            # 大阪杯は2017年から
            if race_name == '大阪杯' and year < 2017:
                continue

            race_urls = generate_race_urls_for_year(year, race_name, pattern)
            year_urls.extend(race_urls)

        all_schedules[year] = year_urls
        logging.info(f"  {year}年: {len(year_urls)}URL候補")

    return all_schedules


def save_extended_schedule(all_schedules: dict):
    """
    拡張スケジュールを保存

    Args:
        all_schedules: 年ごとのレース情報の辞書
    """
    output_dir = Path('data/schedules')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 年ごとに保存
    for year, urls in all_schedules.items():
        # CSVとして保存
        df = pd.DataFrame(urls, columns=['race_id', 'date', 'race_name', 'url', 'venue'])
        csv_file = output_dir / f'keibalab_urls_{year}.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        # テキストファイルとしても保存
        txt_file = output_dir / f'keibalab_urls_{year}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            for race_id, date, name, url, venue in urls:
                f.write(f"{race_id}\t{date}\t{name}\t{url}\n")

        logging.info(f"  {year}年のスケジュールを保存: {csv_file}")

    # 統合版も保存
    all_urls = []
    for year, urls in all_schedules.items():
        all_urls.extend(urls)

    df_all = pd.DataFrame(all_urls, columns=['race_id', 'date', 'race_name', 'url', 'venue'])
    csv_all_file = output_dir / 'keibalab_urls_all.csv'
    df_all.to_csv(csv_all_file, index=False, encoding='utf-8-sig')
    logging.info(f"\n統合スケジュールを保存: {csv_all_file}")
    logging.info(f"  総URL候補数: {len(all_urls)}")

    # サマリー情報を保存
    summary = {
        'start_year': min(all_schedules.keys()),
        'end_year': max(all_schedules.keys()),
        'total_years': len(all_schedules),
        'total_urls': len(all_urls),
        'urls_per_year': {year: len(urls) for year, urls in all_schedules.items()},
    }

    summary_file = output_dir / 'schedule_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"サマリー情報を保存: {summary_file}")


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("G1レース開催日取得スクリプト（拡張版）")
    logging.info("=" * 60)
    logging.info("")

    # 2000-2024年のスケジュールを生成
    start_year = 2000
    end_year = 2024

    logging.info(f"対象期間: {start_year}-{end_year}年")
    logging.info(f"G1レース種類: {len(G1_RACE_PATTERNS)}")
    logging.info("")

    # スケジュール生成
    all_schedules = generate_extended_schedule(start_year, end_year)

    # 保存
    save_extended_schedule(all_schedules)

    logging.info("\n" + "=" * 60)
    logging.info("完了!")
    logging.info("=" * 60)
    logging.info(f"対象年数: {len(all_schedules)}年")
    logging.info(f"総URL候補数: {sum(len(urls) for urls in all_schedules.values())}")
    logging.info(f"推定所要時間: {sum(len(urls) for urls in all_schedules.values()) * 12 / 3600:.1f}時間")
    logging.info("")
    logging.info("注意:")
    logging.info("  - これは推定URL候補です")
    logging.info("  - 実際のレース開催日は年によって異なる場合があります")
    logging.info("  - 12秒のクロールディレイを遵守してください")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

"""
keibalab.jp 2025年G1データ収集
"""
import sys
from pathlib import Path
sys.path.append('.')

from src.data_collection.keibalab_scraper import KeibalabScraper
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 2025年G1レーススケジュール（予定日から推定）
G1_SCHEDULE_2025 = {
    'フェブラリーS': {'month': 2, 'day_range': (15, 25), 'venue_codes': ['05', '06']},
    '高松宮記念': {'month': 3, 'day_range': (20, 31), 'venue_codes': ['07']},
    '大阪杯': {'month': 3, 'day_range': (28, 31), 'venue_codes': ['09']},
    '桜花賞': {'month': 4, 'day_range': (10, 15), 'venue_codes': ['09']},
    '皐月賞': {'month': 4, 'day_range': (17, 22), 'venue_codes': ['06']},
    '天皇賞(春)': {'month': 4, 'day_range': (25, 30), 'venue_codes': ['08']},
    'NHKマイルC': {'month': 5, 'day_range': (8, 13), 'venue_codes': ['05']},
    'ヴィクトリアM': {'month': 5, 'day_range': (15, 20), 'venue_codes': ['05']},
    'オークス': {'month': 5, 'day_range': (22, 27), 'venue_codes': ['05']},
    '日本ダービー': {'month': 5, 'day_range': (28, 31), 'venue_codes': ['05']},
    '安田記念': {'month': 6, 'day_range': (5, 10), 'venue_codes': ['05']},
    '宝塚記念': {'month': 6, 'day_range': (26, 30), 'venue_codes': ['09', '08']},
}


def generate_2025_urls():
    """
    2025年のG1レースURL候補を生成

    Returns:
        (race_id, date, name, url)のリスト
    """
    year = 2025
    urls = []

    for race_name, schedule in G1_SCHEDULE_2025.items():
        month = schedule['month']
        day_start, day_end = schedule['day_range']
        venue_codes = schedule['venue_codes']

        # 日付範囲でループ
        for day in range(day_start, day_end + 1):
            # 各競馬場コードを試す
            for venue_code in venue_codes:
                # 11R, 12Rを試す
                for race_num in [11, 12]:
                    race_id = f"{year}{month:02d}{day:02d}{venue_code}{race_num:02d}"
                    url = f"https://www.keibalab.jp/db/race/{race_id}/"
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    urls.append((race_id, date_str, race_name, url))

    logging.info(f"2025年G1レース候補URL数: {len(urls)}")
    return urls


def collect_2025_g1_data(delay: float = 2.0):
    """
    2025年G1データを収集

    Args:
        delay: クロールディレイ（秒）
    """
    logging.info("="*60)
    logging.info("2025年G1データ収集開始")
    logging.info("="*60)

    # URL候補生成
    urls = generate_2025_urls()

    logging.info(f"\n収集予定:")
    logging.info(f"  G1レース種類: {len(G1_SCHEDULE_2025)}種類")
    logging.info(f"  URL候補数: {len(urls)}")
    logging.info(f"  推定所要時間: {len(urls) * delay / 60:.1f}分")
    logging.info(f"  クロールディレイ: {delay}秒/URL")
    logging.info("")

    # 確認
    print("注意: 2025年のレースは未開催の可能性があります")
    print("開催済みのレースのみが収集されます")
    print("")
    response = input("収集を開始しますか？ (y/n): ")

    if response.lower() != 'y':
        logging.info("収集をキャンセルしました")
        return

    # スクレイパー初期化
    scraper = KeibalabScraper(delay=delay, max_retries=3)

    # データ収集
    results = []
    successful = 0
    failed = 0

    for i, (race_id, date, name, url) in enumerate(urls, 1):
        if i % 20 == 0:
            success_rate = successful / i * 100 if i > 0 else 0
            logging.info(f"\n[{i}/{len(urls)}] 進捗: 成功 {successful}, 失敗 {failed} ({success_rate:.1f}%)")

        # レースデータ収集
        df = scraper.scrape_race(url)

        if df is not None and len(df) > 0:
            # メタデータを追加
            df['expected_race_name'] = name
            df['expected_date'] = date
            df['expected_year'] = 2025
            df['race_id_keibalab'] = race_id

            results.append(df)
            successful += 1
        else:
            failed += 1

    # 結果を統合
    if results:
        combined_df = pd.concat(results, ignore_index=True)

        # 保存
        output_dir = Path('data/raw/keibalab')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'g1_2025_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        # サマリー
        logging.info("\n" + "="*60)
        logging.info("データ収集完了!")
        logging.info("="*60)
        logging.info(f"成功したURL: {successful}/{len(urls)} ({successful / len(urls) * 100:.1f}%)")
        logging.info(f"収集データ数: {len(combined_df)}頭")
        logging.info(f"ユニークレース数: {combined_df['race_id_keibalab'].nunique()}個")
        logging.info(f"保存先: {output_file}")

        # レース別統計
        logging.info("\nG1レース別収集数:")
        race_counts = combined_df.groupby('expected_race_name').size().sort_values(ascending=False)
        print(race_counts)

        return combined_df
    else:
        logging.error("データ収集に失敗しました（全て404エラー）")
        logging.info("2025年のレースがまだ開催されていない可能性があります")
        return None


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("keibalab.jp 2025年G1データ収集スクリプト")
    logging.info("="*60)

    # データ収集
    collect_2025_g1_data(delay=2.0)


if __name__ == "__main__":
    main()

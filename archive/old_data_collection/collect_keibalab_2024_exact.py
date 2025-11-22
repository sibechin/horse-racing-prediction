"""
keibalab.jp 2024年G1データ収集 (正確な開催日版)

JRA公式サイトから取得した正確な開催日に基づいて収集
"""
import sys
from pathlib import Path
sys.path.append('.')

from src.data_collection.keibalab_scraper import KeibalabScraper
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 2024年G1レース (JRA公式より)
# 競馬場コード: 東京=05, 中山=06, 中京=07, 京都=08, 阪神=09
G1_RACES_2024 = [
    {'name': 'フェブラリーS', 'date': '0218', 'venue': '05', 'venue_name': '東京'},
    {'name': '高松宮記念', 'date': '0324', 'venue': '07', 'venue_name': '中京'},
    {'name': '大阪杯', 'date': '0331', 'venue': '09', 'venue_name': '阪神'},
    {'name': '桜花賞', 'date': '0407', 'venue': '09', 'venue_name': '阪神'},
    {'name': '皐月賞', 'date': '0414', 'venue': '06', 'venue_name': '中山'},
    {'name': '天皇賞(春)', 'date': '0428', 'venue': '08', 'venue_name': '京都'},
    {'name': 'NHKマイルC', 'date': '0505', 'venue': '05', 'venue_name': '東京'},
    {'name': 'ヴィクトリアM', 'date': '0512', 'venue': '05', 'venue_name': '東京'},
    {'name': 'オークス', 'date': '0519', 'venue': '05', 'venue_name': '東京'},
    {'name': '日本ダービー', 'date': '0526', 'venue': '05', 'venue_name': '東京'},
    {'name': '安田記念', 'date': '0602', 'venue': '05', 'venue_name': '東京'},
    {'name': '宝塚記念', 'date': '0623', 'venue': '08', 'venue_name': '京都'},
    {'name': 'スプリンターズS', 'date': '0929', 'venue': '06', 'venue_name': '中山'},
    {'name': '秋華賞', 'date': '1013', 'venue': '08', 'venue_name': '京都'},
    {'name': '菊花賞', 'date': '1020', 'venue': '08', 'venue_name': '京都'},
    {'name': '天皇賞(秋)', 'date': '1027', 'venue': '05', 'venue_name': '東京'},
    {'name': 'エリザベス女王杯', 'date': '1110', 'venue': '08', 'venue_name': '京都'},
    {'name': 'マイルCS', 'date': '1117', 'venue': '08', 'venue_name': '京都'},
    {'name': 'ジャパンC', 'date': '1124', 'venue': '05', 'venue_name': '東京'},
    {'name': 'チャンピオンズC', 'date': '1201', 'venue': '05', 'venue_name': '東京'},
    {'name': '有馬記念', 'date': '1222', 'venue': '06', 'venue_name': '中山'},
]


def generate_race_url_candidates(race_info: dict) -> list:
    """
    指定レースのURL候補を生成

    Args:
        race_info: レース情報 (date, venue, name)

    Returns:
        (URL, レース名)のリスト
    """
    urls = []
    year = 2024
    date = race_info['date']  # MMDD形式
    venue = race_info['venue']
    name = race_info['name']

    # 11R, 12Rを両方試す (G1は通常11Rか12R)
    for race_num in [11, 12]:
        race_id = f"{year}{date}{venue}{race_num:02d}"
        url = f"https://www.keibalab.jp/db/race/{race_id}/"
        urls.append((url, name, race_id))

    return urls


def collect_keibalab_2024_exact():
    """
    2024年G1データを正確な開催日で収集
    """
    logging.info("=" * 60)
    logging.info("keibalab.jp 2024年G1データ収集 (正確な開催日版)")
    logging.info("=" * 60)
    logging.info(f"G1レース数: {len(G1_RACES_2024)}")
    logging.info(f"URL候補数: {len(G1_RACES_2024) * 2} (各レース11R/12R)")
    logging.info(f"推定所要時間: {len(G1_RACES_2024) * 2 * 12 / 60:.1f}分")
    logging.info("")

    # 確認
    print("注意:")
    print(f"  - JRA公式サイトから取得した正確な開催日で収集")
    print(f"  - {len(G1_RACES_2024)}レース × 2候補(11R/12R) = {len(G1_RACES_2024) * 2}URL")
    print(f"  - クロールディレイ: 12秒/レース")
    print("")
    response = input("収集を開始しますか？ (y/n): ")

    if response.lower() != 'y':
        logging.info("収集をキャンセルしました")
        return

    # スクレイパー初期化 (12秒ディレイ)
    scraper = KeibalabScraper(delay=12.0, max_retries=3)

    # データ収集
    all_results = []
    successful_races = 0
    failed_races = 0
    race_details = []

    for i, race_info in enumerate(G1_RACES_2024, 1):
        logging.info(f"\n[{i}/{len(G1_RACES_2024)}] {race_info['name']}")
        logging.info(f"  日付: 2024年{race_info['date'][:2]}月{race_info['date'][2:]}日")
        logging.info(f"  競馬場: {race_info['venue_name']}")

        # URL候補を生成 (11R, 12R)
        url_candidates = generate_race_url_candidates(race_info)

        race_found = False
        for url, name, race_id in url_candidates:
            logging.info(f"  試行: {race_id}")

            # レースデータ収集
            df = scraper.scrape_race(url)

            if df is not None and len(df) > 0:
                # メタデータを追加
                df['race_name_expected'] = name
                df['race_date_expected'] = f"2024-{race_info['date'][:2]}-{race_info['date'][2:]}"
                df['venue_expected'] = race_info['venue_name']

                all_results.append(df)
                successful_races += 1
                race_found = True

                race_details.append({
                    'race_name': name,
                    'race_id': race_id,
                    'date': race_info['date'],
                    'venue': race_info['venue_name'],
                    'horses': len(df)
                })

                logging.info(f"  成功 ({len(df)}頭) - レースID: {race_id}")
                break  # 成功したら次のレースへ
            else:
                logging.info(f"  404エラー")

        if not race_found:
            failed_races += 1
            logging.warning(f"  {name}: データが見つかりませんでした")

    # 結果を結合
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # 保存
        output_dir = Path('data/raw/keibalab')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'g1_2024_exact_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        logging.info("\n" + "=" * 60)
        logging.info("データ収集完了!")
        logging.info("=" * 60)
        logging.info(f"成功: {successful_races}/{len(G1_RACES_2024)} レース")
        logging.info(f"失敗: {failed_races}/{len(G1_RACES_2024)} レース")
        logging.info(f"成功率: {successful_races / len(G1_RACES_2024) * 100:.1f}%")
        logging.info(f"総データ数: {len(combined_df)}頭")
        logging.info(f"保存先: {output_file}")
        logging.info("=" * 60)

        # 収集レース一覧
        if race_details:
            logging.info("\n収集されたレース:")
            for detail in race_details:
                logging.info(f"  {detail['race_name']} ({detail['date']}, {detail['venue']}) - {detail['horses']}頭")

        return combined_df
    else:
        logging.error("データ収集に失敗しました")
        return None


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("keibalab.jp 2024年G1データ収集スクリプト")
    logging.info("=" * 60)
    logging.info("")
    logging.info("特徴:")
    logging.info("  - JRA公式サイトから取得した正確な開催日を使用")
    logging.info("  - 無駄な404エラーを最小化")
    logging.info("  - 推定所要時間: 約8分")
    logging.info("")
    logging.info("=" * 60)
    logging.info("")

    # データ収集
    collect_keibalab_2024_exact()


if __name__ == "__main__":
    main()

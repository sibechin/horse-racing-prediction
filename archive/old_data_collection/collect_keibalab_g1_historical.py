"""
keibalab.jp 歴史的G1データ収集スクリプト

非商業利用・研究目的でのデータ収集
- 12秒のクロールディレイ遵守
- 1995-2024年のG1レースデータ収集
"""
import sys
from pathlib import Path
sys.path.append('.')

from src.data_collection.keibalab_scraper import KeibalabScraper
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 主要G1レース一覧
G1_RACES = {
    'フェブラリーS': {'month': 2, 'week': 3, 'venue': '06'},  # 中山
    '高松宮記念': {'month': 3, 'week': 4, 'venue': '07'},  # 中京
    '桜花賞': {'month': 4, 'week': 2, 'venue': '09'},  # 阪神
    '皐月賞': {'month': 4, 'week': 3, 'venue': '06'},  # 中山
    '天皇賞(春)': {'month': 4, 'week': 4, 'venue': '08'},  # 京都
    'NHKマイルC': {'month': 5, 'week': 2, 'venue': '05'},  # 東京
    'ヴィクトリアM': {'month': 5, 'week': 3, 'venue': '05'},  # 東京
    'オークス': {'month': 5, 'week': 4, 'venue': '05'},  # 東京
    '日本ダービー': {'month': 5, 'week': 4, 'venue': '05'},  # 東京
    '安田記念': {'month': 6, 'week': 1, 'venue': '05'},  # 東京
    '宝塚記念': {'month': 6, 'week': 4, 'venue': '09'},  # 阪神
    'スプリンターズS': {'month': 9, 'week': 4, 'venue': '06'},  # 中山
    '秋華賞': {'month': 10, 'week': 3, 'venue': '08'},  # 京都
    '菊花賞': {'month': 10, 'week': 4, 'venue': '08'},  # 京都
    '天皇賞(秋)': {'month': 10, 'week': 4, 'venue': '05'},  # 東京
    'エリザベス女王杯': {'month': 11, 'week': 2, 'venue': '08'},  # 京都
    'マイルCS': {'month': 11, 'week': 3, 'venue': '08'},  # 京都
    'ジャパンC': {'month': 11, 'week': 4, 'venue': '05'},  # 東京
    'チャンピオンズC': {'month': 12, 'week': 1, 'venue': '06'},  # 中山
    '有馬記念': {'month': 12, 'week': 4, 'venue': '06'},  # 中山
}


def generate_keibalab_race_urls(start_year: int, end_year: int) -> list:
    """
    keibalab.jp用のG1レースURL候補を生成

    Args:
        start_year: 開始年
        end_year: 終了年

    Returns:
        (URL, レース名, 年)のリスト
    """
    urls = []

    for year in range(start_year, end_year + 1):
        for race_name, schedule in G1_RACES.items():
            month = schedule['month']
            week = schedule['week']
            venue = schedule['venue']

            # その月の第X週の推定日付
            start_day = (week - 1) * 7 + 1

            # 土日を含む2週間分を探索
            for day_offset in range(14):
                day = start_day + day_offset
                if day < 1 or day > 31:
                    continue

                # 11R, 12Rを試す
                for race_num in [11, 12]:
                    race_id = f"{year}{month:02d}{day:02d}{venue}{race_num:02d}"
                    url = f"https://www.keibalab.jp/db/race/{race_id}/"
                    urls.append((url, race_name, year))

    return urls


def collect_keibalab_g1_historical(start_year: int = 1995, end_year: int = 2024):
    """
    keibalab.jpから歴史的G1データを収集

    Args:
        start_year: 開始年
        end_year: 終了年
    """
    logging.info("=" * 60)
    logging.info("keibalab.jp 歴史的G1データ収集")
    logging.info("=" * 60)
    logging.info(f"期間: {start_year}-{end_year}年")
    logging.info(f"G1レース種類: {len(G1_RACES)}")
    logging.info(f"クロールディレイ: 12秒 (robots.txt遵守)")
    logging.info("")

    # URL生成
    race_urls_data = generate_keibalab_race_urls(start_year, end_year)

    logging.info(f"生成されたURL候補数: {len(race_urls_data)}")
    logging.info(f"推定所要時間: {len(race_urls_data) * 12 / 60:.1f}分 ({len(race_urls_data) * 12 / 3600:.1f}時間)")
    logging.info("")

    # 確認
    print("注意:")
    print(f"  - {len(race_urls_data)}個のレースURL候補を収集します")
    print(f"  - 推定所要時間: {len(race_urls_data) * 12 / 3600:.1f}時間")
    print(f"  - クロールディレイ: 12秒/レース")
    print(f"  - 非商業利用・研究目的のみ")
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

    for i, (url, race_name, year) in enumerate(race_urls_data, 1):
        logging.info(f"\n[{i}/{len(race_urls_data)}] {year}年 {race_name}")
        logging.info(f"  URL: {url}")

        # レースデータ収集
        df = scraper.scrape_race(url)

        if df is not None and len(df) > 0:
            # メタデータを追加
            df['expected_race_name'] = race_name
            df['expected_year'] = year

            all_results.append(df)
            successful_races += 1
            logging.info(f"  成功 ({len(df)}頭)")
        else:
            failed_races += 1
            logging.info(f"  失敗 (404またはエラー)")

        # 進捗表示 (20レースごと)
        if i % 20 == 0:
            logging.info(f"\n--- 進捗 ---")
            logging.info(f"成功: {successful_races}, 失敗: {failed_races}")
            logging.info(f"残り: {len(race_urls_data) - i} レース")
            logging.info(f"成功率: {successful_races / i * 100:.1f}%")
            logging.info(f"残り推定時間: {(len(race_urls_data) - i) * 12 / 60:.1f}分")

    # 結果を結合
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # 保存
        output_dir = Path('data/raw/keibalab')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'g1_historical_{start_year}_{end_year}_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        logging.info("\n" + "=" * 60)
        logging.info("データ収集完了!")
        logging.info("=" * 60)
        logging.info(f"成功: {successful_races}/{len(race_urls_data)} レース")
        logging.info(f"失敗: {failed_races}/{len(race_urls_data)} レース")
        logging.info(f"成功率: {successful_races / len(race_urls_data) * 100:.1f}%")
        logging.info(f"総データ数: {len(combined_df)}頭")
        logging.info(f"ユニークレース数: {combined_df['race_id'].nunique() if 'race_id' in combined_df.columns else 'N/A'}個")
        logging.info(f"保存先: {output_file}")
        logging.info("=" * 60)

        # 統計表示
        if 'race_date' in combined_df.columns:
            logging.info(f"\n期間: {combined_df['race_date'].min()} - {combined_df['race_date'].max()}")

        if 'venue_name' in combined_df.columns:
            logging.info("\n競馬場別:")
            print(combined_df['venue_name'].value_counts())

        if 'track_type' in combined_df.columns:
            logging.info("\n馬場タイプ別:")
            print(combined_df['track_type'].value_counts())

        if 'expected_race_name' in combined_df.columns:
            logging.info("\nG1レース別収集数:")
            race_counts = combined_df.groupby('expected_race_name').size().sort_values(ascending=False)
            print(race_counts)

        return combined_df
    else:
        logging.error("データ収集に失敗しました（全て404エラー）")
        return None


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("keibalab.jp 歴史的G1データ収集スクリプト")
    logging.info("=" * 60)
    logging.info("")
    logging.info("目的:")
    logging.info("  - keibalab.jpから過去のG1データを収集")
    logging.info("  - netkeibaに無い追加情報の取得")
    logging.info("  - レースペース、血統指数などの補完")
    logging.info("")
    logging.info("注意:")
    logging.info("  - 非商業利用・研究目的のみ")
    logging.info("  - robots.txt推奨の12秒ディレイを遵守")
    logging.info("")
    logging.info("=" * 60)
    logging.info("")

    # データ収集
    collect_keibalab_g1_historical(start_year=1995, end_year=2024)


if __name__ == "__main__":
    main()

"""
keibalab.jp 逆順G1データ収集スクリプト

2024年から1995年へ逆順で収集することで効率化:
- 新しい年のデータから収集開始
- データが存在しない年代で早期停止
- 12秒のクロールディレイ遵守
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


def generate_year_race_urls(year: int) -> list:
    """
    指定年のG1レースURL候補を生成

    Args:
        year: 年

    Returns:
        (URL, レース名, 年)のリスト
    """
    urls = []

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


def collect_keibalab_g1_reverse(start_year: int = 2024, end_year: int = 1995,
                                 min_success_rate: float = 0.05):
    """
    keibalab.jpから逆順でG1データを収集

    Args:
        start_year: 開始年 (新しい年)
        end_year: 終了年 (古い年)
        min_success_rate: 年ごとの最小成功率 (これを下回ると停止)
    """
    logging.info("=" * 60)
    logging.info("keibalab.jp 逆順G1データ収集")
    logging.info("=" * 60)
    logging.info(f"期間: {start_year}年 → {end_year}年 (新→旧)")
    logging.info(f"G1レース種類: {len(G1_RACES)}")
    logging.info(f"クロールディレイ: 12秒 (robots.txt遵守)")
    logging.info(f"最小成功率: {min_success_rate*100:.1f}% (これを下回ると停止)")
    logging.info("")

    # 確認
    total_years = start_year - end_year + 1
    estimated_urls_per_year = len(G1_RACES) * 14 * 2  # 20レース * 14日 * 2レース番号

    print("注意:")
    print(f"  - {total_years}年分のデータを新→旧の順で収集")
    print(f"  - 各年約{estimated_urls_per_year}URL候補")
    print(f"  - 推定所要時間: 年によって変動")
    print(f"  - クロールディレイ: 12秒/レース")
    print(f"  - 年ごとの成功率が{min_success_rate*100:.1f}%未満で早期停止")
    print("")
    response = input("収集を開始しますか？ (y/n): ")

    if response.lower() != 'y':
        logging.info("収集をキャンセルしました")
        return

    # スクレイパー初期化 (12秒ディレイ)
    scraper = KeibalabScraper(delay=12.0, max_retries=3)

    # 年ごとにデータ収集
    all_results = []
    total_successful_races = 0
    total_failed_races = 0

    for year in range(start_year, end_year - 1, -1):  # 逆順
        logging.info("\n" + "=" * 60)
        logging.info(f"{year}年のG1データ収集")
        logging.info("=" * 60)

        # 年のURL候補を生成
        race_urls_data = generate_year_race_urls(year)

        logging.info(f"URL候補数: {len(race_urls_data)}")
        logging.info(f"推定所要時間: {len(race_urls_data) * 12 / 60:.1f}分")
        logging.info("")

        # 年ごとの統計
        year_successful = 0
        year_failed = 0
        year_results = []

        for i, (url, race_name, yr) in enumerate(race_urls_data, 1):
            logging.info(f"\n[{i}/{len(race_urls_data)}] {year}年 {race_name}")

            # レースデータ収集
            df = scraper.scrape_race(url)

            if df is not None and len(df) > 0:
                # メタデータを追加
                df['expected_race_name'] = race_name
                df['expected_year'] = yr

                year_results.append(df)
                year_successful += 1
                total_successful_races += 1
                logging.info(f"  成功 ({len(df)}頭)")
            else:
                year_failed += 1
                total_failed_races += 1
                logging.info(f"  失敗 (404またはエラー)")

            # 進捗表示 (20レースごと)
            if i % 20 == 0:
                current_success_rate = year_successful / i if i > 0 else 0
                logging.info(f"\n--- {year}年 進捗 ---")
                logging.info(f"成功: {year_successful}, 失敗: {year_failed}")
                logging.info(f"残り: {len(race_urls_data) - i} URL")
                logging.info(f"成功率: {current_success_rate * 100:.1f}%")

        # 年ごとの結果を追加
        if year_results:
            all_results.extend(year_results)

        # 年ごとの成功率を計算
        year_success_rate = year_successful / len(race_urls_data)

        logging.info(f"\n--- {year}年 完了 ---")
        logging.info(f"成功: {year_successful}/{len(race_urls_data)} URL")
        logging.info(f"成功率: {year_success_rate * 100:.1f}%")
        logging.info(f"収集レース数: {len(year_results)}")

        # 早期停止判定
        if year_success_rate < min_success_rate:
            logging.warning(f"\n成功率が{min_success_rate*100:.1f}%未満のため、{year}年で収集を停止します")
            logging.warning(f"これより古い年({year-1}年以前)はデータが存在しない可能性が高いです")
            break

    # 結果を結合
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # 保存
        output_dir = Path('data/raw/keibalab')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        actual_start_year = combined_df['expected_year'].max()
        actual_end_year = combined_df['expected_year'].min()
        output_file = output_dir / f'g1_reverse_{int(actual_start_year)}_{int(actual_end_year)}_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        logging.info("\n" + "=" * 60)
        logging.info("データ収集完了!")
        logging.info("=" * 60)
        logging.info(f"成功: {total_successful_races}レース")
        logging.info(f"失敗: {total_failed_races}レース")
        logging.info(f"総成功率: {total_successful_races / (total_successful_races + total_failed_races) * 100:.1f}%")
        logging.info(f"総データ数: {len(combined_df)}頭")
        logging.info(f"ユニークレース数: {combined_df['race_id'].nunique() if 'race_id' in combined_df.columns else 'N/A'}個")
        logging.info(f"収集期間: {int(actual_end_year)}-{int(actual_start_year)}年")
        logging.info(f"保存先: {output_file}")
        logging.info("=" * 60)

        # 統計表示
        if 'expected_race_name' in combined_df.columns:
            logging.info("\nG1レース別収集数:")
            race_counts = combined_df.groupby('expected_race_name').size().sort_values(ascending=False)
            print(race_counts)

        if 'expected_year' in combined_df.columns:
            logging.info("\n年別収集数:")
            year_counts = combined_df.groupby('expected_year').size().sort_values(ascending=False)
            print(year_counts)

        return combined_df
    else:
        logging.error("データ収集に失敗しました（全て404エラー）")
        return None


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("keibalab.jp 逆順G1データ収集スクリプト")
    logging.info("=" * 60)
    logging.info("")
    logging.info("改善点:")
    logging.info("  - 2024年から逆順で収集 (新→旧)")
    logging.info("  - 成功率が低い年で早期停止")
    logging.info("  - 効率的なデータ収集")
    logging.info("")
    logging.info("注意:")
    logging.info("  - 非商業利用・研究目的のみ")
    logging.info("  - robots.txt推奨の12秒ディレイを遵守")
    logging.info("")
    logging.info("=" * 60)
    logging.info("")

    # データ収集
    collect_keibalab_g1_reverse(start_year=2024, end_year=1995, min_success_rate=0.05)


if __name__ == "__main__":
    main()

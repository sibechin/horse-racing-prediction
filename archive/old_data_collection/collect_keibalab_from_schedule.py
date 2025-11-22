"""
スケジュールベースkeibalab.jpデータ収集

正確なG1開催日スケジュールを使用した効率的なデータ収集
"""
import sys
from pathlib import Path
sys.path.append('.')

from src.data_collection.keibalab_scraper import KeibalabScraper
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_schedule(year: int) -> list:
    """
    スケジュールファイルからURL候補を読み込み

    Args:
        year: 年

    Returns:
        (race_id, date, name, url)のリスト
    """
    schedule_file = Path(f'data/schedules/keibalab_urls_{year}.txt')

    if not schedule_file.exists():
        logging.error(f"スケジュールファイルが見つかりません: {schedule_file}")
        return []

    races = []
    with open(schedule_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                race_id, date, name, url = parts
                races.append((race_id, date, name, url))

    logging.info(f"{year}年のスケジュールを読み込みました: {len(races)}URL候補")
    return races


def collect_keibalab_from_schedule(years: list, delay: float = 12.0):
    """
    スケジュールに基づいてkeibalab.jpからデータを収集

    Args:
        years: 収集する年のリスト
        delay: クロールディレイ（秒）
    """
    logging.info("=" * 60)
    logging.info("スケジュールベースkeibalab.jpデータ収集")
    logging.info("=" * 60)
    logging.info(f"対象年: {', '.join(map(str, years))}")
    logging.info(f"クロールディレイ: {delay}秒 (robots.txt遵守)")
    logging.info("")

    # 全年のスケジュールを読み込み
    all_races = []
    for year in years:
        races = load_schedule(year)
        all_races.extend(races)

    if not all_races:
        logging.error("スケジュールが見つかりませんでした")
        return

    logging.info(f"総URL候補数: {len(all_races)}")
    logging.info(f"推定所要時間: {len(all_races) * delay / 60:.1f}分")
    logging.info("")

    # 確認
    print("注意:")
    print(f"  - {len(all_races)}個のURL候補を収集します")
    print(f"  - 推定所要時間: {len(all_races) * delay / 60:.1f}分")
    print(f"  - クロールディレイ: {delay}秒/レース")
    print(f"  - 非商業利用・研究目的のみ")
    print("")
    response = input("収集を開始しますか？ (y/n): ")

    if response.lower() != 'y':
        logging.info("収集をキャンセルしました")
        return

    # スクレイパー初期化
    scraper = KeibalabScraper(delay=delay, max_retries=3)

    # データ収集
    all_results = []
    successful_races = 0
    failed_races = 0
    race_details = []

    for i, (race_id, date, name, url) in enumerate(all_races, 1):
        logging.info(f"\n[{i}/{len(all_races)}] {name} ({date})")
        logging.info(f"  レースID: {race_id}")

        # レースデータ収集
        df = scraper.scrape_race(url)

        if df is not None and len(df) > 0:
            # メタデータを追加
            df['expected_race_name'] = name
            df['expected_date'] = date
            df['race_id_keibalab'] = race_id

            all_results.append(df)
            successful_races += 1

            race_details.append({
                'race_name': name,
                'race_id': race_id,
                'date': date,
                'horses': len(df)
            })

            logging.info(f"  成功 ({len(df)}頭)")
        else:
            failed_races += 1
            logging.info(f"  失敗 (404またはエラー)")

        # 進捗表示 (10レースごと)
        if i % 10 == 0:
            success_rate = successful_races / i * 100
            logging.info(f"\n--- 進捗 ---")
            logging.info(f"成功: {successful_races}, 失敗: {failed_races}")
            logging.info(f"残り: {len(all_races) - i} レース")
            logging.info(f"成功率: {success_rate:.1f}%")
            logging.info(f"残り推定時間: {(len(all_races) - i) * delay / 60:.1f}分")

    # 結果を結合
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # 保存
        output_dir = Path('data/raw/keibalab')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        years_str = '_'.join(map(str, years))
        output_file = output_dir / f'g1_schedule_based_{years_str}_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        logging.info("\n" + "=" * 60)
        logging.info("データ収集完了!")
        logging.info("=" * 60)
        logging.info(f"成功: {successful_races}/{len(all_races)} レース")
        logging.info(f"失敗: {failed_races}/{len(all_races)} レース")
        logging.info(f"成功率: {successful_races / len(all_races) * 100:.1f}%")
        logging.info(f"総データ数: {len(combined_df)}頭")
        logging.info(f"ユニークレース数: {combined_df['race_id_keibalab'].nunique()}個")
        logging.info(f"保存先: {output_file}")
        logging.info("=" * 60)

        # 収集レース一覧
        if race_details:
            logging.info("\n収集されたレース:")
            for detail in race_details:
                logging.info(f"  {detail['date']} - {detail['race_name']} ({detail['horses']}頭)")

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
    logging.info("スケジュールベースkeibalab.jpデータ収集スクリプト")
    logging.info("=" * 60)
    logging.info("")
    logging.info("特徴:")
    logging.info("  - 正確なG1開催日に基づいた効率的な収集")
    logging.info("  - 42URL候補 (2023-2024年, 21レース×2)")
    logging.info("  - 推定所要時間: 約8-9分/年")
    logging.info("  - 成功率: 95%以上期待")
    logging.info("")
    logging.info("注意:")
    logging.info("  - 非商業利用・研究目的のみ")
    logging.info("  - robots.txt推奨の12秒ディレイを遵守")
    logging.info("")
    logging.info("=" * 60)
    logging.info("")

    # 2023-2024年のデータを収集
    collect_keibalab_from_schedule(years=[2024, 2023], delay=12.0)


if __name__ == "__main__":
    main()

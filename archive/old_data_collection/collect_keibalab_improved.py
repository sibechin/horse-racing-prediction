"""
keibalab.jp 改良版スクレイパーで効率的データ収集

改良版スクレイパー（調教師、通過順、上り、指数を含む）を使用して
2024年から逆順でG1データを効率的に収集
"""
import sys
from pathlib import Path
sys.path.append('.')

from src.data_collection.keibalab_scraper import KeibalabScraper
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_schedule_csv(year: int) -> list:
    """
    CSVスケジュールファイルからURL候補を読み込み

    Args:
        year: 年

    Returns:
        (race_id, date, name, url)のリスト
    """
    schedule_file = Path(f'data/schedules/keibalab_urls_{year}.csv')

    if not schedule_file.exists():
        logging.warning(f"スケジュールファイルが見つかりません: {schedule_file}")
        return []

    try:
        df = pd.read_csv(schedule_file, encoding='utf-8-sig')
        races = []
        for _, row in df.iterrows():
            races.append((row['race_id'], row['date'], row['race_name'], row['url']))

        logging.info(f"{year}年のスケジュールを読み込みました: {len(races)}URL候補")
        return races
    except Exception as e:
        logging.error(f"スケジュールファイルの読み込みエラー: {e}")
        return []


def collect_year_data(year: int, scraper: KeibalabScraper) -> tuple:
    """
    特定年のデータを収集

    Args:
        year: 年
        scraper: KeibalabScraperインスタンス（改良版）

    Returns:
        (成功データのDataFrame, 成功数, 失敗数)
    """
    logging.info(f"\n{'=' * 60}")
    logging.info(f"{year}年のG1データ収集（改良版スクレイパー使用）")
    logging.info(f"{'=' * 60}")

    # スケジュール読み込み
    races = load_schedule_csv(year)

    if not races:
        return None, 0, 0

    logging.info(f"URL候補数: {len(races)}")
    logging.info(f"推定所要時間: {len(races) * 2 / 60:.1f}分")
    logging.info("")

    # データ収集
    year_results = []
    successful = 0
    failed = 0

    for i, (race_id, date, name, url) in enumerate(races, 1):
        if i % 50 == 0:  # 進捗表示
            success_rate = successful / i * 100 if i > 0 else 0
            logging.info(f"\n[{i}/{len(races)}] 進捗: 成功率 {success_rate:.1f}%")

        # レースデータ収集（改良版スクレイパー）
        df = scraper.scrape_race(url)

        if df is not None and len(df) > 0:
            # メタデータを追加
            df['expected_race_name'] = name
            df['expected_date'] = date
            df['expected_year'] = year
            df['race_id_keibalab'] = race_id

            year_results.append(df)
            successful += 1
        else:
            failed += 1

    # 年の結果を結合
    if year_results:
        year_df = pd.concat(year_results, ignore_index=True)
        return year_df, successful, failed
    else:
        return None, successful, failed


def collect_improved_data(start_year: int = 2024, end_year: int = 2020,
                          min_success_rate: float = 0.03, delay: float = 2.0):
    """
    改良版スクレイパーで効率的にデータを収集

    Args:
        start_year: 開始年 (新しい年)
        end_year: 終了年 (古い年)
        min_success_rate: 最小成功率 (これを下回ると停止)
        delay: クロールディレイ（秒）- デフォルト2秒
    """
    logging.info("=" * 60)
    logging.info("keibalab.jp 改良版スクレイパーでデータ収集")
    logging.info("=" * 60)
    logging.info(f"期間: {start_year}年 → {end_year}年 (新→旧)")
    logging.info(f"クロールディレイ: {delay}秒")
    logging.info(f"早期停止閾値: 成功率{min_success_rate*100:.1f}%未満")
    logging.info("")
    logging.info("改良版スクレイパーの新機能:")
    logging.info("  - 調教師データ（trainer）")
    logging.info("  - 通過順（passing_order）")
    logging.info("  - 上がり3F（last_3f）")
    logging.info("  - α、β、Ω指数（indices）")
    logging.info("")

    # 確認
    print("収集戦略:")
    print(f"  - {start_year}年から{end_year}年へ逆順で収集")
    print(f"  - 各年の成功率が{min_success_rate*100:.1f}%未満で自動停止")
    print(f"  - クロールディレイ: {delay}秒/レース（高速収集）")
    print(f"  - 特徴量: 19個（調教師、通過順、上り、指数を含む）")
    print("")
    response = input("収集を開始しますか？ (y/n): ")

    if response.lower() != 'y':
        logging.info("収集をキャンセルしました")
        return

    # スクレイパー初期化（改良版）
    scraper = KeibalabScraper(delay=delay, max_retries=3)

    # 年ごとにデータ収集
    all_results = []
    collection_summary = []

    for year in range(start_year, end_year - 1, -1):  # 逆順
        # 年のデータ収集
        year_df, successful, failed = collect_year_data(year, scraper)

        total_attempts = successful + failed
        success_rate = successful / total_attempts if total_attempts > 0 else 0

        # 結果を記録
        collection_summary.append({
            'year': year,
            'successful': successful,
            'failed': failed,
            'total_attempts': total_attempts,
            'success_rate': success_rate * 100,
            'total_horses': len(year_df) if year_df is not None else 0
        })

        logging.info(f"\n--- {year}年 完了 ---")
        logging.info(f"成功: {successful}/{total_attempts} URL ({success_rate * 100:.1f}%)")
        if year_df is not None:
            logging.info(f"収集データ数: {len(year_df)}頭")
            logging.info(f"特徴量数: {len(year_df.columns)}個")
            all_results.append(year_df)

        # 早期停止判定
        if success_rate < min_success_rate and year < start_year:
            logging.warning(f"\n成功率が{min_success_rate*100:.1f}%未満のため、{year}年で収集を停止します")
            logging.warning(f"これより古い年({year-1}年以前)はデータが存在しない可能性が高いです")
            break

    # 結果を統合
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # 保存
        output_dir = Path('data/raw/keibalab')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        actual_start_year = int(combined_df['expected_year'].max())
        actual_end_year = int(combined_df['expected_year'].min())
        output_file = output_dir / f'g1_improved_{actual_start_year}_{actual_end_year}_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        # サマリーも保存
        summary_df = pd.DataFrame(collection_summary)
        summary_file = output_dir / f'collection_summary_improved_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')

        logging.info("\n" + "=" * 60)
        logging.info("データ収集完了!")
        logging.info("=" * 60)
        logging.info(f"収集期間: {actual_end_year}-{actual_start_year}年")
        logging.info(f"総データ数: {len(combined_df)}頭")
        logging.info(f"特徴量数: {len(combined_df.columns)}個")
        logging.info(f"ユニークレース数: {combined_df['race_id_keibalab'].nunique()}個")
        logging.info(f"保存先: {output_file}")
        logging.info(f"サマリー: {summary_file}")
        logging.info("=" * 60)

        # 年別統計
        logging.info("\n年別収集統計:")
        print(summary_df.to_string(index=False))

        # 新機能の確認
        logging.info("\n新規追加フィールドのサンプル（1着馬）:")
        if len(combined_df) > 0:
            sample = combined_df.iloc[0]
            logging.info(f"  調教師: {sample.get('trainer', 'N/A')}")
            logging.info(f"  通過順: {sample.get('passing_order', 'N/A')}")
            logging.info(f"  上り3F: {sample.get('last_3f', 'N/A')}")
            logging.info(f"  指数: {sample.get('indices', 'N/A')}")

        # レース別統計
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
    logging.info("keibalab.jp 改良版スクレイパーデータ収集スクリプト")
    logging.info("=" * 60)
    logging.info("")
    logging.info("特徴:")
    logging.info("  - 改良版スクレイパー使用（19特徴量）")
    logging.info("  - 調教師、通過順、上り、指数データを含む")
    logging.info("  - 2024年から逆順で収集 (新→旧)")
    logging.info("  - 成功率が3%未満で自動停止")
    logging.info("  - 高速収集（2秒ディレイ）")
    logging.info("")
    logging.info("推定:")
    logging.info("  - 2024年: 約22分（660URL候補、成功率12%想定）")
    logging.info("  - 多年収集: 成功率により自動調整")
    logging.info("")
    logging.info("=" * 60)
    logging.info("")

    # データ収集（2024年から2000年まで - 遡れる限り遡る）
    collect_improved_data(start_year=2024, end_year=2000, min_success_rate=0.03, delay=2.0)


if __name__ == "__main__":
    main()

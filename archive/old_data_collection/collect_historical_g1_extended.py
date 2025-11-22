"""
歴史的G1データ拡張収集スクリプト

目的:
- 過去30年分のG1レースデータを収集（1995-2024年）
- 現在の102レースから約480レースに拡大
- PyTorch深層学習モデルの性能向上（Top3: 80%→90%, Top1: 20%→30-40%）

収集対象:
- 16主要G1レース × 30年 = 約480レース
"""
import sys
from pathlib import Path
sys.path.append('.')

from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 主要G1レース一覧（netkeibaのレースID）
G1_RACES = {
    # フェブラリーステークス
    'february_stakes': {
        'name': 'フェブラリーステークス',
        'venue': '06',  # 東京 (後に中山、大井)
        'race_num': '11',
        'month': '02',
        'week': 3,  # 第3週頃
    },
    # 高松宮記念
    'takamatsunomiya_kinen': {
        'name': '高松宮記念',
        'venue': '07',  # 中京
        'race_num': '11',
        'month': '03',
        'week': 4,
    },
    # 桜花賞
    'oka_sho': {
        'name': '桜花賞',
        'venue': '09',  # 阪神
        'race_num': '11',
        'month': '04',
        'week': 2,
    },
    # 皐月賞
    'satsuki_sho': {
        'name': '皐月賞',
        'venue': '06',  # 中山
        'race_num': '11',
        'month': '04',
        'week': 3,
    },
    # 天皇賞（春）
    'tennosho_spring': {
        'name': '天皇賞（春）',
        'venue': '08',  # 京都
        'race_num': '11',
        'month': '04',
        'week': 4,
    },
    # NHKマイルカップ
    'nhk_mile_cup': {
        'name': 'NHKマイルカップ',
        'venue': '05',  # 東京
        'race_num': '11',
        'month': '05',
        'week': 2,
    },
    # ヴィクトリアマイル
    'victoria_mile': {
        'name': 'ヴィクトリアマイル',
        'venue': '05',  # 東京
        'race_num': '11',
        'month': '05',
        'week': 3,
    },
    # オークス
    'oaks': {
        'name': 'オークス',
        'venue': '05',  # 東京
        'race_num': '11',
        'month': '05',
        'week': 4,
    },
    # 日本ダービー
    'japanese_derby': {
        'name': '日本ダービー',
        'venue': '05',  # 東京
        'race_num': '11',
        'month': '05',
        'week': 4,
    },
    # 安田記念
    'yasuda_kinen': {
        'name': '安田記念',
        'venue': '05',  # 東京
        'race_num': '11',
        'month': '06',
        'week': 1,
    },
    # 宝塚記念
    'takarazuka_kinen': {
        'name': '宝塚記念',
        'venue': '09',  # 阪神
        'race_num': '11',
        'month': '06',
        'week': 4,
    },
    # スプリンターズステークス
    'sprinters_stakes': {
        'name': 'スプリンターズステークス',
        'venue': '06',  # 中山
        'race_num': '11',
        'month': '09',
        'week': 4,
    },
    # 秋華賞
    'shuka_sho': {
        'name': '秋華賞',
        'venue': '08',  # 京都
        'race_num': '11',
        'month': '10',
        'week': 3,
    },
    # 菊花賞
    'kikka_sho': {
        'name': '菊花賞',
        'venue': '08',  # 京都
        'race_num': '11',
        'month': '10',
        'week': 4,
    },
    # 天皇賞（秋）
    'tennosho_autumn': {
        'name': '天皇賞（秋）',
        'venue': '05',  # 東京
        'race_num': '11',
        'month': '10',
        'week': 4,
    },
    # エリザベス女王杯
    'elizabeth_jo_hai': {
        'name': 'エリザベス女王杯',
        'venue': '08',  # 京都
        'race_num': '11',
        'month': '11',
        'week': 2,
    },
    # マイルチャンピオンシップ
    'mile_championship': {
        'name': 'マイルチャンピオンシップ',
        'venue': '08',  # 京都
        'race_num': '11',
        'month': '11',
        'week': 3,
    },
    # ジャパンカップ
    'japan_cup': {
        'name': 'ジャパンカップ',
        'venue': '05',  # 東京
        'race_num': '11',
        'month': '11',
        'week': 4,
    },
    # チャンピオンズカップ
    'champions_cup': {
        'name': 'チャンピオンズカップ',
        'venue': '06',  # 中京（後に中山）
        'race_num': '11',
        'month': '12',
        'week': 1,
    },
    # 有馬記念
    'arima_kinen': {
        'name': '有馬記念',
        'venue': '06',  # 中山
        'race_num': '12',
        'month': '12',
        'week': 4,
    },
}


def generate_race_id_candidates(year: int, race_info: dict) -> list:
    """
    指定年のG1レースIDの候補を生成

    Args:
        year: 年
        race_info: G1レース情報

    Returns:
        レースIDの候補リスト
    """
    month = race_info['month']  # 文字列 '02', '03' etc.
    venue = race_info['venue']
    race_num = race_info['race_num']
    week = race_info['week']

    # その月の第X週の日曜日を推定（日曜日 = 開催日として推定）
    # 実際には土曜開催のレースもあるため、土日両方を候補に
    candidates = []

    # 第X週の土日を推定
    start_day = (week - 1) * 7 + 1
    for day_offset in range(14):  # 2週間分探索
        day = start_day + day_offset
        if day > 31:
            break

        race_id = f"{year}{month}{day:02d}{venue}{race_num}"
        candidates.append(race_id)

    return candidates


def collect_historical_g1_data(start_year: int = 1995, end_year: int = 2024):
    """
    過去30年分のG1データを収集

    Args:
        start_year: 開始年
        end_year: 終了年
    """
    logging.info("=" * 60)
    logging.info("歴史的G1データ拡張収集")
    logging.info("=" * 60)
    logging.info(f"期間: {start_year}-{end_year}年 ({end_year - start_year + 1}年間)")
    logging.info(f"G1レース数: {len(G1_RACES)}")
    logging.info(f"推定総レース数: {len(G1_RACES) * (end_year - start_year + 1)}")
    logging.info("=" * 60)
    logging.info("")

    # スクレイパー初期化
    scraper = NetkeibaScraperComplete(delay=2.0, max_retries=3)

    # 全レースID候補を生成
    all_race_id_candidates = []
    race_metadata = {}  # レースID → レース情報のマッピング

    for year in range(start_year, end_year + 1):
        for race_key, race_info in G1_RACES.items():
            candidates = generate_race_id_candidates(year, race_info)
            for race_id in candidates:
                all_race_id_candidates.append(race_id)
                race_metadata[race_id] = {
                    'year': year,
                    'race_name': race_info['name'],
                    'race_key': race_key
                }

    logging.info(f"生成されたレースID候補数: {len(all_race_id_candidates)}")
    logging.info(f"推定所要時間: {len(all_race_id_candidates) * 2 / 60:.1f}分")
    logging.info("")

    # 確認
    print("注意:")
    print(f"  - {len(all_race_id_candidates)}個のレースID候補を収集します")
    print(f"  - 推定所要時間: {len(all_race_id_candidates) * 2 / 60:.1f}分")
    print(f"  - 実際に存在するレースのみ収集されます（404エラーはスキップ）")
    print("")
    response = input("収集を開始しますか？ (y/n): ")

    if response.lower() != 'y':
        logging.info("収集をキャンセルしました")
        return

    # データ収集
    all_results = []
    successful_races = 0
    failed_races = 0

    for i, race_id in enumerate(all_race_id_candidates, 1):
        logging.info(f"\n[{i}/{len(all_race_id_candidates)}] レースID: {race_id}")

        if race_id in race_metadata:
            metadata = race_metadata[race_id]
            logging.info(f"  {metadata['year']}年 {metadata['race_name']}")

        # レースデータ収集
        df = scraper.scrape_race_result(race_id)

        if df is not None and len(df) > 0:
            # メタデータを追加
            if race_id in race_metadata:
                df['expected_race_name'] = metadata['race_name']
                df['expected_year'] = metadata['year']

            all_results.append(df)
            successful_races += 1
            logging.info(f"  成功 ({len(df)}頭)")
        else:
            failed_races += 1
            logging.info(f"  失敗（404またはエラー）")

        # 進捗表示
        if i % 50 == 0:
            logging.info(f"\n--- 進捗 ---")
            logging.info(f"成功: {successful_races}, 失敗: {failed_races}")
            logging.info(f"残り: {len(all_race_id_candidates) - i} レース")
            logging.info(f"成功率: {successful_races / i * 100:.1f}%")

    # 結果を結合
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # 保存
        output_dir = Path('data/raw')
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'g1_historical_{start_year}_{end_year}_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        logging.info("\n" + "=" * 60)
        logging.info("データ収集完了!")
        logging.info("=" * 60)
        logging.info(f"成功: {successful_races}/{len(all_race_id_candidates)} レース")
        logging.info(f"失敗: {failed_races}/{len(all_race_id_candidates)} レース")
        logging.info(f"成功率: {successful_races / len(all_race_id_candidates) * 100:.1f}%")
        logging.info(f"総データ数: {len(combined_df)}頭")
        logging.info(f"ユニークレース数: {combined_df['race_id'].nunique()}")
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

        # 現在のデータとの比較
        logging.info("\n" + "=" * 60)
        logging.info("データ拡張の効果")
        logging.info("=" * 60)
        logging.info(f"既存データ: 102レース (~1,909頭) 2016-2024年")
        logging.info(f"新規データ: {combined_df['race_id'].nunique()}レース ({len(combined_df)}頭) {start_year}-{end_year}年")
        logging.info(f"拡張倍率: {combined_df['race_id'].nunique() / 102:.1f}倍")
        logging.info("=" * 60)

        return combined_df
    else:
        logging.error("データ収集に失敗しました")
        return None


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("歴史的G1データ拡張収集スクリプト")
    logging.info("=" * 60)
    logging.info("")
    logging.info("目的:")
    logging.info("  - 過去30年分のG1レースデータを収集")
    logging.info("  - 深層学習モデルの性能向上")
    logging.info("  - Top3的中率: 80% → 90%")
    logging.info("  - Top1的中率: 20% → 30-40%")
    logging.info("")
    logging.info("=" * 60)
    logging.info("")

    # データ収集
    collect_historical_g1_data(start_year=1995, end_year=2024)


if __name__ == "__main__":
    main()

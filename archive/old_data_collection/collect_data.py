"""
データ収集実行スクリプト
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_sample_data():
    """サンプルデータの収集"""
    logger.info("=" * 60)
    logger.info("競馬データ収集開始")
    logger.info("=" * 60)

    # スクレイパー初期化
    scraper = NetkeibaScraperComplete(delay=2.0, max_retries=3)

    # テスト: 1レースを取得
    logger.info("\n[ステップ1] テスト: 1レースを取得")
    test_race_id = "202406030811"  # 2024年皐月賞（サンプル）

    result = scraper.scrape_race_result(test_race_id)

    if result is not None:
        print("\n✅ テスト成功！")
        print("\n=== レース結果プレビュー ===")
        print(result[['race_name', 'horse_name', 'jockey_name',
                     'finish_position', 'time', 'popularity']].head(10).to_string())

        # 保存
        result.to_csv("data/raw/test_race.csv", index=False, encoding='utf-8-sig')
        print(f"\n保存先: data/raw/test_race.csv")
        print(f"取得データ数: {len(result)}頭")
    else:
        print("\n❌ テスト失敗")
        return

    # 本番: 複数レースの収集
    logger.info("\n[ステップ2] 複数レースの収集")

    # 2024年のサンプルレースID（実際のレースIDに置き換えてください）
    race_ids_2024 = [
        # 東京競馬場（05）のサンプル
        "202401070511",  # 1月第1週11R
        "202401140511",  # 1月第2週11R
        "202402040511",  # 2月第1週11R

        # 中山競馬場（06）のサンプル
        "202401070611",  # 1月第1週11R
        "202401140611",  # 1月第2週11R
    ]

    print(f"\n収集するレース数: {len(race_ids_2024)}")
    print("※ これはサンプルです。実際のレースIDに置き換えてください。")

    proceed = input("\n続行しますか? (y/n): ")

    if proceed.lower() == 'y':
        results = scraper.scrape_multiple_races(
            race_ids_2024,
            output_dir="data/raw"
        )

        if len(results) > 0:
            print(f"\n✅ 収集完了！")
            print(f"  総レース数: {results['race_id'].nunique()}")
            print(f"  総データ数: {len(results)}頭")
            print(f"  期間: {results['race_date'].min()} 〜 {results['race_date'].max()}")
        else:
            print("\n❌ データ収集失敗")
    else:
        print("\nキャンセルしました")


def collect_by_year(year: int, max_races: int = 100):
    """
    指定年のデータを収集

    Args:
        year: 年
        max_races: 最大収集レース数
    """
    logger.info(f"=== {year}年のデータ収集 ===")

    scraper = NetkeibaScraperComplete(delay=2.0)

    # レースIDのリストを作成
    # ※ 実際にはカレンダーから取得するか、手動で指定する必要があります
    # ここでは簡略化のため、サンプルIDを使用

    race_ids = []

    # 競馬場コード: 東京(05), 中山(06), 阪神(09), 京都(08)
    venues = ['05', '06', '08', '09']

    # 各月の主要開催日（簡略版）
    sample_dates = [
        '0107', '0114', '0121', '0128',  # 1月
        '0204', '0211', '0218', '0225',  # 2月
        '0303', '0310', '0317', '0324',  # 3月
        '0407', '0414', '0421', '0428',  # 4月
        '0505', '0512', '0519', '0526',  # 5月
        '0602', '0609', '0616', '0623',  # 6月
        '0707', '0714', '0721', '0728',  # 7月
        '0804', '0811', '0818', '0825',  # 8月
        '0902', '0909', '0916', '0923',  # 9月
        '1007', '1014', '1021', '1028',  # 10月
        '1104', '1111', '1118', '1125',  # 11月
        '1202', '1209', '1216', '1223',  # 12月
    ]

    for venue in venues:
        for date in sample_dates:
            # 主要レース（11R, 12R）のみ
            for race_num in [11, 12]:
                race_id = f"{year}{date}{venue}{race_num:02d}"
                race_ids.append(race_id)

                if len(race_ids) >= max_races:
                    break
            if len(race_ids) >= max_races:
                break
        if len(race_ids) >= max_races:
            break

    print(f"\n収集予定レース数: {len(race_ids)}")
    print(f"推定所要時間: {len(race_ids) * 2 / 60:.1f}分")

    proceed = input("\n実行しますか? (y/n): ")

    if proceed.lower() == 'y':
        results = scraper.scrape_multiple_races(race_ids, output_dir="data/raw")

        if len(results) > 0:
            print(f"\n✅ 収集完了！")
            print(f"\n=== 統計情報 ===")
            print(f"総レース数: {results['race_id'].nunique()}")
            print(f"総データ数: {len(results)}頭")
            print(f"競馬場別:")
            print(results['venue_name'].value_counts())
            print(f"\n馬場別:")
            print(results['track_type'].value_counts())
    else:
        print("\nキャンセルしました")


if __name__ == "__main__":
    print("""
    ============================================
    競馬データ収集ツール (netkeiba.com)
    ============================================

    【重要な注意事項】
    1. netkeiba.comの利用規約を確認してください
    2. 過度なアクセスはサーバーに負荷をかけます
    3. リクエスト間隔は2秒以上に設定しています
    4. 商用利用は禁止されている可能性があります

    """)

    print("収集方法を選択してください:")
    print("1. サンプルデータを収集（テスト用）")
    print("2. 指定年のデータを収集")
    print("3. カスタム収集")

    choice = input("\n選択 (1/2/3): ")

    if choice == '1':
        collect_sample_data()

    elif choice == '2':
        year = input("年を入力 (例: 2024): ")
        max_races = input("最大レース数 (例: 100): ")

        try:
            collect_by_year(int(year), int(max_races))
        except ValueError:
            print("❌ 入力が不正です")

    elif choice == '3':
        print("\nカスタム収集:")
        print("race_ids.txtファイルに1行1レースIDで記載してください")
        print("例:")
        print("202406030811")
        print("202405051211")
        print("...")

        if Path("race_ids.txt").exists():
            with open("race_ids.txt", 'r') as f:
                race_ids = [line.strip() for line in f if line.strip()]

            print(f"\n{len(race_ids)}件のレースIDを読み込みました")

            scraper = NetkeibaScraperComplete(delay=2.0)
            results = scraper.scrape_multiple_races(race_ids)

            if len(results) > 0:
                print(f"\n✅ 収集完了！")
        else:
            print("\n❌ race_ids.txtが見つかりません")

    else:
        print("❌ 無効な選択です")

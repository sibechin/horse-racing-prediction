"""
JRA公式サイトから指定日のレーススケジュールを取得
"""
import pandas as pd
from datetime import datetime
import logging
import argparse

# 以下はWebスクレイピング用（オプション）
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    logging.warning("requests/BeautifulSoupが利用できません。手動入力用テンプレートのみ使用可能です。")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_race_schedule(date_str):
    """
    指定日のレーススケジュールを取得

    Args:
        date_str: 日付（YYYYMMDD形式）

    Returns:
        レース情報のリスト
    """
    if not SCRAPING_AVAILABLE:
        logging.error("Webスクレイピングライブラリが利用できません")
        logging.info("pip install requests beautifulsoup4 を実行してください")
        return []

    logging.info(f"レーススケジュール取得: {date_str}")

    # JRA公式サイトのURL（開催情報）
    # 注意: 実際のURLは変更される可能性があります
    url = f"https://www.jra.go.jp/JRADB/accessD.html?CNAME={date_str}"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # HTMLをパース
        soup = BeautifulSoup(response.content, 'html.parser')

        # レース情報を抽出
        # 注意: 実際のHTML構造に応じて調整が必要
        races = []

        logging.info(f"ページ取得成功: {len(soup.text)} bytes")

        # ここでは基本的な構造のみを示します
        # 実際の実装では、JRAのHTMLパース構造に合わせる必要があります

        return races

    except requests.RequestException as e:
        logging.error(f"リクエストエラー: {e}")
        return []
    except Exception as e:
        logging.error(f"エラー: {e}")
        return []


def get_race_info_from_jravan(date_str):
    """
    JRAVAN APIから情報を取得（代替方法）

    Args:
        date_str: 日付（YYYYMMDD形式）
    """
    # JRAVANのAPIキーが必要
    logging.warning("JRAVAN APIは利用できません（APIキーが必要）")
    return []


def create_manual_input_template(date_str, output_path):
    """
    手動入力用のテンプレートCSVを作成

    Args:
        date_str: 日付
        output_path: 出力先
    """
    logging.info(f"手動入力用テンプレート作成: {output_path}")

    # テンプレートのカラム
    columns = [
        'race_id', 'race_name', 'race_date', 'venue_code', 'race_number',
        'horse_name', 'sex_encoded', 'age', 'weight',
        'popularity', 'odds_numeric', 'last_3f',
        'horse_weight_kg', 'horse_weight_change',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'year', 'month', 'day_of_week',
        'field_size', 'race_avg_odds', 'race_avg_weight',
        'race_type_encoded'
    ]

    # サンプル行を作成
    sample_data = {
        'race_id': ['例: 20251123_tokyo_11'],
        'race_name': ['例: ジャパンカップ'],
        'race_date': [date_str],
        'venue_code': [5],  # 5=東京, 6=中山, 7=中京, 8=京都, 9=阪神
        'race_number': [11],
        'horse_name': ['例: 馬名1'],
        'sex_encoded': [0],  # 0=牡, 1=牝, 2=セン
        'age': [4],
        'weight': [57.0],
        'popularity': [1],
        'odds_numeric': [3.5],
        'last_3f': [35.0],
        'horse_weight_kg': [480],
        'horse_weight_change': [0],
        'trainer_encoded': [1],  # 調教師コード
        'trainer_frequency': [50],  # 調教師の出走回数
        'jockey_encoded': [1],  # 騎手コード
        'jockey_frequency': [100],  # 騎手の出走回数
        'year': [int(date_str[:4])],
        'month': [int(date_str[4:6])],
        'day_of_week': [datetime.strptime(date_str, '%Y%m%d').weekday()],
        'field_size': [10],  # 出走頭数
        'race_avg_odds': [25.0],
        'race_avg_weight': [56.0],
        'race_type_encoded': [0]  # レースタイプ
    }

    # DataFrame作成
    df = pd.DataFrame(sample_data)

    # CSV出力
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    logging.info(f"テンプレート作成完了: {output_path}")
    logging.info("\n使い方:")
    logging.info("1. このCSVファイルを開いて、実際のレースデータを入力してください")
    logging.info("2. 各レースの出走馬すべてを行として追加してください")
    logging.info("3. 完成したら、predict_races.py で予測を実行してください")

    return df


def show_venue_codes():
    """競馬場コードの説明を表示"""
    print("\n【競馬場コード】")
    print("  5: 東京競馬場")
    print("  6: 中山競馬場")
    print("  7: 中京競馬場")
    print("  8: 京都競馬場")
    print("  9: 阪神競馬場")

    print("\n【性別コード】")
    print("  0: 牡（オス）")
    print("  1: 牝（メス）")
    print("  2: セン（去勢馬）")


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='レーススケジュール取得')
    parser.add_argument('--date', type=str, help='日付（YYYYMMDD形式）')
    parser.add_argument('--template', type=str, help='手動入力用テンプレート作成')
    parser.add_argument('--codes', action='store_true', help='競馬場コードを表示')

    args = parser.parse_args()

    if args.codes:
        show_venue_codes()
        return

    if args.template:
        # テンプレート作成
        date_str = args.date if args.date else datetime.now().strftime('%Y%m%d')
        create_manual_input_template(date_str, args.template)
        show_venue_codes()
    elif args.date:
        # スケジュール取得
        races = get_race_schedule(args.date)
        if races:
            logging.info(f"{len(races)} レースが見つかりました")
        else:
            logging.warning("レースが見つかりませんでした")
            logging.info("\n代わりに手動入力用テンプレートを使用してください:")
            logging.info(f"  python fetch_race_schedule.py --template race_data_{args.date}.csv --date {args.date}")
    else:
        parser.print_help()
        print("\n例:")
        print("  # 手動入力用テンプレート作成")
        print("  python fetch_race_schedule.py --template race_20251123.csv --date 20251123")
        print("\n  # 競馬場コード確認")
        print("  python fetch_race_schedule.py --codes")


if __name__ == "__main__":
    main()

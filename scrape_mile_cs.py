"""
マイルCS出走馬データスクレイピング

keibalab.jpから出走馬の詳細データを取得
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def scrape_race_entries(race_url):
    """
    レース出走馬情報のスクレイピング

    Args:
        race_url: レースURL

    Returns:
        DataFrame: 出走馬データ
    """
    logging.info(f"レースデータ取得中: {race_url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    response = requests.get(race_url, headers=headers)
    response.encoding = 'utf-8'

    if response.status_code != 200:
        logging.error(f"HTTPエラー: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # レース基本情報
    race_info = extract_race_info(soup)
    logging.info(f"レース名: {race_info.get('race_name', 'N/A')}")
    logging.info(f"距離: {race_info.get('distance', 'N/A')}m")
    logging.info(f"馬場: {race_info.get('surface', 'N/A')}")

    # 出走馬テーブル
    horses_data = extract_horses_data(soup, race_info)

    if not horses_data:
        logging.error("出走馬データが取得できませんでした")
        return None

    df = pd.DataFrame(horses_data)
    logging.info(f"取得完了: {len(df)}頭")

    return df


def extract_race_info(soup):
    """レース基本情報の抽出"""
    race_info = {}

    try:
        # レース名
        race_name_elem = soup.find('h1')
        if race_name_elem:
            race_info['race_name'] = race_name_elem.text.strip()

        # 距離・馬場情報
        race_details = soup.find('div', class_='racedata')
        if race_details:
            text = race_details.text

            # 距離
            distance_match = re.search(r'(\d+)m', text)
            if distance_match:
                race_info['distance'] = int(distance_match.group(1))

            # 芝/ダート
            if '芝' in text:
                race_info['surface'] = '芝'
                race_info['surface_encoded'] = 1
            elif 'ダ' in text or 'ダート' in text:
                race_info['surface'] = 'ダート'
                race_info['surface_encoded'] = 2

            # 馬場状態
            if '良' in text:
                race_info['track_condition'] = '良'
                race_info['track_condition_encoded'] = 1
            elif '稍' in text or '稍重' in text:
                race_info['track_condition'] = '稍重'
                race_info['track_condition_encoded'] = 2
            elif '重' in text:
                race_info['track_condition'] = '重'
                race_info['track_condition_encoded'] = 3
            elif '不' in text:
                race_info['track_condition'] = '不良'
                race_info['track_condition_encoded'] = 4

        # レースクラス（G1固定）
        race_info['race_class'] = 'G1'
        race_info['race_class_encoded'] = 5

    except Exception as e:
        logging.warning(f"レース情報抽出エラー: {e}")

    return race_info


def extract_horses_data(soup, race_info):
    """出走馬データの抽出"""
    horses = []

    try:
        # 出走馬テーブルを探す
        table = soup.find('table', class_='resulttable') or soup.find('table')

        if not table:
            logging.error("出走馬テーブルが見つかりません")
            return None

        rows = table.find_all('tr')[1:]  # ヘッダー行をスキップ

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 4:
                continue

            horse_data = race_info.copy()

            try:
                # 馬番
                horse_num = cols[0].text.strip()
                horse_data['horse_number'] = int(horse_num) if horse_num.isdigit() else 0

                # 馬名
                horse_name_elem = cols[1].find('a') or cols[1]
                horse_data['horse_name'] = horse_name_elem.text.strip()

                # 性齢（例: 4牡、3牝）
                sex_age = cols[2].text.strip() if len(cols) > 2 else ''
                age_match = re.search(r'(\d+)', sex_age)
                if age_match:
                    horse_data['age'] = int(age_match.group(1))

                # 斤量
                weight_text = cols[3].text.strip() if len(cols) > 3 else ''
                weight_match = re.search(r'([\d.]+)', weight_text)
                if weight_match:
                    horse_data['jockey_weight'] = float(weight_match.group(1))

                # 騎手
                jockey_elem = cols[4].find('a') if len(cols) > 4 else None
                if jockey_elem:
                    horse_data['jockey_name'] = jockey_elem.text.strip()

                # 調教師
                trainer_elem = cols[5].find('a') if len(cols) > 5 else None
                if trainer_elem:
                    horse_data['trainer_name'] = trainer_elem.text.strip()

                horses.append(horse_data)

            except Exception as e:
                logging.warning(f"馬データ抽出エラー（行スキップ）: {e}")
                continue

    except Exception as e:
        logging.error(f"テーブル解析エラー: {e}")
        return None

    # 出走頭数を設定
    field_size = len(horses)
    for horse in horses:
        horse['field_size'] = field_size

    return horses


def enrich_with_defaults(df):
    """
    不足している特徴量をデフォルト値で補完

    Args:
        df: 出走馬データ

    Returns:
        補完されたDataFrame
    """
    # モデルが必要とする19特徴量
    required_features = {
        'last_3f': 34.0,  # 平均的な上がり3ハロン
        'jockey_frequency': 150,  # 平均的な騎乗回数
        'horse_weight_kg': 480,  # 平均的な馬体重
        'weight_change': 0,  # 馬体重増減
        'horse_frequency': 20,  # 平均的な出走回数
        'jockey_win_rate': 0.10,  # 平均的な勝率
        'corner_rank_3': 5,  # 平均的なコーナー順位
        'corner_rank_4': 5,
        'trainer_frequency': 100,  # 平均的な管理回数
        'corner_rank_2': 5,
        'corner_rank_1': 5,
        'horse_win_rate': 0.08,
        'trainer_win_rate': 0.10,
    }

    # 既に取得済みの特徴量
    existing_features = ['distance', 'field_size', 'age', 'surface_encoded',
                        'track_condition_encoded', 'race_class_encoded']

    # 不足特徴量を補完
    for feature, default_value in required_features.items():
        if feature not in df.columns:
            df[feature] = default_value
            logging.info(f"デフォルト値補完: {feature} = {default_value}")

    # 既存特徴量の確認
    for feature in existing_features:
        if feature not in df.columns:
            if feature == 'distance':
                df[feature] = 1600  # マイルCS固定
            elif feature == 'surface_encoded':
                df[feature] = 1  # 芝
            elif feature == 'track_condition_encoded':
                df[feature] = 1  # 良
            elif feature == 'race_class_encoded':
                df[feature] = 5  # G1

    return df


def save_scraped_data(df, output_path):
    """スクレイピングデータの保存"""
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"データ保存: {output_path}")


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("マイルCS出走馬データスクレイピング")
    logging.info("=" * 60)

    race_url = "https://www.keibalab.jp/db/race/202511230811/"

    # スクレイピング実行
    df = scrape_race_entries(race_url)

    if df is None:
        logging.error("スクレイピング失敗")
        return

    # デフォルト値補完
    df = enrich_with_defaults(df)

    # データ保存
    output_path = Path("data/mile_cs_2025.csv")
    output_path.parent.mkdir(exist_ok=True)

    save_scraped_data(df, output_path)

    # データ確認
    logging.info("\n" + "=" * 60)
    logging.info("取得データサマリー")
    logging.info("=" * 60)
    logging.info(f"出走頭数: {len(df)}")
    logging.info(f"カラム数: {len(df.columns)}")

    print("\n取得した馬名:")
    if 'horse_name' in df.columns:
        for i, name in enumerate(df['horse_name'], 1):
            print(f"  {i:2d}. {name}")

    logging.info("\n✓ スクレイピング完了")
    logging.info(f"次のステップ: python predict_mile_cs.py を実行してください")


if __name__ == "__main__":
    main()

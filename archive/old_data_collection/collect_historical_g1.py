"""
2016-2024年 G1レースデータ収集

主要G1レースのレースIDパターンから、過去9年分のデータを収集
"""
import sys
sys.path.append('.')
from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete
import pandas as pd
from pathlib import Path

print('='*60)
print('歴史的G1レースデータ収集 (2016-2024)')
print('='*60)
print()

# 主要G1レースの典型的な開催パターン
# 2025年のパターンを基に、過去のレースIDを推測
g1_race_patterns = {
    # フェブラリーS: 2月下旬 東京ダート1600m
    'フェブラリーS': {
        'month': 2,
        'venue_code': '05',  # 東京
        'meeting': 1,  # 第1回
        'day_range': [6, 7, 8, 9]  # 週目
    },

    # 高松宮記念: 3月末 中京芝1200m
    '高松宮記念': {
        'month': 3,
        'venue_code': '07',  # 中京
        'meeting': 1,
        'day_range': [5, 6, 7, 8]
    },

    # 桜花賞: 4月中旬 阪神芝1600m
    '桜花賞': {
        'month': 4,
        'venue_code': '09',  # 阪神
        'meeting': 2,
        'day_range': [2, 3, 4]
    },

    # 皐月賞: 4月中旬-下旬 中山芝2000m
    '皐月賞': {
        'month': 4,
        'venue_code': '06',  # 中山
        'meeting': 3,
        'day_range': [3, 4, 5]
    },

    # 天皇賞(春): 5月上旬 京都芝3200m
    '天皇賞(春)': {
        'month': 5,
        'venue_code': '08',  # 京都
        'meeting': 2,
        'day_range': [1, 2, 3]
    },

    # NHKマイルC: 5月上旬 東京芝1600m
    'NHKマイルC': {
        'month': 5,
        'venue_code': '05',
        'meeting': 2,
        'day_range': [2, 3, 4]
    },

    # ヴィクトリアマイル: 5月中旬 東京芝1600m
    'ヴィクトリアマイル': {
        'month': 5,
        'venue_code': '05',
        'meeting': 2,
        'day_range': [3, 4, 5]
    },

    # オークス: 5月下旬 東京芝2400m
    'オークス': {
        'month': 5,
        'venue_code': '05',
        'meeting': 3,
        'day_range': [3, 4, 5]
    },

    # 日本ダービー: 5月末-6月上旬 東京芝2400m
    '日本ダービー': {
        'month': 6,
        'venue_code': '05',
        'meeting': 2,
        'day_range': [1, 2, 3]
    },

    # 安田記念: 6月上旬 東京芝1600m
    '安田記念': {
        'month': 6,
        'venue_code': '05',
        'meeting': 3,
        'day_range': [1, 2, 3]
    },

    # 宝塚記念: 6月下旬 阪神芝2200m
    '宝塚記念': {
        'month': 6,
        'venue_code': '09',
        'meeting': 3,
        'day_range': [3, 4, 5]
    },

    # スプリンターズS: 9月末-10月初 中山芝1200m
    'スプリンターズS': {
        'month': 10,
        'venue_code': '06',
        'meeting': 4,
        'day_range': [1, 2]
    },

    # 秋華賞: 10月中旬 京都芝2000m
    '秋華賞': {
        'month': 10,
        'venue_code': '08',
        'meeting': 4,
        'day_range': [2, 3, 4]
    },

    # 菊花賞: 10月下旬 京都芝3000m
    '菊花賞': {
        'month': 10,
        'venue_code': '08',
        'meeting': 4,
        'day_range': [4, 5, 6]
    },

    # 天皇賞(秋): 10月末-11月初 東京芝2000m
    '天皇賞(秋)': {
        'month': 11,
        'venue_code': '05',
        'meeting': 4,
        'day_range': [1, 2, 3]
    },

    # エリザベス女王杯: 11月中旬 京都芝2200m
    'エリザベス女王杯': {
        'month': 11,
        'venue_code': '08',
        'meeting': 4,
        'day_range': [2, 3, 4]
    },
}

# レースIDを生成
race_ids = []
race_info = {}

years = range(2016, 2025)  # 2016-2024

print(f'対象年: {min(years)}-{max(years)}')
print(f'レース種類: {len(g1_race_patterns)}種類')
print()

for year in years:
    for race_name, pattern in g1_race_patterns.items():
        venue = pattern['venue_code']
        meeting = pattern['meeting']

        # 各日程の可能性を試す
        for day in pattern['day_range']:
            # レースID: YYYY KK NN cc RR
            # 11R (主要レース) を試す
            race_id = f'{year}{meeting:02d}{day:02d}{venue}11'
            race_ids.append(race_id)
            race_info[race_id] = f'{race_name} ({year}年)'

print(f'生成されたレースID数: {len(race_ids)}')
print(f'推定所要時間: {len(race_ids) * 2 / 60:.1f}分')
print()

# 確認
print('サンプルレースID (最初の10個):')
for rid in race_ids[:10]:
    print(f'  {rid}: {race_info.get(rid, "Unknown")}')
print()

# 収集開始の確認
response = input('データ収集を開始しますか? (y/n): ')
if response.lower() != 'y':
    print('キャンセルしました')
    exit()

# スクレイパー初期化
scraper = NetkeibaScraperComplete(delay=2.0, max_retries=3)

# データ収集
print()
print('データ収集開始...')
print()

results = scraper.scrape_multiple_races(race_ids, output_dir='data/raw')

if len(results) > 0:
    print()
    print('='*60)
    print('収集完了!')
    print('='*60)
    print(f'成功したレース数: {results["race_id"].nunique()}')
    print(f'総データ数: {len(results)}頭')
    if 'race_date' in results.columns:
        print(f'期間: {results["race_date"].min()} - {results["race_date"].max()}')
    print()

    # 保存
    output_file = 'data/raw/g1_races_2016_2024_historical.csv'
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'データ保存: {output_file}')
    print()

    # 統計
    print('年別レース数:')
    if 'race_date' in results.columns:
        results['year'] = pd.to_datetime(results['race_date']).dt.year
        print(results.groupby('year')['race_id'].nunique())
    print()

    print('競馬場別:')
    if 'venue_name' in results.columns:
        print(results['venue_name'].value_counts())
    print()

    print('馬場タイプ別:')
    if 'track_type' in results.columns:
        print(results['track_type'].value_counts())
else:
    print()
    print('❌ データ収集失敗')
    print('レースIDのパターンを調整する必要があります')

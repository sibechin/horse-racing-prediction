"""
2025年G1レースデータ収集スクリプト

2025年2月〜11月の G1レースをnetkeibaから収集
"""
import sys
sys.path.append('.')
from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete
import pandas as pd
from pathlib import Path

print('='*60)
print('2025年 G1レースデータ収集')
print('='*60)
print()

# 2025年G1レース一覧（netkeibaレースID）
# フォーマット: YYYY MM DD 競馬場コード レース番号
# 競馬場: 東京=05, 中山=06, 京都=08, 阪神=09, 中京=07

g1_races_2025 = {
    # フェブラリーステークス (2/23, 中山11R)
    '202502230611': 'フェブラリーS',

    # 高松宮記念 (3/30, 中京11R)
    '202503300711': '高松宮記念',

    # 桜花賞 (4/13, 阪神11R)
    '202504130911': '桜花賞',

    # 皐月賞 (4/20, 中山11R)
    '202504200611': '皐月賞',

    # 天皇賞(春) (5/4, 京都11R)
    '202505040811': '天皇賞(春)',

    # NHKマイルカップ (5/11, 東京11R)
    '202505110511': 'NHKマイルC',

    # ヴィクトリアマイル (5/18, 東京11R)
    '202505180511': 'ヴィクトリアマイル',

    # オークス (5/25, 東京11R)
    '202505250511': 'オークス',

    # 日本ダービー (6/1, 東京11R)
    '202506010511': '日本ダービー',

    # 安田記念 (6/8, 東京11R)
    '202506080511': '安田記念',

    # 宝塚記念 (6/15, 阪神11R) - 2025年は阪神開催
    '202506150911': '宝塚記念',

    # スプリンターズステークス (9/28, 中山11R)
    '202509280611': 'スプリンターズS',

    # 秋華賞 (10/19, 京都11R)
    '202510190811': '秋華賞',

    # 菊花賞 (10/26, 京都11R)
    '202510260811': '菊花賞',

    # 天皇賞(秋) (11/2, 東京11R)
    '202511020511': '天皇賞(秋)',

    # エリザベス女王杯 (11/16, 京都11R)
    '202511160811': 'エリザベス女王杯',
}

race_ids = list(g1_races_2025.keys())

print(f'収集対象レース数: {len(race_ids)}')
print(f'推定所要時間: {len(race_ids) * 2 / 60:.1f}分')
print()

print('対象レース:')
for race_id, race_name in g1_races_2025.items():
    print(f'  {race_id}: {race_name}')
print()

# スクレイパー初期化
scraper = NetkeibaScraperComplete(delay=2.0, max_retries=3)

# データ収集
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
    print(f'期間: {results["race_date"].min()} - {results["race_date"].max()}')
    print()

    # 保存
    output_file = 'data/raw/g1_races_2025.csv'
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'データ保存: {output_file}')
    print()

    # 統計
    print('レース別データ数:')
    for race_id in results['race_id'].unique():
        race_data = results[results['race_id'] == race_id]
        race_name = g1_races_2025.get(race_id, 'Unknown')
        print(f'  {race_name}: {len(race_data)}頭')
else:
    print()
    print('❌ データ収集失敗')
    print('netkeibaにアクセスできない、またはレースIDが正しくない可能性があります')

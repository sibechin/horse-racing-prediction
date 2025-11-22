"""
2025年G1レースデータ収集 (正しいレースID使用)

Web検索で見つけた正しいnetkeibaレースIDを使用
"""
import sys
sys.path.append('.')
from src.data_collection.netkeiba_scraper_complete import NetkeibaScraperComplete
import pandas as pd
from pathlib import Path

print('='*60)
print('2025年 G1レースデータ収集 (正しいID)')
print('='*60)
print()

# 2025年G1レース (正しいnetkeibaレースID)
g1_races_2025_correct = {
    # Web検索で確認済み
    '202505010811': 'フェブラリーS (2/23 東京)',
    '202505021211': '日本ダービー (6/1 東京)',
    '202505030211': '安田記念 (6/8 東京)',

    # 追加調査が必要
    # '202501050211': '高松宮記念 (3/30 中京)',  # 要確認
    # '202502030911': '桜花賞 (4/13 阪神)',      # 要確認
    # '202501030611': '皐月賞 (4/20 中山)',      # 要確認
    # '202503020811': '天皇賞(春) (5/4 京都)',   # 要確認
    # '202505020511': 'NHKマイルC (5/11 東京)',  # 要確認
    # '202505030311': 'ヴィクトリアマイル (5/18 東京)', # 要確認
    # '202505031011': 'オークス (5/25 東京)',    # 要確認
}

race_ids = list(g1_races_2025_correct.keys())

print(f'収集対象レース数: {len(race_ids)}')
print(f'推定所要時間: {len(race_ids) * 2 / 60:.1f}分')
print()

print('対象レース:')
for race_id, race_name in g1_races_2025_correct.items():
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
    if 'race_date' in results.columns:
        print(f'期間: {results["race_date"].min()} - {results["race_date"].max()}')
    print()

    # 保存
    output_file = 'data/raw/g1_races_2025_correct.csv'
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'データ保存: {output_file}')
    print()

    # 統計
    print('レース別データ数:')
    for race_id in results['race_id'].unique():
        race_data = results[results['race_id'] == race_id]
        race_name = g1_races_2025_correct.get(race_id, 'Unknown')
        print(f'  {race_name}: {len(race_data)}頭')

    print()
    print('馬場タイプ別:')
    if 'track_type' in results.columns:
        print(results['track_type'].value_counts())
else:
    print()
    print('❌ データ収集失敗')
    print('レースIDが正しくない可能性があります')

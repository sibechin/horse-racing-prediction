import pandas as pd

df = pd.read_csv('data/processed/keibalab_g1_2025_processed.csv', encoding='utf-8-sig')

race_info = df.groupby('race_id')[['expected_race_name', 'expected_date']].first().sort_values('expected_date')

print(f'Total unique races: {len(race_info)}')
print(f'Total records: {len(df)}')
print('\nAll races:')

for idx, (race_id, row) in enumerate(race_info.iterrows(), 1):
    race_name = row['expected_race_name']
    race_date = row['expected_date']
    num_horses = len(df[df['race_id'] == race_id])
    print(f'{idx:2d}. {race_date} | {race_name:30s} | {num_horses:2d}頭 | ID: {race_id}')

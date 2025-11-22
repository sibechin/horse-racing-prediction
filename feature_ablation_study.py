"""
特徴量アブレーション研究

リークなしの特徴量で最適な組み合わせを探索
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# リークなしの安全な特徴量
SAFE_FEATURES = {
    'odds': 'オッズ',
    'popularity_rank': '人気順位',
    'weight': '斤量',
    'age': '馬齢',
    'field_size': '出走頭数',
    'distance': '距離',
    'sex_encoded': '性別',
    'month': '月',
    'day_of_week': '曜日',
    'race_avg_odds': 'レース平均オッズ',
}

class RaceDataset(Dataset):
    def __init__(self, races):
        self.races = races
    def __len__(self):
        return len(self.races)
    def __getitem__(self, i):
        X, y = self.races[i]['X'], self.races[i]['y']
        n = len(X)
        if n < 18:
            X = np.vstack([X, np.zeros((18-n, X.shape[1]))])
            y = np.concatenate([y, np.zeros(18-n)])
        return torch.FloatTensor(X[:18]), torch.FloatTensor(y[:18]), n

class SimpleTransformer(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.3))
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, dropout=0.3, batch_first=True), 2
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        return self.out(self.tf(self.proj(x))).squeeze(-1)

def prepare_data(df, feature_cols):
    """データ準備"""
    races = []
    for rid in df['race_id'].unique():
        rd = df[df['race_id'] == rid].sample(frac=1, random_state=None)
        X = rd[feature_cols].fillna(0).values
        pos = pd.to_numeric(rd['finish_position'], errors='coerce').values
        y = (pos == 1).astype(float)
        if len(X) >= 2 and not np.isnan(pos).all():
            races.append({'X': X, 'y': y})
    return races

def train_and_evaluate(train_races, test_races, n_features, epochs=40):
    """訓練と評価"""
    if len(train_races) < 10:
        return {'top1': 0, 'top3': 0, 'total': 0}

    scaler = StandardScaler()
    scaler.fit(np.vstack([r['X'] for r in train_races]))

    for r in train_races + test_races:
        r['X'] = scaler.transform(r['X'])

    model = SimpleTransformer(n_features)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    train_dl = DataLoader(RaceDataset(train_races), batch_size=8, shuffle=True)

    for _ in range(epochs):
        model.train()
        for X, y, _ in train_dl:
            loss = F.binary_cross_entropy(torch.sigmoid(model(X)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 評価
    model.eval()
    results = {'top1': 0, 'top2': 0, 'top3': 0, 'top3_exact': 0}

    for r in test_races:
        X = torch.FloatTensor(r['X']).unsqueeze(0)
        with torch.no_grad():
            scores = torch.sigmoid(model(X)).numpy()[0, :len(r['y'])]

        pred = np.argsort(scores)[::-1]
        actual = np.argsort(r['y'])[::-1]

        if pred[0] == actual[0]:
            results['top1'] += 1
        if set(pred[:2]) == set(actual[:2]):
            results['top2'] += 1
        if set(pred[:3]) == set(actual[:3]):
            results['top3'] += 1
        if list(pred[:3]) == list(actual[:3]):
            results['top3_exact'] += 1

    results['total'] = len(test_races)
    return results

def main():
    logging.info("="*60)
    logging.info("特徴量アブレーション研究")
    logging.info("="*60)

    # データ読み込み
    df = pd.read_csv('data/processed/merged_data_all.csv', encoding='utf-8-sig')
    df['year'] = df['year'].fillna(2020)

    # 時系列分割
    train_df = df[df['year'] < 2023]
    test_df = df[df['year'] >= 2023]

    logging.info(f"訓練: {train_df['race_id'].nunique()} races")
    logging.info(f"テスト: {test_df['race_id'].nunique()} races")

    # 利用可能な特徴量を確認
    available = [f for f in SAFE_FEATURES.keys() if f in df.columns]
    logging.info(f"\n利用可能な安全特徴量: {available}")

    results_all = []

    # 1. 単一特徴量テスト
    logging.info("\n" + "="*60)
    logging.info("1. 単一特徴量テスト")
    logging.info("="*60)

    for feat in available:
        train_races = prepare_data(train_df, [feat])
        test_races = prepare_data(test_df, [feat])

        res = train_and_evaluate(train_races, test_races, 1, epochs=30)

        if res['total'] > 0:
            top1_rate = res['top1'] / res['total']
            results_all.append({
                'features': feat,
                'n_features': 1,
                'top1': top1_rate,
                'top1_count': res['top1'],
                'total': res['total']
            })
            logging.info(f"  {feat}: 単勝 {top1_rate:.1%} ({res['top1']}/{res['total']})")

    # 2. 2特徴量の組み合わせ
    logging.info("\n" + "="*60)
    logging.info("2. 2特徴量の組み合わせ (上位のみ表示)")
    logging.info("="*60)

    combo_results = []
    for combo in combinations(available, 2):
        feat_list = list(combo)
        train_races = prepare_data(train_df, feat_list)
        test_races = prepare_data(test_df, feat_list)

        res = train_and_evaluate(train_races, test_races, 2, epochs=30)

        if res['total'] > 0:
            combo_results.append({
                'features': '+'.join(feat_list),
                'n_features': 2,
                'top1': res['top1'] / res['total'],
                'top3': res['top3'] / res['total'],
                'top1_count': res['top1'],
                'total': res['total']
            })

    combo_results.sort(key=lambda x: x['top1'], reverse=True)
    for r in combo_results[:10]:
        logging.info(f"  {r['features']}: 単勝 {r['top1']:.1%}, 三連複 {r['top3']:.1%}")

    results_all.extend(combo_results)

    # 3. 3特徴量の組み合わせ
    logging.info("\n" + "="*60)
    logging.info("3. 3特徴量の組み合わせ (上位のみ表示)")
    logging.info("="*60)

    combo_results = []
    for combo in combinations(available, 3):
        feat_list = list(combo)
        train_races = prepare_data(train_df, feat_list)
        test_races = prepare_data(test_df, feat_list)

        res = train_and_evaluate(train_races, test_races, 3, epochs=30)

        if res['total'] > 0:
            combo_results.append({
                'features': '+'.join(feat_list),
                'n_features': 3,
                'top1': res['top1'] / res['total'],
                'top3': res['top3'] / res['total'],
                'top1_count': res['top1'],
                'total': res['total']
            })

    combo_results.sort(key=lambda x: x['top1'], reverse=True)
    for r in combo_results[:10]:
        logging.info(f"  {r['features']}: 単勝 {r['top1']:.1%}, 三連複 {r['top3']:.1%}")

    results_all.extend(combo_results)

    # 4. 全特徴量
    logging.info("\n" + "="*60)
    logging.info("4. 全特徴量")
    logging.info("="*60)

    train_races = prepare_data(train_df, available)
    test_races = prepare_data(test_df, available)

    res = train_and_evaluate(train_races, test_races, len(available), epochs=50)

    if res['total'] > 0:
        logging.info(f"  全特徴量 ({len(available)}個): 単勝 {res['top1']/res['total']:.1%}, 三連複 {res['top3']/res['total']:.1%}")
        results_all.append({
            'features': 'ALL',
            'n_features': len(available),
            'top1': res['top1'] / res['total'],
            'top3': res['top3'] / res['total'],
            'top1_count': res['top1'],
            'total': res['total']
        })

    # 5. 最良の組み合わせを表示
    logging.info("\n" + "="*60)
    logging.info("最良の特徴量組み合わせ TOP10")
    logging.info("="*60)

    results_all.sort(key=lambda x: x['top1'], reverse=True)

    for i, r in enumerate(results_all[:10], 1):
        top3 = r.get('top3', 0)
        logging.info(f"{i}. {r['features']}: 単勝 {r['top1']:.1%} ({r['top1_count']}/{r['total']}), 三連複 {top3:.1%}")

    # ベースライン比較
    logging.info("\n" + "="*60)
    logging.info("ベースライン比較")
    logging.info("="*60)

    # 1番人気ベースライン
    correct = 0
    total = 0
    for rid in test_df['race_id'].unique():
        rd = test_df[test_df['race_id'] == rid]
        odds = rd['odds'].values
        pos = pd.to_numeric(rd['finish_position'], errors='coerce').values
        if not np.isnan(odds).all() and not np.isnan(pos).all():
            if np.nanargmin(odds) == np.nanargmin(pos):
                correct += 1
            total += 1

    logging.info(f"1番人気（オッズ最低）: {correct/total:.1%} ({correct}/{total})")
    logging.info(f"最良モデル: {results_all[0]['top1']:.1%}")

if __name__ == "__main__":
    main()

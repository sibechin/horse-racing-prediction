"""
包括的特徴量アブレーション研究

全ての特徴量組み合わせをテスト
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
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 使用可能な特徴量（ターゲット変数とID以外）
CANDIDATE_FEATURES = [
    'year', 'month', 'day_of_week', 'venue_code', 'race_number',
    'sex_encoded', 'age', 'weight', 'popularity', 'odds_numeric',
    'last_3f', 'horse_weight_kg', 'horse_weight_change', 'field_size',
    'race_avg_odds', 'race_avg_weight', 'trainer_encoded', 'trainer_frequency',
    'jockey_encoded', 'jockey_frequency', 'race_type_encoded'
]


class RaceDataset(Dataset):
    def __init__(self, races, max_horses=18):
        self.races = races
        self.max_horses = max_horses

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        X, y = self.races[idx]['X'], self.races[idx]['y']
        n = len(X)

        if n < self.max_horses:
            X = np.vstack([X, np.zeros((self.max_horses - n, X.shape[1]))])
            y = np.concatenate([y, np.zeros(self.max_horses - n)])
            mask = np.concatenate([np.zeros(n), np.ones(self.max_horses - n)])
        else:
            X, y = X[:self.max_horses], y[:self.max_horses]
            mask = np.zeros(self.max_horses)

        return (torch.FloatTensor(X), torch.FloatTensor(y),
                torch.BoolTensor(mask.astype(bool)), n)


class SimpleTransformer(nn.Module):
    def __init__(self, d_in, d_model=128, n_heads=8, n_layers=3, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True),
            n_layers
        )
        self.out = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, mask=None):
        return self.out(self.tf(self.proj(x), src_key_padding_mask=mask)).squeeze(-1)


def prepare_races(df, features):
    """レースデータ準備"""
    races = []
    for rid in df['race_id'].unique():
        rd = df[df['race_id'] == rid].sample(frac=1, random_state=None)
        X = rd[features].fillna(0).values
        y = (rd['finish_position_numeric'] == 1).astype(float).values

        if len(X) >= 2 and not np.isnan(y).all():
            races.append({'X': X, 'y': y})
    return races


def train_evaluate(train_races, test_races, n_features, epochs=30, batch_size=256):
    """訓練と評価"""
    if len(train_races) < 10 or len(test_races) < 5:
        return None

    # スケーリング
    scaler = StandardScaler()
    all_X = np.vstack([r['X'] for r in train_races])
    scaler.fit(all_X)

    for r in train_races + test_races:
        r['X'] = scaler.transform(r['X'])

    # モデル - GPU最適化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleTransformer(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Mixed precision training for faster GPU performance
    scaler_amp = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    train_dl = DataLoader(
        RaceDataset(train_races),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == 'cuda'),  # Faster data transfer to GPU
        num_workers=8,  # Parallel data loading (increased)
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # Prefetch batches to reduce GPU wait time
    )

    # 訓練 - Mixed precision で高速化
    for _ in range(epochs):
        model.train()
        for X, y, mask, _ in train_dl:
            X, y, mask = X.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            if scaler_amp:  # GPU with mixed precision
                with torch.amp.autocast('cuda'):
                    pred = torch.sigmoid(model(X, mask))
                # Loss computation must be outside autocast
                loss = F.binary_cross_entropy(pred.float() * (~mask).float(), y * (~mask).float())
                opt.zero_grad()
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            else:  # CPU or no mixed precision
                pred = torch.sigmoid(model(X, mask))
                loss = F.binary_cross_entropy(pred * (~mask), y * (~mask))
                opt.zero_grad()
                loss.backward()
                opt.step()

    # 評価
    model.eval()
    results = {'top1': 0, 'top2': 0, 'top3': 0, 'top3_exact': 0}

    with torch.no_grad():
        for r in test_races:
            X = torch.FloatTensor(r['X']).unsqueeze(0).to(device)
            scores = torch.sigmoid(model(X)).cpu().numpy()[0, :len(r['y'])]

            pred_rank = np.argsort(scores)[::-1]
            actual_rank = np.argsort(r['y'])[::-1]

            if pred_rank[0] == actual_rank[0]:
                results['top1'] += 1
            if set(pred_rank[:2]) == set(actual_rank[:2]):
                results['top2'] += 1
            if set(pred_rank[:3]) == set(actual_rank[:3]):
                results['top3'] += 1
            if list(pred_rank[:3]) == list(actual_rank[:3]):
                results['top3_exact'] += 1

    n = len(test_races)
    return {k: v / n for k, v in results.items()} | {'total': n, 'raw': results}


def main():
    logging.info("="*70)
    logging.info("包括的特徴量アブレーション研究")
    logging.info("="*70)

    # GPU情報
    if torch.cuda.is_available():
        logging.info(f"\n[GPU] {torch.cuda.get_device_name(0)}")
        logging.info(f"   CUDA Version: {torch.version.cuda}")
        logging.info(f"   メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logging.info(f"   最適化: Mixed Precision (FP16) + Batch Size 256")
        logging.info(f"   DataLoader: 8 Workers + Prefetch Factor 4")
        logging.info(f"   モデル: d_model=128, n_heads=8, n_layers=3")
    else:
        logging.info("\n[CPU] GPU利用不可")

    # データ読み込み
    df = pd.read_csv('data/processed/keibalab_g1_2000_2024_cleaned.csv', encoding='utf-8-sig')
    df_test = pd.read_csv('data/processed/keibalab_g1_2025_cleaned.csv', encoding='utf-8-sig')

    logging.info(f"訓練データ: {len(df)} records, {df['race_id'].nunique()} races")
    logging.info(f"テストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races")

    # 利用可能な特徴量
    available = [f for f in CANDIDATE_FEATURES if f in df.columns and f in df_test.columns]
    logging.info(f"\n利用可能な特徴量 ({len(available)}個):")
    for f in available:
        logging.info(f"  - {f}")

    all_results = []

    # ========================================
    # 1. 単一特徴量テスト
    # ========================================
    logging.info("\n" + "="*70)
    logging.info("1. 単一特徴量テスト")
    logging.info("="*70)

    for feat in available:
        train_races = prepare_races(df, [feat])
        test_races = prepare_races(df_test, [feat])

        res = train_evaluate(train_races, test_races, 1, epochs=25)
        if res:
            all_results.append({
                'features': [feat],
                'n_features': 1,
                **res
            })
            logging.info(f"  {feat}: 単勝 {res['top1']:.1%}, 三連複 {res['top3']:.1%}")

    # ========================================
    # 2. 2特徴量の組み合わせ
    # ========================================
    logging.info("\n" + "="*70)
    logging.info(f"2. 2特徴量の組み合わせ ({len(list(combinations(available, 2)))}通り)")
    logging.info("="*70)

    combo2_results = []
    for i, combo in enumerate(combinations(available, 2)):
        if i % 20 == 0:
            logging.info(f"  進捗: {i}/{len(list(combinations(available, 2)))}")

        feat_list = list(combo)
        train_races = prepare_races(df, feat_list)
        test_races = prepare_races(df_test, feat_list)

        res = train_evaluate(train_races, test_races, 2, epochs=25)
        if res:
            combo2_results.append({
                'features': feat_list,
                'n_features': 2,
                **res
            })

    combo2_results.sort(key=lambda x: x['top1'], reverse=True)
    logging.info("\n  上位10組み合わせ:")
    for r in combo2_results[:10]:
        logging.info(f"    {'+'.join(r['features'])}: 単勝 {r['top1']:.1%}, 三連複 {r['top3']:.1%}")
    all_results.extend(combo2_results)

    # ========================================
    # 3. 3特徴量の組み合わせ
    # ========================================
    logging.info("\n" + "="*70)
    logging.info(f"3. 3特徴量の組み合わせ ({len(list(combinations(available, 3)))}通り)")
    logging.info("="*70)

    combo3_results = []
    combos3 = list(combinations(available, 3))
    for i, combo in enumerate(combos3):
        if i % 100 == 0:
            logging.info(f"  進捗: {i}/{len(combos3)}")

        feat_list = list(combo)
        train_races = prepare_races(df, feat_list)
        test_races = prepare_races(df_test, feat_list)

        res = train_evaluate(train_races, test_races, 3, epochs=25)
        if res:
            combo3_results.append({
                'features': feat_list,
                'n_features': 3,
                **res
            })

    combo3_results.sort(key=lambda x: x['top1'], reverse=True)
    logging.info("\n  上位10組み合わせ:")
    for r in combo3_results[:10]:
        logging.info(f"    {'+'.join(r['features'])}: 単勝 {r['top1']:.1%}, 三連複 {r['top3']:.1%}")
    all_results.extend(combo3_results)

    # ========================================
    # 4. 4特徴量の組み合わせ（サンプリング）
    # ========================================
    logging.info("\n" + "="*70)
    logging.info("4. 4特徴量の組み合わせ (上位3特徴量ベース)")
    logging.info("="*70)

    # 上位3特徴量の組み合わせをベースに4特徴量を試す
    top3_features = set()
    for r in combo3_results[:20]:
        top3_features.update(r['features'])

    combo4_results = []
    tested = set()

    for base_combo in combo3_results[:30]:
        base = set(base_combo['features'])
        for extra in available:
            if extra not in base:
                new_combo = tuple(sorted(base | {extra}))
                if new_combo not in tested:
                    tested.add(new_combo)
                    feat_list = list(new_combo)

                    train_races = prepare_races(df, feat_list)
                    test_races = prepare_races(df_test, feat_list)

                    res = train_evaluate(train_races, test_races, 4, epochs=25)
                    if res:
                        combo4_results.append({
                            'features': feat_list,
                            'n_features': 4,
                            **res
                        })

    combo4_results.sort(key=lambda x: x['top1'], reverse=True)
    logging.info("\n  上位10組み合わせ:")
    for r in combo4_results[:10]:
        logging.info(f"    {'+'.join(r['features'])}: 単勝 {r['top1']:.1%}, 三連複 {r['top3']:.1%}")
    all_results.extend(combo4_results)

    # ========================================
    # 5. 5-8特徴量（貪欲法）
    # ========================================
    logging.info("\n" + "="*70)
    logging.info("5. 5-8特徴量 (貪欲法)")
    logging.info("="*70)

    # 最良の4特徴量からスタート
    if combo4_results:
        best_features = set(combo4_results[0]['features'])
    else:
        best_features = set(combo3_results[0]['features'])

    for n_feat in range(5, 9):
        best_score = 0
        best_new_feat = None

        for feat in available:
            if feat not in best_features:
                test_features = list(best_features | {feat})
                train_races = prepare_races(df, test_features)
                test_races = prepare_races(df_test, test_features)

                res = train_evaluate(train_races, test_races, len(test_features), epochs=30)
                if res and res['top1'] > best_score:
                    best_score = res['top1']
                    best_new_feat = feat
                    best_res = res

        if best_new_feat:
            best_features.add(best_new_feat)
            all_results.append({
                'features': list(best_features),
                'n_features': n_feat,
                **best_res
            })
            logging.info(f"  {n_feat}特徴量: {'+'.join(best_features)}")
            logging.info(f"    単勝 {best_res['top1']:.1%}, 三連複 {best_res['top3']:.1%}")

    # ========================================
    # 6. 全特徴量
    # ========================================
    logging.info("\n" + "="*70)
    logging.info("6. 全特徴量")
    logging.info("="*70)

    train_races = prepare_races(df, available)
    test_races = prepare_races(df_test, available)

    res = train_evaluate(train_races, test_races, len(available), epochs=40)
    if res:
        all_results.append({
            'features': available,
            'n_features': len(available),
            **res
        })
        logging.info(f"  全{len(available)}特徴量: 単勝 {res['top1']:.1%}, 三連複 {res['top3']:.1%}")

    # ========================================
    # 最終結果
    # ========================================
    logging.info("\n" + "="*70)
    logging.info("最終結果 - TOP20")
    logging.info("="*70)

    all_results.sort(key=lambda x: x['top1'], reverse=True)

    for i, r in enumerate(all_results[:20], 1):
        feat_str = '+'.join(r['features']) if len(r['features']) <= 5 else f"{len(r['features'])}特徴量"
        logging.info(f"{i:2d}. [{r['n_features']}] {feat_str}")
        logging.info(f"    単勝: {r['top1']:.1%}, 馬連: {r['top2']:.1%}, 三連複: {r['top3']:.1%}, 三連単: {r['top3_exact']:.1%}")

    # 結果保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'models/feature_study_results_{timestamp}.json'

    save_results = [{
        'features': r['features'],
        'n_features': r['n_features'],
        'top1': r['top1'],
        'top2': r['top2'],
        'top3': r['top3'],
        'top3_exact': r['top3_exact']
    } for r in all_results[:100]]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)

    logging.info(f"\n結果保存: {output_path}")


if __name__ == "__main__":
    main()

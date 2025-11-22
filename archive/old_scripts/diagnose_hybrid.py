"""
ハイブリッドモデル診断スクリプト

0% Top1精度の原因を特定
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# データ読み込み
df = pd.read_csv("data/processed/cleaned_data_all.csv", encoding='utf-8-sig')
logging.info(f"データ: {len(df)}レコード, {df['race_id'].nunique()}レース")

# LightGBMモデル読み込み
lgb_models = []
for i in range(1, 8):
    model_path = Path(f'models/optimized/optimized_model_fold{i}.txt')
    if model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        lgb_models.append(model)

logging.info(f"LightGBMモデル: {len(lgb_models)}個")

# 1レースで診断
test_race_id = df['race_id'].iloc[0]
race_data = df[df['race_id'] == test_race_id].copy()

logging.info(f"\n診断レース: {test_race_id}")
logging.info(f"出走頭数: {len(race_data)}")
logging.info(f"\n実際の順位:")
print(race_data[['horse_name', 'finish_position', 'odds']].sort_values('finish_position'))

# LightGBM予測
feature_cols = lgb_models[0].feature_name()
available_features = [f for f in feature_cols if f in race_data.columns]

X = race_data[available_features].copy()
for col in X.columns:
    if X[col].isnull().any():
        X.loc[:, col] = X[col].fillna(X[col].median())

lgb_preds = []
for model in lgb_models:
    pred = model.predict(X)
    lgb_preds.append(pred)

lgb_ensemble = np.mean(lgb_preds, axis=0)

logging.info(f"\nLightGBM予測スコア:")
for idx, (name, score, pos) in enumerate(zip(race_data['horse_name'], lgb_ensemble, race_data['finish_position'])):
    logging.info(f"  {name}: {score:.4f} (実際{int(pos)}着)")

# 予測順位
predicted_order = np.argsort(-lgb_ensemble)
logging.info(f"\nLightGBM予測順位:")
for rank, idx in enumerate(predicted_order[:5], 1):
    horse_name = race_data.iloc[idx]['horse_name']
    actual_pos = race_data.iloc[idx]['finish_position']
    logging.info(f"  {rank}位予測: {horse_name} (実際{int(actual_pos)}着)")

# 実際の1着
true_winner_idx = race_data['finish_position'].values.argmin()
true_winner_name = race_data.iloc[true_winner_idx]['horse_name']

pred_winner_idx = predicted_order[0]
pred_winner_name = race_data.iloc[pred_winner_idx]['horse_name']

logging.info(f"\n実際の勝者: {true_winner_name} (インデックス {true_winner_idx})")
logging.info(f"予測の勝者: {pred_winner_name} (インデックス {pred_winner_idx})")
logging.info(f"正解: {true_winner_idx == pred_winner_idx}")

# 正規化の影響確認
lgb_norm = (lgb_ensemble - lgb_ensemble.min()) / (lgb_ensemble.max() - lgb_ensemble.min() + 1e-10)
logging.info(f"\n正規化後:")
for idx, (name, score_orig, score_norm) in enumerate(zip(race_data['horse_name'], lgb_ensemble, lgb_norm)):
    logging.info(f"  {name}: {score_orig:.4f} → {score_norm:.4f}")

# オッズ逆数
odds_values = race_data['odds'].values
odds_inv = 1.0 / (odds_values + 1e-10)
odds_norm = (odds_inv - odds_inv.min()) / (odds_inv.max() - odds_inv.min() + 1e-10)

logging.info(f"\nオッズ逆数 (正規化後):")
for name, odds, score in zip(race_data['horse_name'], odds_values, odds_norm):
    logging.info(f"  {name}: オッズ{odds:.1f}倍 → {score:.4f}")

# ハイブリッド
hybrid = 0.70 * lgb_norm + 0.05 * odds_norm
logging.info(f"\nハイブリッド (LightGBM 70% + オッズ 5%):")
for name, score in zip(race_data['horse_name'], hybrid):
    logging.info(f"  {name}: {score:.4f}")

hybrid_order = np.argsort(-hybrid)
hybrid_winner_idx = hybrid_order[0]
hybrid_winner_name = race_data.iloc[hybrid_winner_idx]['horse_name']

logging.info(f"\nハイブリッド予測勝者: {hybrid_winner_name}")
logging.info(f"正解: {true_winner_idx == hybrid_winner_idx}")

"""
アンサンブル + 統計分析ハイブリッドモデル

フェーズ1: 5-Fold モデルのアンサンブル
フェーズ2: 統計分析との融合
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from pathlib import Path
import json

print("="*60)
print("アンサンブル + 統計分析ハイブリッドモデル")
print("="*60)
print()

# データ読み込み
df = pd.read_csv("data/processed/features_engineered.csv", encoding="utf-8-sig")

# race_dateをdatetime型に変換
df['race_date'] = pd.to_datetime(df['race_date'])

print(f"データ読み込み: {len(df)}レコード")
print(f"レース数: {df['race_id'].nunique()}")
print()

# 特徴量準備
feature_cols = [
    "distance", "weight", "age",
    "track_type_encoded", "track_condition_encoded", "weather_encoded",
    "sex_encoded", "distance_category_encoded", "track_combined_encoded",
    "jockey_win_rate", "jockey_top3_rate", "jockey_track_win_rate",
    "horse_win_rate", "horse_top3_rate", "horse_dist_win_rate", "horse_race_count",
    "field_size", "race_avg_odds", "popularity_rank", "odds",
    "year", "month", "day_of_week",
    "time_seconds"
]

available_features = [col for col in feature_cols if col in df.columns]

# 欠損値処理
df[available_features] = df[available_features].fillna(0)

X = df[available_features].values
y = df["finish_position"].values
groups = df["race_id"].values

# ===========================================================
# フェーズ1: 5-Fold LightGBMアンサンブル
# ===========================================================
print("="*60)
print("フェーズ1: 5-Fold LightGBMアンサンブル")
print("="*60)
print()

# 最適化されたモデルを読み込み
model_dir = Path("models")
fold_models = []

print("Foldモデル読み込み中...")
for i in range(1, 6):
    model_path = model_dir / f"lightgbm_lambdarank_refined_fold{i}.txt"
    if model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        fold_models.append(model)
        print(f"  Fold {i} 読み込み完了")
    else:
        print(f"  警告: Fold {i} が見つかりません")

print(f"\n読み込んだモデル数: {len(fold_models)}")
print()

# アンサンブル予測（単純平均）
print("アンサンブル予測を生成中...")
ensemble_predictions = np.zeros(len(X))

for i, model in enumerate(fold_models, 1):
    pred = model.predict(X)
    ensemble_predictions += pred
    print(f"  Fold {i} 予測完了")

ensemble_predictions /= len(fold_models)
print("アンサンブル予測完了")
print()

# ===========================================================
# フェーズ2: 統計分析モデル
# ===========================================================
print("="*60)
print("フェーズ2: 統計分析モデル")
print("="*60)
print()

# 2-1. オッズベース確率モデル
print("1. オッズベース確率モデル")
print("   市場オッズから勝率を推定")

# オッズから確率を計算（単純化）
# 理論上: 確率 ≈ 1 / オッズ
# 実際はマージンがあるため補正が必要
df['odds_probability'] = 1.0 / (df['odds'] + 0.1)  # 0除算防止

# レースごとに正規化（合計が1になるように）
def normalize_probabilities(group):
    total = group['odds_probability'].sum()
    group['odds_probability_normalized'] = group['odds_probability'] / total
    return group

df = df.groupby('race_id', group_keys=False).apply(normalize_probabilities)

print(f"   平均オッズ確率: {df['odds_probability_normalized'].mean():.4f}")
print()

# 2-2. 統計的成績スコア
print("2. 統計的成績スコアモデル")
print("   騎手・馬の過去成績を統計的に重み付け")

# 騎手スコア（勝率 × 3 + 複勝率）
df['jockey_score'] = df['jockey_win_rate'] * 3 + df['jockey_top3_rate']

# 馬スコア（勝率 × 3 + 複勝率 + 距離適性）
df['horse_score'] = (
    df['horse_win_rate'] * 3 +
    df['horse_top3_rate'] +
    df['horse_dist_win_rate'] * 2
)

# 総合統計スコア
df['statistical_score'] = (
    df['jockey_score'] * 0.4 +  # 騎手40%
    df['horse_score'] * 0.4 +    # 馬40%
    (1.0 / (df['popularity_rank'] + 1)) * 0.2  # 人気20%
)

print(f"   平均統計スコア: {df['statistical_score'].mean():.4f}")
print()

# 2-3. ロジスティック回帰モデル
print("3. ロジスティック回帰モデル")
print("   統計的手法で勝率を予測")

# Top3フィニッシュを目的変数に
y_binary = (df['finish_position'] <= 3).astype(int)

# 主要特徴量のみ使用（統計モデルは解釈性重視）
stat_features = [
    'jockey_win_rate', 'jockey_top3_rate',
    'horse_win_rate', 'horse_top3_rate',
    'odds', 'popularity_rank',
    'weight', 'age', 'field_size'
]

X_stat = df[stat_features].fillna(0).values

# スケーリング（ロジスティック回帰に必要）
scaler = StandardScaler()
X_stat_scaled = scaler.fit_transform(X_stat)

# 5-Fold CVで評価
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

logistic_predictions = np.zeros(len(X))

print("   5-Fold CV訓練中...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_stat_scaled, y_binary, groups), 1):
    X_train, X_val = X_stat_scaled[train_idx], X_stat_scaled[val_idx]
    y_train = y_binary[train_idx]

    # ロジスティック回帰訓練
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # 確率予測
    val_probs = lr_model.predict_proba(X_val)[:, 1]
    logistic_predictions[val_idx] = val_probs

    print(f"      Fold {fold} 完了")

print("   ロジスティック回帰完了")
print()

# ===========================================================
# フェーズ3: ハイブリッド予測
# ===========================================================
print("="*60)
print("フェーズ3: ハイブリッド予測")
print("="*60)
print()

print("複数の予測手法を統合:")
print("  1. LightGBM アンサンブル (機械学習)")
print("  2. オッズベース確率 (市場)")
print("  3. 統計スコア (過去成績)")
print("  4. ロジスティック回帰 (統計モデル)")
print()

# 各予測を正規化（0-1範囲に）
def normalize_predictions(pred):
    min_val = pred.min()
    max_val = pred.max()
    if max_val - min_val == 0:
        return np.ones_like(pred) * 0.5
    return (pred - min_val) / (max_val - min_val)

ensemble_norm = normalize_predictions(ensemble_predictions)
odds_norm = df['odds_probability_normalized'].values
stat_norm = normalize_predictions(df['statistical_score'].values)
logistic_norm = logistic_predictions

# ハイブリッド予測（重み付け平均）
# 重み: LightGBM 50%, オッズ 20%, 統計スコア 15%, ロジスティック 15%
hybrid_predictions = (
    0.50 * ensemble_norm +
    0.20 * odds_norm +
    0.15 * stat_norm +
    0.15 * logistic_norm
)

df['ensemble_prediction'] = ensemble_predictions
df['odds_probability'] = odds_norm
df['statistical_score_norm'] = stat_norm
df['logistic_prediction'] = logistic_norm
df['hybrid_prediction'] = hybrid_predictions

print("ハイブリッド重み付け:")
print("  LightGBM アンサンブル: 50%")
print("  オッズ確率:           20%")
print("  統計スコア:           15%")
print("  ロジスティック回帰:   15%")
print()

# ===========================================================
# 評価: レースごとの予測ランキング生成
# ===========================================================
print("="*60)
print("予測評価")
print("="*60)
print()

# レースごとに予測ランキングを作成
# LightGBMのLambdaRankは低いスコア = 良い順位なので ascending=True
# ハイブリッドとオッズは高いスコア = 良い順位なので ascending=False
df['predicted_rank_ensemble'] = df.groupby('race_id')['ensemble_prediction'].rank(ascending=True, method='min')
df['predicted_rank_hybrid'] = df.groupby('race_id')['hybrid_prediction'].rank(ascending=False, method='min')
df['predicted_rank_odds'] = df.groupby('race_id')['odds_probability'].rank(ascending=False, method='min')

# Top1的中率（1着予測）
def calculate_top1_accuracy(df, pred_col):
    top1_predicted = df[df[pred_col] == 1]
    correct = (top1_predicted['finish_position'] == 1).sum()
    total = len(df['race_id'].unique())
    return correct / total if total > 0 else 0

# Top3的中率（3着以内予測）
def calculate_top3_accuracy(df, pred_col):
    top3_predicted = df[df[pred_col] <= 3]
    correct = (top3_predicted['finish_position'] <= 3).sum()
    total = len(top3_predicted)
    return correct / total if total > 0 else 0

acc_ensemble_top1 = calculate_top1_accuracy(df, 'predicted_rank_ensemble')
acc_hybrid_top1 = calculate_top1_accuracy(df, 'predicted_rank_hybrid')
acc_odds_top1 = calculate_top1_accuracy(df, 'predicted_rank_odds')

acc_ensemble_top3 = calculate_top3_accuracy(df, 'predicted_rank_ensemble')
acc_hybrid_top3 = calculate_top3_accuracy(df, 'predicted_rank_hybrid')
acc_odds_top3 = calculate_top3_accuracy(df, 'predicted_rank_odds')

print("Top1的中率（1着予測）:")
print(f"  LightGBM アンサンブル: {acc_ensemble_top1:.2%}")
print(f"  ハイブリッド:         {acc_hybrid_top1:.2%}")
print(f"  オッズのみ:           {acc_odds_top1:.2%}")
print()

print("Top3的中率（3着以内予測）:")
print(f"  LightGBM アンサンブル: {acc_ensemble_top3:.2%}")
print(f"  ハイブリッド:         {acc_hybrid_top3:.2%}")
print(f"  オッズのみ:           {acc_odds_top3:.2%}")
print()

# ===========================================================
# 結果保存
# ===========================================================
print("="*60)
print("結果保存")
print("="*60)
print()

# 予測結果をCSVに保存
output_file = "data/processed/predictions_with_hybrid.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"予測結果保存: {output_file}")

# サマリーをJSON保存
results_summary = {
    "model_type": "ensemble_hybrid",
    "ensemble_models": len(fold_models),
    "hybrid_weights": {
        "lightgbm_ensemble": 0.50,
        "odds_probability": 0.20,
        "statistical_score": 0.15,
        "logistic_regression": 0.15
    },
    "accuracy_metrics": {
        "top1_accuracy": {
            "ensemble": float(acc_ensemble_top1),
            "hybrid": float(acc_hybrid_top1),
            "odds_only": float(acc_odds_top1)
        },
        "top3_accuracy": {
            "ensemble": float(acc_ensemble_top3),
            "hybrid": float(acc_hybrid_top3),
            "odds_only": float(acc_odds_top3)
        }
    },
    "data_stats": {
        "total_records": len(df),
        "total_races": int(df['race_id'].nunique()),
        "features_used": available_features,
        "statistical_features": stat_features
    }
}

results_file = model_dir / "ensemble_hybrid_results.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print(f"サマリー保存: {results_file}")
print()

# ===========================================================
# サンプル予測表示
# ===========================================================
print("="*60)
print("サンプル予測（最新5レース）")
print("="*60)
print()

# 最新のレースIDを取得
latest_races = df.nlargest(5, 'race_date')['race_id'].unique()[:5]

for race_id in latest_races:
    race_data = df[df['race_id'] == race_id].copy()
    race_data = race_data.sort_values('hybrid_prediction', ascending=False)

    print(f"レースID: {race_id}")
    print(f"日付: {race_data.iloc[0]['race_date']}")
    print()
    print(f"{'順位':<4} {'実着順':<6} {'ハイブリッド予測':<15} {'オッズ':<8} {'人気':<6}")
    print("-" * 50)

    for idx, row in race_data.head(5).iterrows():
        print(f"{int(row['predicted_rank_hybrid']):<4} "
              f"{int(row['finish_position']):<6} "
              f"{row['hybrid_prediction']:<15.4f} "
              f"{row['odds']:<8.1f} "
              f"{int(row['popularity_rank']):<6}")
    print()

print("="*60)
print("完了!")
print("="*60)
print()
print("アンサンブル + 統計ハイブリッドモデルが完成しました。")
print()
print("主な特徴:")
print("  - 5つのLightGBMモデルのアンサンブル")
print("  - オッズベース確率モデル")
print("  - 統計的成績スコア")
print("  - ロジスティック回帰")
print("  - 4つの予測手法の重み付け統合")
print()
print("次のステップ:")
print("  - Streamlit Webアプリの開発")
print("  - リアルタイム予測機能の実装")

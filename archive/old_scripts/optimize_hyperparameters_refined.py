"""
LightGBM ハイパーパラメータ最適化 (高精度版)
- 500 trials
- 探索空間の細分化 (現在の最適値付近を集中探索)
- 多目的最適化 (NDCG@1/3/5の重み付け)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import optuna
from optuna.samplers import TPESampler
import json
from pathlib import Path

print("="*60)
print("LightGBM ハイパーパラメータ最適化 (高精度版)")
print("="*60)
print()

# データ読み込み
df = pd.read_csv("data/processed/features_engineered.csv", encoding="utf-8-sig")
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
print(f"使用特徴量: {len(available_features)}")

# 欠損値処理
missing_counts = df[available_features].isnull().sum()
if missing_counts.sum() > 0:
    df[available_features] = df[available_features].fillna(0)

X = df[available_features].values
y = df["finish_position"].values
groups = df["race_id"].values
weights = df["time_weight"].values

print(f"X shape: {X.shape}")
print(f"レース数: {len(np.unique(groups))}")
print()

# 前回の最適化結果を読み込み
baseline_results_file = Path("models/optimization_results.json")
if baseline_results_file.exists():
    with open(baseline_results_file, 'r') as f:
        baseline_data = json.load(f)
    baseline_ndcg3 = baseline_data['summary']['val_ndcg@3_mean']
    baseline_ndcg1 = baseline_data['summary']['val_ndcg@1_mean']
    baseline_ndcg5 = baseline_data['summary']['val_ndcg@5_mean']
    print(f"前回の最適化結果:")
    print(f"  NDCG@1: {baseline_ndcg1:.4f}")
    print(f"  NDCG@3: {baseline_ndcg3:.4f}")
    print(f"  NDCG@5: {baseline_ndcg5:.4f}")
    print()
else:
    baseline_ndcg1 = baseline_ndcg3 = baseline_ndcg5 = None


def objective(trial):
    """
    Optuna目的関数: 多目的最適化
    NDCG@1 (30%) + NDCG@3 (50%) + NDCG@5 (20%)
    """
    # 細分化された探索空間 (前回の最適値付近)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "boosting_type": "gbdt",
        "verbose": -1,

        # 最適化するパラメータ (範囲を絞る)
        "num_leaves": trial.suggest_int("num_leaves", 50, 100),
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.85, 0.99),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.85, 0.99),
        "bagging_freq": trial.suggest_int("bagging_freq", 4, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 15),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.3, 2.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 3.0, 10.0),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1.0, 4.0)
    }

    # 5-Fold Cross Validation
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    val_ndcg1_scores = []
    val_ndcg3_scores = []
    val_ndcg5_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train, groups_val = groups[train_idx], groups[val_idx]
        weights_train, weights_val = weights[train_idx], weights[val_idx]

        # グループサイズ計算
        train_groups_unique, train_group_sizes = np.unique(groups_train, return_counts=True)
        val_groups_unique, val_group_sizes = np.unique(groups_val, return_counts=True)
        train_group_dict = dict(zip(train_groups_unique, train_group_sizes))
        val_group_dict = dict(zip(val_groups_unique, val_group_sizes))
        train_group_list = [train_group_dict[g] for g in train_groups_unique]
        val_group_list = [val_group_dict[g] for g in val_groups_unique]

        # データセット作成
        train_data = lgb.Dataset(X_train, label=y_train, group=train_group_list, weight=weights_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=val_group_list, weight=weights_val, reference=train_data)

        # 訓練
        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.record_evaluation(evals_result)
            ]
        )

        # スコア取得
        best_iter = model.best_iteration
        val_ndcg1 = evals_result["valid"]["ndcg@1"][best_iter - 1]
        val_ndcg3 = evals_result["valid"]["ndcg@3"][best_iter - 1]
        val_ndcg5 = evals_result["valid"]["ndcg@5"][best_iter - 1]

        val_ndcg1_scores.append(val_ndcg1)
        val_ndcg3_scores.append(val_ndcg3)
        val_ndcg5_scores.append(val_ndcg5)

    # 多目的最適化: 重み付け平均
    mean_ndcg1 = np.mean(val_ndcg1_scores)
    mean_ndcg3 = np.mean(val_ndcg3_scores)
    mean_ndcg5 = np.mean(val_ndcg5_scores)

    # NDCG@1: 30%, NDCG@3: 50%, NDCG@5: 20%
    weighted_score = 0.3 * mean_ndcg1 + 0.5 * mean_ndcg3 + 0.2 * mean_ndcg5

    return weighted_score


print("="*60)
print("Optuna最適化開始 (高精度版)")
print("="*60)
print("試行回数: 500")
print("最適化メトリック: 重み付けNDCG (0.3*@1 + 0.5*@3 + 0.2*@5)")
print("探索空間: 前回最適値付近を集中探索")
print()
print("推定所要時間: 30-60分")
print()

# Optunaスタディ作成
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42)
)

# 最適化実行
study.optimize(objective, n_trials=500, show_progress_bar=True)

print()
print("="*60)
print("最適化完了")
print("="*60)
print()

# 最適パラメータ
best_params = study.best_params
best_score = study.best_value

print("最適ハイパーパラメータ:")
for key, value in best_params.items():
    print(f"  {key:20s}: {value}")

print()
print(f"最適スコア (重み付けNDCG): {best_score:.4f}")

# 最適パラメータで最終モデルを訓練（全データでCV）
print()
print("="*60)
print("最適パラメータで最終モデル訓練")
print("="*60)
print()

final_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "boosting_type": "gbdt",
    "verbose": -1,
    **best_params
}

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

cv_results = {
    "train_ndcg@1": [], "train_ndcg@3": [], "train_ndcg@5": [],
    "val_ndcg@1": [], "val_ndcg@3": [], "val_ndcg@5": [],
    "best_iterations": []
}

models = []
feature_importance_list = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    print(f"Fold {fold}/{n_splits}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train, groups_val = groups[train_idx], groups[val_idx]
    weights_train, weights_val = weights[train_idx], weights[val_idx]

    train_groups_unique, train_group_sizes = np.unique(groups_train, return_counts=True)
    val_groups_unique, val_group_sizes = np.unique(groups_val, return_counts=True)
    train_group_dict = dict(zip(train_groups_unique, train_group_sizes))
    val_group_dict = dict(zip(val_groups_unique, val_group_sizes))
    train_group_list = [train_group_dict[g] for g in train_groups_unique]
    val_group_list = [val_group_dict[g] for g in val_groups_unique]

    train_data = lgb.Dataset(X_train, label=y_train, group=train_group_list, weight=weights_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_group_list, weight=weights_val, reference=train_data)

    evals_result = {}
    model = lgb.train(
        final_params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.record_evaluation(evals_result)
        ]
    )

    models.append(model)
    best_iter = model.best_iteration
    cv_results["best_iterations"].append(best_iter)

    for metric in ["ndcg@1", "ndcg@3", "ndcg@5"]:
        train_score = evals_result["train"][metric][best_iter - 1]
        val_score = evals_result["valid"][metric][best_iter - 1]
        cv_results[f"train_{metric}"].append(train_score)
        cv_results[f"val_{metric}"].append(val_score)
        print(f"  {metric}: train={train_score:.4f}, val={val_score:.4f}")

    feature_importance_list.append(model.feature_importance(importance_type="gain"))
    print()

print("="*60)
print("最終CV結果")
print("="*60)
print()

final_ndcg1 = np.mean(cv_results["val_ndcg@1"])
final_ndcg3 = np.mean(cv_results["val_ndcg@3"])
final_ndcg5 = np.mean(cv_results["val_ndcg@5"])

for metric in ["ndcg@1", "ndcg@3", "ndcg@5"]:
    val_mean = np.mean(cv_results[f"val_{metric}"])
    val_std = np.std(cv_results[f"val_{metric}"])
    print(f"{metric}: {val_mean:.4f} +/- {val_std:.4f}")

# Top features
avg_importance = np.mean(feature_importance_list, axis=0)
importance_df = pd.DataFrame({
    "feature": available_features,
    "importance": avg_importance
}).sort_values("importance", ascending=False)

print()
print("Top 15 Features:")
for idx, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:,.0f}")

# モデル保存
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

best_fold_idx = np.argmax(cv_results["val_ndcg@3"])
best_model = models[best_fold_idx]
best_model.save_model(str(model_dir / "lightgbm_lambdarank_refined.txt"))

# すべてのFoldモデルも保存
for i, model in enumerate(models, 1):
    model.save_model(str(model_dir / f"lightgbm_lambdarank_refined_fold{i}.txt"))

# 結果保存
optimization_results = {
    "method": "refined_optimization",
    "optuna_n_trials": 500,
    "search_space": "narrowed around previous optimum",
    "objective": "weighted NDCG (0.3*@1 + 0.5*@3 + 0.2*@5)",
    "optuna_best_params": best_params,
    "optuna_best_weighted_score": float(best_score),
    "features": available_features,
    "final_params": final_params,
    "n_splits": n_splits,
    "cv_results": {k: [float(v) for v in vals] for k, vals in cv_results.items()},
    "best_fold": int(best_fold_idx + 1),
    "summary": {
        "val_ndcg@1_mean": float(final_ndcg1),
        "val_ndcg@3_mean": float(final_ndcg3),
        "val_ndcg@5_mean": float(final_ndcg5)
    }
}

# 前回との比較
if baseline_ndcg3 is not None:
    improvement_ndcg1 = (final_ndcg1 - baseline_ndcg1) / baseline_ndcg1 * 100
    improvement_ndcg3 = (final_ndcg3 - baseline_ndcg3) / baseline_ndcg3 * 100
    improvement_ndcg5 = (final_ndcg5 - baseline_ndcg5) / baseline_ndcg5 * 100

    optimization_results["comparison"] = {
        "previous_ndcg@1": float(baseline_ndcg1),
        "refined_ndcg@1": float(final_ndcg1),
        "improvement_ndcg@1_percent": float(improvement_ndcg1),
        "previous_ndcg@3": float(baseline_ndcg3),
        "refined_ndcg@3": float(final_ndcg3),
        "improvement_ndcg@3_percent": float(improvement_ndcg3),
        "previous_ndcg@5": float(baseline_ndcg5),
        "refined_ndcg@5": float(final_ndcg5),
        "improvement_ndcg@5_percent": float(improvement_ndcg5)
    }

with open(model_dir / "refined_optimization_results.json", "w", encoding="utf-8") as f:
    json.dump(optimization_results, f, indent=2, ensure_ascii=False)

print()
print("="*60)
print("保存完了")
print("="*60)
print("モデル: models/lightgbm_lambdarank_refined.txt")
print("結果: models/refined_optimization_results.json")
print()

if baseline_ndcg3 is not None:
    print("="*60)
    print("最終比較 (前回の最適化 vs 高精度版)")
    print("="*60)
    print(f"NDCG@1:")
    print(f"  前回:     {baseline_ndcg1:.4f}")
    print(f"  高精度版: {final_ndcg1:.4f}")
    print(f"  改善:     {improvement_ndcg1:+.2f}%")
    print()
    print(f"NDCG@3:")
    print(f"  前回:     {baseline_ndcg3:.4f}")
    print(f"  高精度版: {final_ndcg3:.4f}")
    print(f"  改善:     {improvement_ndcg3:+.2f}%")
    print()
    print(f"NDCG@5:")
    print(f"  前回:     {baseline_ndcg5:.4f}")
    print(f"  高精度版: {final_ndcg5:.4f}")
    print(f"  改善:     {improvement_ndcg5:+.2f}%")
    print("="*60)

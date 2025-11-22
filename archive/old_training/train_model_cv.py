"""
LightGBM LambdaRank with K-Fold Cross Validation
時系列重み付け + ハイパーパラメータチューニング
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import pickle
from pathlib import Path
import json

print("="*60)
print("LightGBM LambdaRank K-Fold Cross Validation")
print("="*60)
print()

df = pd.read_csv("data/processed/features_engineered.csv", encoding="utf-8-sig")
print(f"Data loaded: {len(df)} records")
print(f"Races: {df['race_id'].nunique()}")
print()

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
print(f"Features: {len(available_features)}")

missing_counts = df[available_features].isnull().sum()
if missing_counts.sum() > 0:
    df[available_features] = df[available_features].fillna(0)

X = df[available_features].values
y = df["finish_position"].values
groups = df["race_id"].values
weights = df["time_weight"].values

print(f"X shape: {X.shape}")
print(f"Unique races: {len(np.unique(groups))}")
print()

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "min_data_in_leaf": 20,
    "max_depth": 7
}

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

print("="*60)
print(f"{n_splits}-Fold Cross Validation")
print("="*60)
print()

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
    
    print(f"  Train: {len(X_train)} ({len(np.unique(groups_train))} races)")
    print(f"  Val: {len(X_val)} ({len(np.unique(groups_val))} races)")
    
    train_groups_unique, train_group_sizes = np.unique(groups_train, return_counts=True)
    val_groups_unique, val_group_sizes = np.unique(groups_val, return_counts=True)
    train_group_dict = dict(zip(train_groups_unique, train_group_sizes))
    val_group_dict = dict(zip(val_groups_unique, val_group_sizes))
    train_group_list = [train_group_dict[g] for g in train_groups_unique]
    val_group_list = [val_group_dict[g] for g in val_groups_unique]
    
    train_data = lgb.Dataset(X_train, label=y_train, group=train_group_list, weight=weights_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_group_list, weight=weights_val, reference=train_data)
    
    evals_result = {}
    model = lgb.train(params, train_data, num_boost_round=500,
        valid_sets=[train_data, val_data], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.record_evaluation(evals_result)])
    
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
print("CV Results Summary")
print("="*60)
print()

for metric in ["ndcg@1", "ndcg@3", "ndcg@5"]:
    val_mean = np.mean(cv_results[f"val_{metric}"])
    val_std = np.std(cv_results[f"val_{metric}"])
    print(f"{metric}: {val_mean:.4f} +/- {val_std:.4f}")

avg_importance = np.mean(feature_importance_list, axis=0)
importance_df = pd.DataFrame({"feature": available_features, "importance": avg_importance}).sort_values("importance", ascending=False)

print()
print("Top 15 Features:")
for idx, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:,.0f}")

best_fold_idx = np.argmax(cv_results["val_ndcg@3"])
best_model = models[best_fold_idx]

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
best_model.save_model(str(model_dir / "lightgbm_lambdarank_best.txt"))

for i, model in enumerate(models, 1):
    model.save_model(str(model_dir / f"lightgbm_lambdarank_fold{i}.txt"))

metadata = {
    "features": available_features,
    "params": params,
    "n_splits": n_splits,
    "cv_results": {k: [float(v) for v in vals] for k, vals in cv_results.items()},
    "best_fold": int(best_fold_idx + 1),
    "summary": {
        f"val_{m}_mean": float(np.mean(cv_results[f"val_{m}"])) 
        for m in ["ndcg@1", "ndcg@3", "ndcg@5"]
    }
}

with open(model_dir / "cv_results.json", "w") as f:
    json.dump(metadata, f, indent=2)

print()
print("Training completed!")

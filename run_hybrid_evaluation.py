"""
ハイブリッドシステムの実行と評価
Dynamic Alpha Tuning + 複数の重み設定で性能比較
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SimpleModelWrapper:
    """シンプルなモデルラッパー"""

    def __init__(self, model_type, model_path):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None

    def load(self):
        """モデル読み込み"""
        if self.model_type == "lightgbm":
            self.model = lgb.Booster(model_file=self.model_path)
        elif self.model_type == "statistical":
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
        else:
            logging.warning(f"未対応のモデルタイプ: {self.model_type}")

    def predict(self, X):
        """予測"""
        if self.model is None:
            self.load()

        if self.model_type == "lightgbm":
            return self.model.predict(X)
        elif self.model_type == "statistical":
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            return np.random.rand(len(X))


def normalize_scores(scores, method='minmax'):
    """スコア正規化"""
    if method == 'minmax':
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return np.ones_like(scores) * 0.5
        return (scores - min_val) / (max_val - min_val)
    elif method == 'softmax':
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()
    else:
        return scores


def estimate_dynamic_alpha(race_data, model_confidences):
    """Dynamic Alpha Tuning"""
    # レース特性抽出
    field_size = len(race_data)
    avg_odds = race_data['odds'].mean() if 'odds' in race_data.columns else 10.0

    # モデルの適合度評価
    suitability = {}

    # LightGBM: 大規模レースに強い
    suitability['lightgbm'] = 0.5
    if field_size > 16:
        suitability['lightgbm'] += 0.2

    # Statistical: 小規模レース、データ不足
    suitability['statistical'] = 0.3
    if field_size < 12:
        suitability['statistical'] += 0.3

    # スコア計算
    scores = {}
    for model_name in model_confidences:
        confidence = model_confidences[model_name]
        suit = suitability.get(model_name, 0.5)
        scores[model_name] = confidence * suit

    # Softmaxで正規化
    scores_array = np.array(list(scores.values()))
    exp_scores = np.exp(scores_array - scores_array.max())
    alphas_array = exp_scores / exp_scores.sum()

    alphas = {model: alpha for model, alpha in zip(scores.keys(), alphas_array)}

    return alphas


def hybrid_predict(models, race_data, weights, use_dat=False):
    """ハイブリッド予測"""
    predictions = {}
    confidences = {}

    # 各モデルで予測
    for model_name, model in models.items():
        try:
            # 特徴量選択（モデルごとに異なる可能性）
            if model_name == "lightgbm":
                # LightGBMの特徴量
                feature_cols = [col for col in race_data.columns
                               if col not in ['race_id', 'horse_id', 'finish_position',
                                            'finish_position_numeric', 'horse_name']]
                X = race_data[feature_cols].fillna(0)
            elif model_name == "statistical":
                # 統計モデルの特徴量
                feature_cols = ['weight', 'age']
                X = race_data[[col for col in feature_cols if col in race_data.columns]].fillna(0)
            else:
                X = race_data.select_dtypes(include=[np.number]).fillna(0)

            preds = model.predict(X)
            preds_norm = normalize_scores(preds, method='minmax')

            predictions[model_name] = preds_norm

            # 信頼度（スコアの分散から計算）
            variance = preds_norm.std()
            confidences[model_name] = 1.0 / (1.0 + variance)

        except Exception as e:
            logging.warning(f"モデル {model_name} の予測失敗: {e}")
            predictions[model_name] = np.zeros(len(race_data))
            confidences[model_name] = 0.0

    # 重み決定
    if use_dat:
        weights = estimate_dynamic_alpha(race_data, confidences)
        logging.debug(f"DAT重み: {weights}")

    # 重み付け結合
    combined_scores = np.zeros(len(race_data))
    for model_name, preds in predictions.items():
        weight = weights.get(model_name, 0.0)
        combined_scores += weight * preds

    return combined_scores, weights


def evaluate_hybrid(df, models, weights, use_dat=False):
    """ハイブリッドモデル評価"""
    logging.info(f"\n評価開始: DAT={'ON' if use_dat else 'OFF'}, 重み={weights}")

    all_predictions = []
    all_labels = []
    race_weights_history = []

    # レースごとに予測
    for race_id in df['race_id'].unique():
        race_data = df[df['race_id'] == race_id].copy()

        if len(race_data) < 2:
            continue

        # ハイブリッド予測
        predictions, actual_weights = hybrid_predict(
            models, race_data, weights, use_dat=use_dat
        )

        labels = race_data['finish_position_numeric'].values if 'finish_position_numeric' in race_data.columns else race_data['finish_position'].values

        all_predictions.append(predictions)
        all_labels.append(labels)

        if use_dat:
            race_weights_history.append(actual_weights)

    # メトリクス計算
    metrics = {}

    ndcg_scores = []
    top1_correct = 0
    top3_correct = 0
    total_races = 0

    for preds, labels in zip(all_predictions, all_labels):
        if len(preds) < 2:
            continue

        # NDCG
        relevance = 20 - labels
        try:
            ndcg1 = ndcg_score([relevance], [preds], k=1)
            ndcg3 = ndcg_score([relevance], [preds], k=3)
            ndcg_scores.append((ndcg1, ndcg3))
        except:
            pass

        # Top1
        pred_winner = np.argmax(preds)
        actual_winner = np.argmin(labels)
        if pred_winner == actual_winner:
            top1_correct += 1

        # Top3
        if len(preds) >= 3:
            pred_top3 = set(np.argsort(preds)[-3:])
            actual_top3 = set(np.argsort(labels)[:3])
            if len(pred_top3 & actual_top3) > 0:
                top3_correct += 1

        total_races += 1

    if ndcg_scores:
        metrics['ndcg@1'] = np.mean([s[0] for s in ndcg_scores])
        metrics['ndcg@3'] = np.mean([s[1] for s in ndcg_scores])

    if total_races > 0:
        metrics['top1_accuracy'] = top1_correct / total_races
        metrics['top3_hit_rate'] = top3_correct / total_races
        metrics['total_races'] = total_races

    # DAT使用時は平均重みも記録
    if use_dat and race_weights_history:
        avg_weights = {}
        for model_name in weights.keys():
            avg_weights[model_name] = np.mean([w.get(model_name, 0) for w in race_weights_history])
        metrics['avg_dat_weights'] = avg_weights

    return metrics


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("ハイブリッドシステム評価")
    logging.info("="*60)

    # データ読み込み
    train_file = Path("data/processed/keibalab_g1_2000_2024_processed.csv")
    if not train_file.exists():
        logging.error(f"データが見つかりません: {train_file}")
        return

    logging.info(f"\nデータ読み込み: {train_file}")
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    logging.info(f"  レコード数: {len(df)}")
    logging.info(f"  レース数: {df['race_id'].nunique()}")

    # サンプリング（高速化のため）
    sample_races = df['race_id'].unique()[:500]  # 最初の500レース
    df_sample = df[df['race_id'].isin(sample_races)].copy()
    logging.info(f"  サンプル: {len(df_sample)} records, {df_sample['race_id'].nunique()} races")

    # モデル読み込み
    logging.info("\nモデル読み込み...")
    models = {}

    # LightGBM
    lightgbm_path = Path("models/optimized/optimized_model_fold1.txt")
    if lightgbm_path.exists():
        models['lightgbm'] = SimpleModelWrapper('lightgbm', str(lightgbm_path))
        logging.info("  ✓ LightGBM")

    # Conditional Logit
    logit_models = list(Path("models/statistical").glob("conditional_logit_*.pkl"))
    if logit_models:
        models['statistical'] = SimpleModelWrapper('statistical', str(logit_models[-1]))
        logging.info("  ✓ Conditional Logit")

    if len(models) < 2:
        logging.error("モデルが不足しています（最低2つ必要）")
        return

    # 評価実行
    logging.info("\n" + "="*60)
    logging.info("評価実行")
    logging.info("="*60)

    results = []

    # 1. 固定重み（均等）
    weights_equal = {name: 1.0 / len(models) for name in models}
    metrics_equal = evaluate_hybrid(df_sample, models, weights_equal, use_dat=False)
    results.append({
        'method': '固定重み（均等）',
        'weights': weights_equal,
        'metrics': metrics_equal
    })

    # 2. 固定重み（LightGBM優先）
    weights_lgb_heavy = {'lightgbm': 0.7, 'statistical': 0.3}
    metrics_lgb = evaluate_hybrid(df_sample, models, weights_lgb_heavy, use_dat=False)
    results.append({
        'method': '固定重み（LightGBM 70%）',
        'weights': weights_lgb_heavy,
        'metrics': metrics_lgb
    })

    # 3. 固定重み（統計モデル優先）
    weights_stat_heavy = {'lightgbm': 0.3, 'statistical': 0.7}
    metrics_stat = evaluate_hybrid(df_sample, models, weights_stat_heavy, use_dat=False)
    results.append({
        'method': '固定重み（Statistical 70%）',
        'weights': weights_stat_heavy,
        'metrics': metrics_stat
    })

    # 4. Dynamic Alpha Tuning
    weights_initial = {name: 1.0 / len(models) for name in models}
    metrics_dat = evaluate_hybrid(df_sample, models, weights_initial, use_dat=True)
    results.append({
        'method': 'Dynamic Alpha Tuning',
        'weights': weights_initial,
        'metrics': metrics_dat
    })

    # 結果サマリー
    logging.info("\n" + "="*60)
    logging.info("結果サマリー")
    logging.info("="*60)

    # 表形式で表示
    print("\n| 手法 | NDCG@1 | NDCG@3 | Top1精度 | Top3適中率 |")
    print("|------|--------|--------|----------|-----------|")

    for result in results:
        method = result['method']
        metrics = result['metrics']

        ndcg1 = metrics.get('ndcg@1', 0)
        ndcg3 = metrics.get('ndcg@3', 0)
        top1 = metrics.get('top1_accuracy', 0)
        top3 = metrics.get('top3_hit_rate', 0)

        print(f"| {method} | {ndcg1:.4f} | {ndcg3:.4f} | {top1:.2%} | {top3:.2%} |")

    # 詳細をJSONで保存
    output_dir = Path("models/hybrid_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"hybrid_results_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logging.info(f"\n✓ 結果保存: {output_file}")

    # ベスト手法
    best_result = max(results, key=lambda r: r['metrics'].get('ndcg@3', 0))
    logging.info("\n" + "="*60)
    logging.info("ベスト手法")
    logging.info("="*60)
    logging.info(f"手法: {best_result['method']}")
    logging.info(f"NDCG@3: {best_result['metrics'].get('ndcg@3', 0):.4f}")
    logging.info(f"Top1精度: {best_result['metrics'].get('top1_accuracy', 0):.2%}")

    if 'avg_dat_weights' in best_result['metrics']:
        logging.info(f"平均重み: {best_result['metrics']['avg_dat_weights']}")


if __name__ == "__main__":
    main()

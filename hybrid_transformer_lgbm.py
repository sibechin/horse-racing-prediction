"""
LightGBM + Transformer ハイブリッドモデル

2つのモデルを融合してTOP1予測精度を向上
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import lightgbm as lgb
from datetime import datetime
import json
from transformer_top1_v2 import TransformerTop1TrainerV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridTransformerLGBM:
    """LightGBM + Transformer ハイブリッド予測システム"""

    def __init__(self):
        self.lgbm_model = None
        self.transformer_trainer = None
        self.lgbm_weight = 0.5
        self.transformer_weight = 0.5

        self.feature_cols = [
            'year', 'month', 'day_of_week', 'venue_code', 'race_number',
            'sex_encoded', 'age', 'weight', 'last_3f', 'horse_weight_kg',
            'horse_weight_change', 'field_size', 'race_avg_odds', 'race_avg_weight',
            'trainer_encoded', 'trainer_frequency', 'jockey_encoded', 'jockey_frequency',
            'race_type_encoded'
        ]

    def load_models(self, lgbm_path, transformer_path):
        """モデル読み込み"""
        logging.info("モデル読み込み中...")

        # LightGBM
        self.lgbm_model = lgb.Booster(model_file=str(lgbm_path))
        logging.info(f"  LightGBM: {lgbm_path}")

        # Transformer
        self.transformer_trainer = TransformerTop1TrainerV2.load(transformer_path)
        logging.info(f"  Transformer: {transformer_path}")

    def predict_lgbm(self, race_data):
        """LightGBM予測"""
        available = [c for c in self.feature_cols if c in race_data.columns]
        X = race_data[available]
        return self.lgbm_model.predict(X)

    def predict_transformer(self, race_data):
        """Transformer予測"""
        return self.transformer_trainer.predict_race(race_data)

    def normalize_scores(self, scores):
        """スコアを0-1に正規化"""
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s > 0:
            return (scores - min_s) / (max_s - min_s)
        return np.ones_like(scores) / len(scores)

    def predict(self, race_data, method='weighted'):
        """
        ハイブリッド予測

        Args:
            race_data: 1レースのデータ
            method: 'weighted', 'max', 'rank'
        """
        lgbm_scores = self.predict_lgbm(race_data)
        trans_scores = self.predict_transformer(race_data)

        # 正規化
        lgbm_norm = self.normalize_scores(lgbm_scores)
        trans_norm = self.normalize_scores(trans_scores)

        if method == 'weighted':
            # 重み付き平均
            return self.lgbm_weight * lgbm_norm + self.transformer_weight * trans_norm

        elif method == 'max':
            # 各馬の最大スコア
            return np.maximum(lgbm_norm, trans_norm)

        elif method == 'rank':
            # ランク融合（Borda Count）
            lgbm_rank = np.argsort(np.argsort(-lgbm_scores))
            trans_rank = np.argsort(np.argsort(-trans_scores))
            return -(lgbm_rank + trans_rank)  # 負にして高い方が良い

        return lgbm_norm + trans_norm

    def tune_weights(self, df_train, n_samples=200):
        """検証データで最適な重みを探索"""
        logging.info("重み最適化中...")

        race_ids = df_train['race_id'].unique()
        np.random.shuffle(race_ids)
        sample_races = race_ids[:min(n_samples, len(race_ids))]

        best_acc = 0
        best_weights = (0.5, 0.5)

        for lgbm_w in np.arange(0.1, 1.0, 0.1):
            trans_w = 1.0 - lgbm_w
            self.lgbm_weight = lgbm_w
            self.transformer_weight = trans_w

            correct = 0
            total = 0

            for race_id in sample_races:
                race_data = df_train[df_train['race_id'] == race_id]
                if len(race_data) < 2:
                    continue

                try:
                    scores = self.predict(race_data, method='weighted')
                    positions = race_data['finish_position_numeric'].values

                    pred_winner = np.argmax(scores)
                    actual_winner = np.argmin(positions)

                    if pred_winner == actual_winner:
                        correct += 1
                    total += 1
                except:
                    continue

            acc = correct / total if total > 0 else 0

            if acc > best_acc:
                best_acc = acc
                best_weights = (lgbm_w, trans_w)

        self.lgbm_weight, self.transformer_weight = best_weights
        logging.info(f"最適重み: LightGBM={self.lgbm_weight:.1f}, Transformer={self.transformer_weight:.1f}")
        logging.info(f"検証精度: {best_acc:.2%}")

        return best_weights

    def evaluate(self, df_test):
        """テストデータで評価"""
        results = {'weighted': {'top1': 0, 'top3': 0},
                   'max': {'top1': 0, 'top3': 0},
                   'rank': {'top1': 0, 'top3': 0},
                   'lgbm_only': {'top1': 0, 'top3': 0},
                   'transformer_only': {'top1': 0, 'top3': 0}}

        total = 0

        for race_id in df_test['race_id'].unique():
            race_data = df_test[df_test['race_id'] == race_id]
            positions = race_data['finish_position_numeric'].values
            actual_winner = np.argmin(positions)
            actual_top3 = set(np.argsort(positions)[:3])

            try:
                # 各方法で予測
                for method in ['weighted', 'max', 'rank']:
                    scores = self.predict(race_data, method=method)
                    pred_winner = np.argmax(scores)
                    pred_top3 = set(np.argsort(scores)[-3:])

                    if pred_winner == actual_winner:
                        results[method]['top1'] += 1
                    if len(pred_top3 & actual_top3) > 0:
                        results[method]['top3'] += 1

                # 単体モデル
                lgbm_scores = self.normalize_scores(self.predict_lgbm(race_data))
                trans_scores = self.normalize_scores(self.predict_transformer(race_data))

                if np.argmax(lgbm_scores) == actual_winner:
                    results['lgbm_only']['top1'] += 1
                if len(set(np.argsort(lgbm_scores)[-3:]) & actual_top3) > 0:
                    results['lgbm_only']['top3'] += 1

                if np.argmax(trans_scores) == actual_winner:
                    results['transformer_only']['top1'] += 1
                if len(set(np.argsort(trans_scores)[-3:]) & actual_top3) > 0:
                    results['transformer_only']['top3'] += 1

                total += 1
            except Exception as e:
                logging.warning(f"Race {race_id}: {e}")

        # 結果表示
        logging.info(f"\n{'='*60}")
        logging.info(f"評価結果 ({total}レース)")
        logging.info(f"{'='*60}")

        for method, metrics in results.items():
            top1_acc = metrics['top1'] / total if total > 0 else 0
            top3_acc = metrics['top3'] / total if total > 0 else 0
            logging.info(f"{method:20s}: Top1={top1_acc:.2%}, Top3={top3_acc:.2%}")

        return results, total


def main():
    logging.info("="*60)
    logging.info("LightGBM + Transformer ハイブリッドモデル")
    logging.info("="*60)

    # データ読み込み
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")

    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')

    for df in [df_train, df_test]:
        if 'finish_position_numeric' not in df.columns:
            df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

    # モデルパス
    lgbm_path = Path("models/top3_no_odds/top3_no_odds_model_20251121_053558.txt")

    transformer_dir = Path("models/transformer_top1_v2")
    transformer_files = list(transformer_dir.glob("transformer_top1_v2_*.pth"))
    transformer_path = max(transformer_files, key=lambda p: p.stat().st_mtime)

    # ハイブリッドシステム
    system = HybridTransformerLGBM()
    system.load_models(lgbm_path, transformer_path)

    # 重み最適化
    system.tune_weights(df_train, n_samples=300)

    # 評価
    results, total = system.evaluate(df_test)

    # 結果保存
    output_dir = Path("models/hybrid_transformer_lgbm")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    save_results = {
        'weights': {'lgbm': system.lgbm_weight, 'transformer': system.transformer_weight},
        'results': {k: {m: v/total for m, v in metrics.items()} for k, metrics in results.items()},
        'total_races': total
    }

    with open(output_dir / f"results_{timestamp}.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    logging.info(f"\n結果保存: {output_dir}")


if __name__ == "__main__":
    main()

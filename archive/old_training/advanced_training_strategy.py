"""
精度最優先の高度な訓練戦略

データ量増加に伴う最適化:
1. 過学習リスク評価
2. 最適なデータ分割
3. 高度なアンサンブル
4. ファインチューニング戦略
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import ndcg_score
import optuna
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AdvancedTrainingStrategy:
    """精度最優先の訓練戦略"""

    def __init__(self, data_path: str, baseline_path: str = None):
        self.data = pd.read_csv(data_path, encoding='utf-8-sig')
        self.baseline_results = None

        if baseline_path and Path(baseline_path).exists():
            with open(baseline_path, 'r') as f:
                self.baseline_results = json.load(f)

        logging.info(f"データ読み込み: {len(self.data)}レコード")

    def analyze_data_scale_effect(self):
        """データ量と精度の関係を分析"""
        logging.info("=" * 60)
        logging.info("データスケール効果分析")
        logging.info("=" * 60)

        # データサイズの段階的テスト
        sizes = [1000, 2000, 3000, 4000, 5000, len(self.data)]
        results = []

        for size in sizes:
            if size > len(self.data):
                continue

            logging.info(f"\nデータサイズ: {size}レコードでテスト")

            # サンプリング (時系列順を維持)
            df_sample = self.data.sort_values('race_date').tail(size)

            # 簡易訓練
            score = self._quick_train(df_sample)
            results.append({
                'size': size,
                'ndcg@1': score,
                'records': len(df_sample),
                'races': df_sample['race_id'].nunique()
            })

            logging.info(f"  NDCG@1: {score:.4f}")

        # 結果サマリー
        results_df = pd.DataFrame(results)
        logging.info("\n" + "=" * 60)
        logging.info("データサイズと精度の関係")
        logging.info("=" * 60)
        print(results_df.to_string(index=False))

        # Learning curve分析
        if len(results) >= 3:
            # 最後の改善率を計算
            last_improvement = (results[-1]['ndcg@1'] - results[-2]['ndcg@1']) / results[-2]['ndcg@1'] * 100
            logging.info(f"\n最新の改善率: {last_improvement:.2f}%")

            if last_improvement < 1.0:
                logging.warning("⚠️ 精度の伸びが鈍化 - データ追加の効果が限定的")
            else:
                logging.info("✅ データ追加により精度向上継続中")

        return results_df

    def _quick_train(self, df):
        """簡易訓練（データスケール分析用）"""
        # 特徴量選択
        feature_cols = [col for col in df.columns if col not in [
            'race_id', 'race_date', 'horse_name', 'horse_id', 'jockey_id',
            'finish_position', 'venue_name', 'time', 'margin', 'passing_order'
        ]]

        # 欠損値処理
        df_clean = df.copy()
        for col in feature_cols:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # 時系列分割 (2-fold)
        unique_races = df_clean['race_id'].unique()
        split_point = len(unique_races) * 2 // 3

        train_races = unique_races[:split_point]
        val_races = unique_races[split_point:]

        train_data = df_clean[df_clean['race_id'].isin(train_races)]
        val_data = df_clean[df_clean['race_id'].isin(val_races)]

        # グループ作成
        train_groups = train_data.groupby('race_id').size().values
        val_groups = val_data.groupby('race_id').size().values

        # LightGBM訓練
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'num_leaves': 63,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'verbose': -1,
            'num_threads': -1
        }

        train_set = lgb.Dataset(
            train_data[feature_cols],
            label=train_data['finish_position'],
            group=train_groups
        )

        val_set = lgb.Dataset(
            val_data[feature_cols],
            label=val_data['finish_position'],
            group=val_groups,
            reference=train_set
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=100,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )

        # 評価
        val_pred = model.predict(val_data[feature_cols])

        ndcg_scores = []
        for race_id in val_races:
            race_mask = val_data['race_id'] == race_id
            if race_mask.sum() > 1:
                y_true = val_data[race_mask]['finish_position'].values
                y_pred = val_pred[race_mask]

                # NDCG@1計算
                ideal = np.sort(y_true)
                actual_ranks = np.argsort(y_pred)

                dcg = (2 ** y_true[actual_ranks[0]] - 1) / np.log2(2)
                idcg = (2 ** ideal[0] - 1) / np.log2(2)

                if idcg > 0:
                    ndcg_scores.append(dcg / idcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def design_optimal_cv_strategy(self):
        """最適なCV戦略を設計"""
        logging.info("=" * 60)
        logging.info("CV戦略最適化")
        logging.info("=" * 60)

        total_races = self.data['race_id'].nunique()
        total_records = len(self.data)

        logging.info(f"総レース数: {total_races}")
        logging.info(f"総レコード数: {total_records}")
        logging.info(f"1レースあたり平均: {total_records/total_races:.1f}頭")

        # 過学習リスク評価
        logging.info("\n過学習リスク評価:")

        # データ/パラメータ比率
        estimated_params = 1000  # LightGBMの概算パラメータ数
        data_param_ratio = total_records / estimated_params

        logging.info(f"  データ/パラメータ比率: {data_param_ratio:.1f}")

        if data_param_ratio < 10:
            logging.warning("  ⚠️ 高リスク - データ量に対してモデルが複雑")
            recommended_folds = 10
        elif data_param_ratio < 30:
            logging.warning("  ⚠️ 中リスク - 正則化強化推奨")
            recommended_folds = 7
        else:
            logging.info("  ✅ 低リスク - 十分なデータ量")
            recommended_folds = 5

        logging.info(f"\n推奨Fold数: {recommended_folds}")

        # 時系列考慮の重要性
        date_range = pd.to_datetime(self.data['race_date']).max() - pd.to_datetime(self.data['race_date']).min()
        logging.info(f"データ期間: {date_range.days}日")

        if date_range.days > 365:
            logging.info("  ✅ 時系列分割を推奨 (長期データ)")
            use_time_series_split = True
        else:
            logging.info("  GroupKFoldで十分")
            use_time_series_split = False

        return {
            'recommended_folds': recommended_folds,
            'use_time_series_split': use_time_series_split,
            'overfitting_risk': 'high' if data_param_ratio < 10 else 'medium' if data_param_ratio < 30 else 'low'
        }

    def recommend_regularization_strategy(self):
        """正則化戦略を推奨"""
        logging.info("=" * 60)
        logging.info("正則化戦略")
        logging.info("=" * 60)

        feature_count = len([col for col in self.data.columns if col not in [
            'race_id', 'race_date', 'horse_name', 'horse_id', 'jockey_id',
            'finish_position', 'venue_name', 'time', 'margin', 'passing_order'
        ]])

        logging.info(f"特徴量数: {feature_count}")

        recommendations = []

        # L1/L2正則化
        recommendations.append({
            'method': 'L1正則化 (lambda_l1)',
            'value': 'Optuna最適化範囲: 0.1-10',
            'purpose': '不要な特徴量を自動除去'
        })

        recommendations.append({
            'method': 'L2正則化 (lambda_l2)',
            'value': 'Optuna最適化範囲: 1-50',
            'purpose': '重みの過剰な増大を抑制'
        })

        # Feature fraction
        if feature_count > 30:
            recommendations.append({
                'method': 'Feature Fraction',
                'value': '0.6-0.8 (特徴量多い)',
                'purpose': 'ランダム性導入で過学習防止'
            })
        else:
            recommendations.append({
                'method': 'Feature Fraction',
                'value': '0.8-0.95 (特徴量少ない)',
                'purpose': '情報量を維持'
            })

        # Bagging
        recommendations.append({
            'method': 'Bagging Fraction',
            'value': '0.7-0.9',
            'purpose': 'データのランダムサンプリング'
        })

        # Early stopping
        recommendations.append({
            'method': 'Early Stopping',
            'value': 'Patience: 50-100',
            'purpose': '過学習の兆候で訓練停止'
        })

        # Max depth
        recommendations.append({
            'method': 'Max Depth',
            'value': '4-8',
            'purpose': '木の深さ制限で複雑度抑制'
        })

        logging.info("\n推奨正則化パラメータ:")
        for rec in recommendations:
            logging.info(f"  • {rec['method']}: {rec['value']}")
            logging.info(f"    目的: {rec['purpose']}")

        return recommendations

    def suggest_training_pipeline(self):
        """訓練パイプライン提案"""
        logging.info("=" * 60)
        logging.info("推奨訓練パイプライン")
        logging.info("=" * 60)

        pipeline = [
            "1. データ前処理",
            "   - 障害レース除外",
            "   - 異常値除去",
            "   - 欠損値補完",
            "",
            "2. 特徴量エンジニアリング",
            "   - 既存特徴量 + 新データベース特徴量",
            "   - Feature importance分析",
            "   - 低重要度特徴量の除去 (閾値: 0.1%未満)",
            "",
            "3. データ分割戦略",
            "   - 時系列考慮のGroupKFold (推奨7-10 Folds)",
            "   - または TimeSeriesSplit (長期データの場合)",
            "",
            "4. ハイパーパラメータ最適化 (Optuna)",
            "   - Trials: 200-300 (精度優先)",
            "   - 目的関数: NDCG@1 (最優先)",
            "   - 探索範囲:",
            "     • num_leaves: 31-127",
            "     • max_depth: 4-8",
            "     • learning_rate: 0.001-0.05",
            "     • lambda_l1: 0.1-10",
            "     • lambda_l2: 1-50",
            "     • feature_fraction: 0.6-0.9",
            "     • bagging_fraction: 0.7-0.9",
            "",
            "5. アンサンブル戦略",
            "   - 各Foldのモデルを保持",
            "   - 予測時は全モデルの平均",
            "   - 重み付けアンサンブル (Validation性能基準)",
            "",
            "6. ハイブリッド予測",
            "   - LightGBM: 60% (データ量増加で重み増)",
            "   - オッズ: 15% (既存20%から削減)",
            "   - 統計: 15%",
            "   - Logistic Regression: 10%",
            "",
            "7. 過学習検証",
            "   - Train vs Validation NDCGギャップ監視",
            "   - 許容ギャップ: <5%",
            "   - Learning curve可視化",
            "",
            "8. 最終評価",
            "   - Hold-out test set (2024年最新データ)",
            "   - Baseline比較",
            "   - Feature importance分析",
            "   - エラー分析 (外れ値レース特定)"
        ]

        for line in pipeline:
            logging.info(line)

        # 推定所要時間
        logging.info("\n" + "=" * 60)
        logging.info("推定所要時間")
        logging.info("=" * 60)

        data_size = len(self.data)
        if data_size < 5000:
            time_estimate = "2-3時間"
        elif data_size < 10000:
            time_estimate = "3-5時間"
        else:
            time_estimate = "5-7時間"

        logging.info(f"データサイズ: {data_size}レコード")
        logging.info(f"推定時間: {time_estimate}")
        logging.info("  内訳:")
        logging.info("    - 前処理: 10-15分")
        logging.info("    - 特徴量エンジニアリング: 15-20分")
        logging.info("    - Optuna最適化: 60-180分 (最大)")
        logging.info("    - CV訓練: 30-60分")
        logging.info("    - 評価・検証: 15-20分")


def main():
    """メイン実行"""

    # データパス
    data_path = "data/processed/cleaned_data_all.csv"

    # まだデータがない場合
    if not Path(data_path).exists():
        logging.warning(f"{data_path} が存在しません")
        logging.info("既存のmerged_data_all.csvを使用します")
        data_path = "data/processed/merged_data_all.csv"

    if not Path(data_path).exists():
        logging.error("データファイルが見つかりません")
        return

    # Baseline結果
    baseline_path = "models/baseline_backup_v1/refined_optimization_results.json"

    # 戦略インスタンス
    strategy = AdvancedTrainingStrategy(data_path, baseline_path)

    # 1. データスケール効果分析
    scale_results = strategy.analyze_data_scale_effect()

    # 2. CV戦略最適化
    cv_strategy = strategy.design_optimal_cv_strategy()

    # 3. 正則化戦略
    regularization = strategy.recommend_regularization_strategy()

    # 4. 訓練パイプライン提案
    strategy.suggest_training_pipeline()

    # 結果保存
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)

    # サマリー保存
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_size': len(strategy.data),
        'cv_strategy': cv_strategy,
        'regularization_recommendations': regularization,
        'scale_analysis': scale_results.to_dict('records')
    }

    with open(output_dir / 'training_strategy_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"\n戦略サマリー保存: {output_dir / 'training_strategy_summary.json'}")


if __name__ == "__main__":
    main()

"""
モデル比較評価フレームワーク

全モデルを統一的なインターフェースでテストし、性能を比較する
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import torch
import torch.nn as nn

from model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelWrapper:
    """モデルラッパー - 統一インターフェース"""

    def __init__(self, model_id: str, model_type: str, model_path: str):
        self.model_id = model_id
        self.model_type = model_type
        self.model_path = model_path
        self.model = None

    def load(self):
        """モデル読み込み"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, race_groups: pd.Series) -> np.ndarray:
        """
        予測実行

        Args:
            X: 特徴量
            race_groups: レースID

        Returns:
            予測スコア（各馬のスコア）
        """
        raise NotImplementedError


class LightGBMWrapper(ModelWrapper):
    """LightGBMモデルラッパー"""

    def load(self):
        """モデル読み込み"""
        try:
            self.model = lgb.Booster(model_file=self.model_path)
            logging.info(f"✓ LightGBMモデル読み込み: {self.model_path}")
        except Exception as e:
            logging.error(f"✗ モデル読み込みエラー: {e}")
            raise

    def predict(self, X: pd.DataFrame, race_groups: pd.Series) -> np.ndarray:
        """予測"""
        if self.model is None:
            self.load()

        predictions = self.model.predict(X)
        return predictions


class TransformerWrapper(ModelWrapper):
    """Transformerモデルラッパー"""

    def load(self):
        """モデル読み込み"""
        # PyTorchモデルの読み込み実装
        pass

    def predict(self, X: pd.DataFrame, race_groups: pd.Series) -> np.ndarray:
        """予測"""
        # 実装予定
        pass


class ModelComparator:
    """モデル比較器"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.models = {}

    def load_model(self, model_id: str) -> ModelWrapper:
        """モデル読み込み"""
        model_info = self.registry.get_model(model_id)
        if not model_info:
            raise ValueError(f"モデルが見つかりません: {model_id}")

        model_type = model_info["model_type"]
        model_path = model_info["model_path"]

        # モデルタイプに応じてラッパー作成
        if model_type == "LightGBM":
            wrapper = LightGBMWrapper(model_id, model_type, model_path)
        elif model_type == "Transformer":
            wrapper = TransformerWrapper(model_id, model_type, model_path)
        elif model_type == "Attention":
            wrapper = TransformerWrapper(model_id, model_type, model_path)
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")

        wrapper.load()
        self.models[model_id] = wrapper

        return wrapper

    def evaluate_model(
        self,
        model_id: str,
        test_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "finish_position",
        race_id_col: str = "race_id"
    ) -> Dict[str, float]:
        """
        モデル評価

        Args:
            model_id: モデルID
            test_data: テストデータ
            feature_cols: 特徴量カラム
            label_col: ラベルカラム
            race_id_col: レースIDカラム

        Returns:
            評価指標
        """
        logging.info(f"モデル評価開始: {model_id}")

        # モデル読み込み
        if model_id not in self.models:
            self.load_model(model_id)

        model = self.models[model_id]

        # 予測
        X = test_data[feature_cols]
        y = test_data[label_col]
        race_groups = test_data[race_id_col]

        predictions = model.predict(X, race_groups)

        # 評価指標計算
        metrics = self._calculate_metrics(predictions, y, race_groups)

        logging.info(f"✓ 評価完了: {model_id}")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

        return metrics

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: pd.Series,
        race_groups: pd.Series
    ) -> Dict[str, float]:
        """評価指標計算"""
        metrics = {}

        # レース単位で評価
        ndcg_scores = []
        top1_correct = 0
        top3_correct = 0
        total_races = 0

        for race_id in race_groups.unique():
            mask = race_groups == race_id
            race_preds = predictions[mask]
            race_labels = labels[mask].values

            if len(race_preds) < 2:
                continue

            # NDCG
            relevance = 20 - race_labels
            try:
                ndcg1 = ndcg_score([relevance], [race_preds], k=1)
                ndcg3 = ndcg_score([relevance], [race_preds], k=3)
                ndcg_scores.append((ndcg1, ndcg3))
            except:
                pass

            # Top1 Accuracy
            pred_winner = np.argmax(race_preds)
            actual_winner = np.argmin(race_labels)
            if pred_winner == actual_winner:
                top1_correct += 1

            # Top3 Hit Rate
            if len(race_preds) >= 3:
                pred_top3 = set(np.argsort(race_preds)[-3:])
                actual_top3 = set(np.argsort(race_labels)[:3])
                if len(pred_top3 & actual_top3) > 0:
                    top3_correct += 1

            total_races += 1

        # メトリクス集計
        if ndcg_scores:
            metrics["ndcg@1"] = np.mean([s[0] for s in ndcg_scores])
            metrics["ndcg@3"] = np.mean([s[1] for s in ndcg_scores])

        if total_races > 0:
            metrics["top1_accuracy"] = top1_correct / total_races
            metrics["top3_hit_rate"] = top3_correct / total_races
            metrics["total_races"] = total_races

        return metrics

    def compare_models(
        self,
        model_ids: List[str],
        test_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "finish_position",
        race_id_col: str = "race_id",
        experiment_name: str = "model_comparison"
    ) -> pd.DataFrame:
        """
        複数モデルを比較

        Args:
            model_ids: 比較するモデルIDリスト
            test_data: テストデータ
            feature_cols: 特徴量カラム
            label_col: ラベルカラム
            race_id_col: レースIDカラム
            experiment_name: 実験名

        Returns:
            比較結果（DataFrame）
        """
        logging.info("="*60)
        logging.info(f"モデル比較実験: {experiment_name}")
        logging.info("="*60)
        logging.info(f"比較対象: {len(model_ids)}モデル")
        logging.info(f"テストデータ: {len(test_data)}レコード, {test_data[race_id_col].nunique()}レース")

        results = []

        for model_id in model_ids:
            try:
                metrics = self.evaluate_model(
                    model_id,
                    test_data,
                    feature_cols,
                    label_col,
                    race_id_col
                )

                model_info = self.registry.get_model(model_id)

                result = {
                    "model_id": model_id,
                    "model_name": model_info["model_name"],
                    "model_type": model_info["model_type"],
                    "version": model_info["version"],
                    **metrics
                }

                results.append(result)

            except Exception as e:
                logging.error(f"✗ エラー ({model_id}): {e}")

        # DataFrame作成
        df_results = pd.DataFrame(results)

        # 実験を登録
        self.registry.register_experiment(
            experiment_name=experiment_name,
            models=model_ids,
            test_data=str(test_data),
            results=df_results.to_dict('records'),
            description=f"{len(model_ids)}モデルの比較評価"
        )

        # 結果表示
        logging.info("\n" + "="*60)
        logging.info("比較結果")
        logging.info("="*60)
        print(df_results.to_string(index=False))

        return df_results

    def generate_comparison_report(
        self,
        df_results: pd.DataFrame,
        output_path: str
    ):
        """
        比較レポート生成

        Args:
            df_results: 比較結果
            output_path: 出力先
        """
        report_lines = [
            "# モデル比較レポート",
            "",
            f"**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**比較モデル数:** {len(df_results)}",
            "",
            "---",
            "",
            "## 性能比較",
            ""
        ]

        # テーブル
        report_lines.append("| モデル名 | タイプ | NDCG@1 | NDCG@3 | Top1精度 | Top3適中率 |")
        report_lines.append("|---------|-------|--------|--------|----------|-----------|")

        for _, row in df_results.iterrows():
            report_lines.append(
                f"| {row['model_name']} | {row['model_type']} | "
                f"{row.get('ndcg@1', 0):.4f} | {row.get('ndcg@3', 0):.4f} | "
                f"{row.get('top1_accuracy', 0):.2%} | {row.get('top3_hit_rate', 0):.2%} |"
            )

        report_lines.append("")

        # ベストモデル
        if 'ndcg@1' in df_results.columns:
            best_idx = df_results['ndcg@1'].idxmax()
            best = df_results.loc[best_idx]

            report_lines.append("## ベストモデル")
            report_lines.append("")
            report_lines.append(f"**モデル:** {best['model_name']} ({best['model_type']})")
            report_lines.append(f"**NDCG@1:** {best['ndcg@1']:.4f}")
            report_lines.append(f"**Top1精度:** {best['top1_accuracy']:.2%}")
            report_lines.append("")

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        logging.info(f"✓ 比較レポート保存: {output_path}")


def main():
    """メイン実行 - 既存モデルの登録とレポート生成"""
    # レジストリ初期化
    registry = ModelRegistry()

    # 既存のLightGBMモデルを登録
    registry.register_model(
        model_name="LightGBM_7Fold_Optimized",
        model_type="LightGBM",
        model_path="models/optimized/optimized_model_fold1.txt",
        version="v1.0",
        metrics={
            "ndcg@1": 0.926,
            "ndcg@3": 0.6351,
            "top1_accuracy": 0.10,
            "top3_hit_rate": 0.0455
        },
        hyperparameters={
            "objective": "lambdarank",
            "num_leaves": 73,
            "learning_rate": 0.00189
        },
        training_data="data/processed/keibalab_g1_2000_2024_processed.csv",
        description="Optuna最適化済み7-FoldモデL（Fold 1）",
        tags=["baseline", "optimized"]
    )

    # レポート生成
    report = registry.generate_report("models/registry/MODEL_REGISTRY_REPORT.md")
    print("\n" + report)


if __name__ == "__main__":
    main()

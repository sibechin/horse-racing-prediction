"""
モデルレジストリ - 全モデルのバージョン管理と比較評価

各モデルを保存し、性能を追跡し、比較検討を可能にする
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelRegistry:
    """モデルレジストリ - バージョン管理と比較"""

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """レジストリ読み込み"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"models": [], "experiments": []}

    def _save_registry(self):
        """レジストリ保存"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register_model(
        self,
        model_name: str,
        model_type: str,
        model_path: str,
        version: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        training_data: str,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """
        モデルを登録

        Args:
            model_name: モデル名（例: "LightGBM_7Fold"）
            model_type: モデルタイプ（例: "LightGBM", "Transformer", "Attention", "Hybrid"）
            model_path: モデルファイルのパス
            version: バージョン（例: "v1.0", "2025-11-21"）
            metrics: 性能指標 {"ndcg@1": 0.926, "top1_accuracy": 0.10, ...}
            hyperparameters: ハイパーパラメータ
            training_data: 訓練データのパス
            description: 説明
            tags: タグ（例: ["production", "g1_specialized"]）

        Returns:
            model_id: 登録されたモデルID
        """
        model_id = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "model_path": str(model_path),
            "version": version,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "training_data": str(training_data),
            "description": description,
            "tags": tags or [],
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }

        self.registry["models"].append(model_entry)
        self._save_registry()

        logging.info(f"✓ モデル登録完了: {model_id}")
        logging.info(f"  タイプ: {model_type}")
        logging.info(f"  主要指標: {metrics}")

        return model_id

    def register_experiment(
        self,
        experiment_name: str,
        models: List[str],
        test_data: str,
        results: Dict[str, Any],
        description: str = ""
    ) -> str:
        """
        実験を登録（複数モデルの比較テスト）

        Args:
            experiment_name: 実験名
            models: 比較対象のモデルIDリスト
            test_data: テストデータのパス
            results: 実験結果
            description: 説明

        Returns:
            experiment_id: 実験ID
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment_entry = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "models": models,
            "test_data": str(test_data),
            "results": results,
            "description": description,
            "executed_at": datetime.now().isoformat()
        }

        self.registry["experiments"].append(experiment_entry)
        self._save_registry()

        logging.info(f"✓ 実験登録完了: {experiment_id}")

        return experiment_id

    def get_model(self, model_id: str) -> Optional[Dict]:
        """モデル情報取得"""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                return model
        return None

    def list_models(
        self,
        model_type: str = None,
        tags: List[str] = None,
        status: str = "active"
    ) -> List[Dict]:
        """
        モデル一覧取得

        Args:
            model_type: フィルタするモデルタイプ
            tags: フィルタするタグ
            status: ステータス（"active", "archived", "deprecated"）

        Returns:
            モデルリスト
        """
        models = self.registry["models"]

        # フィルタ
        if status:
            models = [m for m in models if m.get("status") == status]

        if model_type:
            models = [m for m in models if m["model_type"] == model_type]

        if tags:
            models = [m for m in models if any(tag in m.get("tags", []) for tag in tags)]

        return models

    def compare_models(
        self,
        model_ids: List[str],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        モデル比較表を生成

        Args:
            model_ids: 比較するモデルIDリスト
            metrics: 比較する指標（None=全指標）

        Returns:
            比較表（DataFrame）
        """
        comparison_data = []

        for model_id in model_ids:
            model = self.get_model(model_id)
            if not model:
                logging.warning(f"モデルが見つかりません: {model_id}")
                continue

            row = {
                "model_id": model_id,
                "model_name": model["model_name"],
                "model_type": model["model_type"],
                "version": model["version"],
                "registered_at": model["registered_at"]
            }

            # メトリクス追加
            model_metrics = model.get("metrics", {})
            if metrics:
                for metric in metrics:
                    row[metric] = model_metrics.get(metric, None)
            else:
                row.update(model_metrics)

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def get_best_model(
        self,
        metric: str = "ndcg@1",
        model_type: str = None,
        tags: List[str] = None
    ) -> Optional[Dict]:
        """
        最良モデルを取得

        Args:
            metric: 評価指標
            model_type: フィルタするモデルタイプ
            tags: フィルタするタグ

        Returns:
            最良モデル
        """
        models = self.list_models(model_type=model_type, tags=tags)

        if not models:
            return None

        # メトリクスでソート
        models_with_metric = [m for m in models if metric in m.get("metrics", {})]
        if not models_with_metric:
            return None

        best_model = max(models_with_metric, key=lambda m: m["metrics"][metric])
        return best_model

    def archive_model(self, model_id: str):
        """モデルをアーカイブ（非アクティブ化）"""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                model["status"] = "archived"
                model["archived_at"] = datetime.now().isoformat()
                self._save_registry()
                logging.info(f"✓ モデルをアーカイブ: {model_id}")
                return

        logging.warning(f"モデルが見つかりません: {model_id}")

    def generate_report(self, output_path: str = None) -> str:
        """
        レジストリ全体のレポート生成

        Args:
            output_path: 出力先（None=標準出力）

        Returns:
            レポート（マークダウン形式）
        """
        report_lines = [
            "# モデルレジストリレポート",
            "",
            f"**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 登録モデル一覧",
            ""
        ]

        # モデルタイプごとに整理
        model_types = {}
        for model in self.registry["models"]:
            mtype = model["model_type"]
            if mtype not in model_types:
                model_types[mtype] = []
            model_types[mtype].append(model)

        for mtype, models in sorted(model_types.items()):
            report_lines.append(f"### {mtype} ({len(models)}モデル)")
            report_lines.append("")

            # テーブル
            report_lines.append("| モデル名 | バージョン | NDCG@1 | Top1精度 | 登録日時 | ステータス |")
            report_lines.append("|---------|-----------|--------|----------|----------|-----------|")

            for model in sorted(models, key=lambda m: m["registered_at"], reverse=True):
                metrics = model.get("metrics", {})
                ndcg = metrics.get("ndcg@1", metrics.get("NDCG@1", "-"))
                top1 = metrics.get("top1_accuracy", metrics.get("Top1精度", "-"))

                if isinstance(ndcg, float):
                    ndcg = f"{ndcg:.4f}"
                if isinstance(top1, float):
                    top1 = f"{top1:.2%}"

                report_lines.append(
                    f"| {model['model_name']} | {model['version']} | {ndcg} | {top1} | "
                    f"{model['registered_at'][:10]} | {model.get('status', 'active')} |"
                )

            report_lines.append("")

        # 実験一覧
        if self.registry["experiments"]:
            report_lines.append("## 実験履歴")
            report_lines.append("")

            for exp in sorted(self.registry["experiments"],
                            key=lambda e: e["executed_at"], reverse=True)[:10]:
                report_lines.append(f"### {exp['experiment_name']}")
                report_lines.append(f"**実行日時:** {exp['executed_at']}")
                report_lines.append(f"**対象モデル:** {len(exp['models'])}モデル")
                if exp.get("description"):
                    report_lines.append(f"**説明:** {exp['description']}")
                report_lines.append("")

        # ベストモデル
        report_lines.append("## 現在のベストモデル")
        report_lines.append("")

        for mtype in model_types.keys():
            best = self.get_best_model(model_type=mtype)
            if best:
                metrics = best.get("metrics", {})
                report_lines.append(f"**{mtype}:** {best['model_name']} (v{best['version']})")
                report_lines.append(f"- NDCG@1: {metrics.get('ndcg@1', '-')}")
                report_lines.append("")

        report = "\n".join(report_lines)

        # 保存
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logging.info(f"✓ レポート保存: {output_path}")

        return report


# ユーティリティ関数
def register_existing_models():
    """既存モデルをレジストリに登録"""
    registry = ModelRegistry()

    # LightGBM 7-Fold最適化モデル
    registry.register_model(
        model_name="LightGBM_7Fold_Optimized",
        model_type="LightGBM",
        model_path="models/optimized/optimized_model_fold1-7.txt",
        version="v1.0",
        metrics={
            "ndcg@1": 0.926,
            "ndcg@3": 0.6351,
            "top1_accuracy": 0.10,
            "top3_hit_rate": 0.0455,
            "test_ndcg@1_2025_g1": 0.0  # 2025年G1テスト
        },
        hyperparameters={
            "objective": "lambdarank",
            "num_leaves": 73,
            "learning_rate": 0.00189,
            "max_depth": 5,
            "feature_fraction": 0.914,
            "bagging_fraction": 0.899,
            "lambda_l1": 0.704,
            "lambda_l2": 5.989
        },
        training_data="data/processed/keibalab_g1_2000_2024_processed.csv",
        description="Optuna最適化済み7-FoldモデL。訓練データで92.6%達成も2025年G1で0%",
        tags=["optimized", "production", "baseline"]
    )

    logging.info("✓ 既存モデルの登録完了")


if __name__ == "__main__":
    # レジストリ初期化と既存モデル登録
    register_existing_models()

    # レポート生成
    registry = ModelRegistry()
    report = registry.generate_report("models/registry/MODEL_REGISTRY_REPORT.md")
    print(report)

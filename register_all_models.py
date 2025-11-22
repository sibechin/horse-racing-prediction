"""
全モデルをレジストリに登録
"""
from model_registry import ModelRegistry
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("="*60)
    logging.info("全モデルのレジストリ登録")
    logging.info("="*60)

    registry = ModelRegistry()

    # 1. LightGBM (既存)
    logging.info("\n1. LightGBM 7-Fold Optimized")
    registry.register_model(
        model_name="LightGBM_7Fold_Optimized",
        model_type="LightGBM",
        model_path="models/optimized/optimized_model_fold1.txt",
        version="v1.0",
        metrics={
            "ndcg@1": 0.926,
            "ndcg@3": 0.6351,
            "top1_accuracy": 0.10,
            "top3_hit_rate": 0.0455,
            "test_2025_g1_ndcg@1": 0.0
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
        description="Optuna最適化済み7-Foldモデル。訓練データで92.6%達成も2025年G1で0%",
        tags=["baseline", "optimized", "lightgbm"]
    )

    # 2. Conditional Logit
    logging.info("\n2. Conditional Logit Model")
    logit_models = list(Path("models/statistical").glob("conditional_logit_*.pkl"))
    if logit_models:
        latest_logit = sorted(logit_models)[-1]
        registry.register_model(
            model_name="Conditional_Logit",
            model_type="Statistical",
            model_path=str(latest_logit),
            version="v1.0",
            metrics={
                "training_accuracy": 0.4633,
                "features_count": 2
            },
            hyperparameters={
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced"
            },
            training_data="data/processed/keibalab_g1_2000_2024_processed.csv",
            description="ロジスティック回帰ベースのConditional Logitモデル",
            tags=["statistical", "logit"]
        )

    # 3. Transformer
    logging.info("\n3. Transformer Model")
    transformer_models = list(Path("models/transformer").glob("transformer_quick_*.pth"))
    if transformer_models:
        latest_tf = sorted(transformer_models)[-1]
        registry.register_model(
            model_name="Transformer_Quick",
            model_type="Transformer",
            model_path=str(latest_tf),
            version="v1.0",
            metrics={
                "ndcg@3": 0.7159,
                "top1_accuracy": 0.1133,
                "top3_hit_rate": 0.6084
            },
            hyperparameters={
                "d_model": 32,
                "nhead": 2,
                "num_layers": 1,
                "dropout": 0.1,
                "epochs": 10
            },
            training_data="data/processed/keibalab_g1_2000_2024_processed.csv",
            description="軽量Transformerモデル（10エポック訓練）",
            tags=["deep_learning", "transformer"]
        )

    # 4. Attention
    logging.info("\n4. Attention Model")
    attention_models = list(Path("models/attention").glob("attention_quick_*.pth"))
    if attention_models:
        latest_att = sorted(attention_models)[-1]
        registry.register_model(
            model_name="Attention_Quick",
            model_type="Attention",
            model_path=str(latest_att),
            version="v1.0",
            metrics={
                "ndcg@3": 0.7177,
                "top1_accuracy": 0.1192,
                "top3_hit_rate": 0.6051
            },
            hyperparameters={
                "hidden_dim": 32,
                "num_attention_heads": 2,
                "dropout": 0.1,
                "epochs": 10
            },
            training_data="data/processed/keibalab_g1_2000_2024_processed.csv",
            description="軽量Attentionモデル（10エポック訓練）",
            tags=["deep_learning", "attention"]
        )

    # レポート生成
    logging.info("\n" + "="*60)
    logging.info("レジストリレポート生成")
    logging.info("="*60)

    report = registry.generate_report("models/registry/MODEL_REGISTRY_REPORT.md")
    print("\n" + report)

    # モデル比較表
    logging.info("\n" + "="*60)
    logging.info("モデル比較表")
    logging.info("="*60)

    all_models = registry.list_models()
    model_ids = [m["model_id"] for m in all_models]

    if len(model_ids) > 1:
        comparison_df = registry.compare_models(model_ids, metrics=["ndcg@3", "top1_accuracy", "top3_hit_rate"])
        print("\n" + comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()

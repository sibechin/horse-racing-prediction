"""
ディープラーニングモデルの高速訓練
Transformer & Attention（エポック数削減版）
"""
import sys
sys.path.append('.')

from train_transformer_attention_models import (
    HorseRaceDataset, collate_fn,
    TransformerRankingModel, AttentionRankingModel,
    ListwiseLoss, train_epoch, evaluate_model
)

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def quick_train(model_type, epochs=10):
    """
    高速訓練（デモ用）

    Args:
        model_type: 'transformer' or 'attention'
        epochs: エポック数（デフォルト: 10）
    """
    logging.info("="*60)
    logging.info(f"{model_type.upper()}モデル 高速訓練")
    logging.info("="*60)

    # デバイス
    device = torch.device('cpu')
    logging.info(f"デバイス: {device}")

    # データ読み込み
    train_file = "data/processed/keibalab_g1_2000_2024_processed.csv"
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    logging.info(f"データ: {len(df)} records, {df['race_id'].nunique()} races")

    # 特徴量（利用可能なもののみ）
    possible_features = [
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight', 'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size', 'race_avg_weight',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded', 'distance', 'track_type_encoded'
    ]

    available_features = [col for col in possible_features if col in df.columns]
    logging.info(f"使用特徴量: {len(available_features)}個")

    X = df[available_features].copy()
    y = df['finish_position_numeric'].copy() if 'finish_position_numeric' in df.columns else df['finish_position'].copy()
    race_groups = df['race_id'].copy()

    # データセット
    dataset = HorseRaceDataset(X, y, race_groups)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # モデル作成
    input_dim = len(available_features)

    if model_type == 'transformer':
        model = TransformerRankingModel(
            input_dim=input_dim,
            d_model=32,  # 軽量化
            nhead=2,     # 軽量化
            num_layers=1,  # 軽量化
            dropout=0.1
        ).to(device)
    elif model_type == 'attention':
        model = AttentionRankingModel(
            input_dim=input_dim,
            hidden_dim=32,  # 軽量化
            num_attention_heads=2,  # 軽量化
            dropout=0.1
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logging.info(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # 損失・最適化
    criterion = ListwiseLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練ループ
    best_ndcg = 0.0
    best_model_path = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)

        if epoch % 2 == 0 or epoch == epochs:
            metrics = evaluate_model(model, dataloader, device)

            logging.info(f"Epoch {epoch}/{epochs}")
            logging.info(f"  Loss: {train_loss:.4f}")
            logging.info(f"  NDCG@3: {metrics['ndcg@3']:.4f}")
            logging.info(f"  Top1 Acc: {metrics['top1_accuracy']:.4f}")
            logging.info(f"  Top3 Hit: {metrics['top3_hit_rate']:.4f}")

            # ベストモデル保存
            if metrics['ndcg@3'] > best_ndcg:
                best_ndcg = metrics['ndcg@3']

                output_dir = Path(f"models/{model_type}")
                output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = output_dir / f'{model_type}_quick_{timestamp}.pth'

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': metrics,
                    'input_dim': input_dim,
                    'feature_columns': available_features
                }, model_path)

                best_model_path = model_path
                logging.info(f"  ✓ ベストモデル保存: {model_path}")

    # 最終評価
    final_metrics = evaluate_model(model, dataloader, device)

    logging.info("\n" + "="*60)
    logging.info(f"{model_type.upper()}モデル訓練完了！")
    logging.info("="*60)
    logging.info(f"最終性能:")
    logging.info(f"  NDCG@3: {final_metrics['ndcg@3']:.4f}")
    logging.info(f"  1着的中率: {final_metrics['top1_accuracy']:.2%}")
    logging.info(f"  3着内的中率: {final_metrics['top3_hit_rate']:.2%}")
    logging.info(f"保存先: {best_model_path}")

    return best_model_path, final_metrics


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("ディープラーニングモデル 高速訓練スクリプト")
    logging.info("="*60)

    results = {}

    # Transformerモデル
    try:
        logging.info("\n1. Transformerモデル")
        tf_path, tf_metrics = quick_train('transformer', epochs=10)
        results['transformer'] = {'path': tf_path, 'metrics': tf_metrics}
    except Exception as e:
        logging.error(f"Transformer訓練エラー: {e}")
        results['transformer'] = None

    # Attentionモデル
    try:
        logging.info("\n2. Attentionモデル")
        att_path, att_metrics = quick_train('attention', epochs=10)
        results['attention'] = {'path': att_path, 'metrics': att_metrics}
    except Exception as e:
        logging.error(f"Attention訓練エラー: {e}")
        results['attention'] = None

    # 結果サマリー
    logging.info("\n" + "="*60)
    logging.info("訓練完了サマリー")
    logging.info("="*60)

    for model_name, result in results.items():
        if result:
            metrics = result['metrics']
            logging.info(f"\n{model_name.upper()}:")
            logging.info(f"  NDCG@3: {metrics['ndcg@3']:.4f}")
            logging.info(f"  Top1: {metrics['top1_accuracy']:.2%}")
            logging.info(f"  Top3: {metrics['top3_hit_rate']:.2%}")
            logging.info(f"  保存先: {result['path']}")
        else:
            logging.info(f"\n{model_name.upper()}: 訓練失敗")


if __name__ == "__main__":
    main()

"""
Attention重みの可視化

Transformerモデルがどの特徴量に注目しているかを可視化
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from transformer_top1_model import TransformerTop1Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_attention_weights(trainer, df, race_id, save_path=None):
    """
    特定のレースのAttention重みを可視化

    Args:
        trainer: TransformerTop1Trainer インスタンス
        df: データフレーム
        race_id: 可視化するレースID
        save_path: 保存パス（Noneなら表示のみ）
    """
    # レースデータ抽出
    race_data = df[df['race_id'] == race_id]

    if len(race_data) == 0:
        logging.error(f"レースID {race_id} が見つかりません")
        return

    logging.info(f"レースID: {race_id}")
    logging.info(f"出走頭数: {len(race_data)}")

    # 予測とAttention取得
    X, _, _ = trainer.prepare_data(race_data)
    X = trainer.scaler.transform(X)

    X_tensor = torch.FloatTensor(X).unsqueeze(1).to(trainer.device)

    trainer.model.eval()
    with torch.no_grad():
        predictions = trainer.model(X_tensor)
        attention_weights = trainer.model.get_attention_weights()

    # Attention重み（最後のヘッドの平均）
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    attention = attention_weights.cpu().numpy()
    attention_mean = attention.mean(axis=1)  # ヘッド方向で平均

    # 各馬の特徴量に対する平均Attention
    # Shape: (batch_size, seq_len)
    attention_per_feature = attention_mean.squeeze(1)  # (batch_size, d_model)

    # 予測スコアでソート
    predictions_np = predictions.cpu().numpy()
    sorted_indices = np.argsort(predictions_np)[::-1]

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 予測確率
    ax1 = axes[0]
    horse_numbers = race_data['horse_number'].values if 'horse_number' in race_data.columns else np.arange(1, len(race_data)+1)

    ax1.barh(range(len(predictions_np)), predictions_np[sorted_indices])
    ax1.set_yticks(range(len(predictions_np)))
    ax1.set_yticklabels([f"Horse {horse_numbers[i]}" for i in sorted_indices])
    ax1.set_xlabel('Predicted Probability')
    ax1.set_title('Predicted Winner Probability')
    ax1.invert_yaxis()

    # 実際の着順がある場合、色分け
    if 'finish_position_numeric' in race_data.columns:
        actual_positions = race_data['finish_position_numeric'].values
        colors = ['green' if actual_positions[i] <= 3 else 'blue' for i in sorted_indices]
        for i, (bar, color) in enumerate(zip(ax1.patches, colors)):
            bar.set_color(color)

    # 右: Attention重み（上位5頭）
    ax2 = axes[1]
    top5_indices = sorted_indices[:5]

    feature_names = trainer.feature_cols
    attention_matrix = attention_per_feature[top5_indices]

    # Heatmap
    im = ax2.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.set_yticks(range(len(top5_indices)))
    ax2.set_yticklabels([f"Horse {horse_numbers[i]}" for i in top5_indices])
    ax2.set_title('Attention Weights (Top 5 Predicted Horses)')

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"✓ 保存: {save_path}")
    else:
        plt.show()

    plt.close()

    # 特徴量重要度（全馬の平均Attention）
    avg_attention_per_feature = attention_per_feature.mean(axis=0)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'attention': avg_attention_per_feature
    }).sort_values('attention', ascending=False)

    logging.info("\n特徴量重要度（Attention平均）:")
    print(feature_importance.to_string(index=False))

    return feature_importance


def compare_feature_importance(trainer, df, top_n=10):
    """
    全レースでの特徴量重要度を集計

    Args:
        trainer: TransformerTop1Trainer インスタンス
        df: データフレーム
        top_n: 表示する上位N個
    """
    logging.info("="*60)
    logging.info("全レースでの特徴量重要度分析")
    logging.info("="*60)

    all_attention = []

    for race_id in df['race_id'].unique()[:50]:  # 最初の50レースで分析
        race_data = df[df['race_id'] == race_id]

        X, _, _ = trainer.prepare_data(race_data)
        X = trainer.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(trainer.device)

        trainer.model.eval()
        with torch.no_grad():
            _ = trainer.model(X_tensor)
            attention_weights = trainer.model.get_attention_weights()

        attention = attention_weights.cpu().numpy()
        attention_mean = attention.mean(axis=(0, 1, 2))  # 全次元で平均
        all_attention.append(attention_mean)

    # 平均Attention
    avg_attention = np.mean(all_attention, axis=0)

    feature_importance = pd.DataFrame({
        'feature': trainer.feature_cols,
        'avg_attention': avg_attention
    }).sort_values('avg_attention', ascending=False)

    logging.info(f"\n特徴量重要度Top{top_n}:")
    print(feature_importance.head(top_n).to_string(index=False))

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))

    top_features = feature_importance.head(top_n)
    ax.barh(range(len(top_features)), top_features['avg_attention'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Average Attention Weight')
    ax.set_title(f'Feature Importance (Top {top_n}) - Averaged Across Races')
    ax.invert_yaxis()

    output_dir = Path("models/transformer_top1")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "feature_importance.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f"\n✓ 保存: {save_path}")
    plt.close()

    return feature_importance


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("Attention重みの可視化")
    logging.info("="*60)

    # モデル読み込み
    model_dir = Path("models/transformer_top1")
    model_files = list(model_dir.glob("transformer_top1_*.pth"))

    if not model_files:
        logging.error("訓練済みモデルが見つかりません")
        return

    # 最新モデル
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logging.info(f"\nモデル読み込み: {latest_model}")

    trainer = TransformerTop1Trainer.load(latest_model)

    # テストデータ
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')

    if 'finish_position_numeric' not in df_test.columns:
        df_test['finish_position_numeric'] = pd.to_numeric(df_test['finish_position'], errors='coerce')

    # 特徴量重要度（全レース）
    feature_importance = compare_feature_importance(trainer, df_test, top_n=10)

    # 個別レースの可視化（最新レース）
    latest_race_id = df_test['race_id'].iloc[-1]

    output_dir = Path("models/transformer_top1")
    save_path = output_dir / f"attention_race_{latest_race_id}.png"

    visualize_attention_weights(trainer, df_test, latest_race_id, save_path)

    logging.info("\n完了!")


if __name__ == "__main__":
    main()

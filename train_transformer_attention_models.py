"""
Transformer & Attention機構による競馬ランキング予測モデル
LightGBMと比較検討するための実装
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ndcg_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =====================================================
# データセット
# =====================================================

class HorseRaceDataset(Dataset):
    """競馬データセット"""
    def __init__(self, features, labels, race_groups):
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.FloatTensor(labels.values)
        self.race_groups = race_groups

        # レースごとにインデックスを整理
        self.race_indices = []
        for race_id in race_groups.unique():
            mask = race_groups == race_id
            indices = np.where(mask)[0]
            self.race_indices.append(indices)

    def __len__(self):
        return len(self.race_indices)

    def __getitem__(self, idx):
        indices = self.race_indices[idx]
        return (
            self.features[indices],
            self.labels[indices],
            len(indices)
        )


def collate_fn(batch):
    """可変長のレースデータをバッチ化"""
    features_list, labels_list, lengths = zip(*batch)

    # 最大頭数に合わせてパディング
    max_len = max(lengths)
    batch_size = len(batch)
    feature_dim = features_list[0].shape[1]

    features_padded = torch.zeros(batch_size, max_len, feature_dim)
    labels_padded = torch.zeros(batch_size, max_len)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (feat, lab, length) in enumerate(zip(features_list, labels_list, lengths)):
        features_padded[i, :length] = feat
        labels_padded[i, :length] = lab
        masks[i, :length] = True

    return features_padded, labels_padded, masks


# =====================================================
# Model 1: Transformer-based Ranking Model
# =====================================================

class TransformerRankingModel(nn.Module):
    """Transformerベースのランキングモデル"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerRankingModel, self).__init__()

        # 入力エンベディング
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 出力層
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, mask):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) - True for valid positions
        Returns:
            scores: (batch_size, seq_len)
        """
        # Embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # Transformer (padding位置をマスク)
        # PyTorchのTransformerは True=無視, False=有効 なので反転
        src_key_padding_mask = ~mask
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 各馬のスコア
        scores = self.output_layer(x).squeeze(-1)  # (batch_size, seq_len)

        # マスク位置は非常に低いスコアに
        scores = scores.masked_fill(~mask, -1e9)

        return scores


# =====================================================
# Model 2: Attention-based Ranking Model
# =====================================================

class AttentionRankingModel(nn.Module):
    """Attentionベースのランキングモデル"""
    def __init__(self, input_dim, hidden_dim=64, num_attention_heads=4, dropout=0.1):
        super(AttentionRankingModel, self).__init__()

        # 入力エンベディング
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-Head Self-Attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Layer Norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 出力層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) - True for valid positions
        Returns:
            scores: (batch_size, seq_len)
        """
        # Embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)

        # Self-Attention with residual connection
        attn_out, _ = self.multihead_attention(x, x, x, key_padding_mask=~mask)
        x = self.norm1(x + attn_out)

        # Feed Forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        # 各馬のスコア
        scores = self.output_layer(x).squeeze(-1)  # (batch_size, seq_len)

        # マスク位置は非常に低いスコアに
        scores = scores.masked_fill(~mask, -1e9)

        return scores


# =====================================================
# 損失関数
# =====================================================

class ListwiseLoss(nn.Module):
    """ListNet-style ranking loss"""
    def __init__(self):
        super(ListwiseLoss, self).__init__()

    def forward(self, scores, labels, mask):
        """
        Args:
            scores: (batch_size, seq_len) - 予測スコア
            labels: (batch_size, seq_len) - 実際の順位（1=1着, 2=2着, ...）
            mask: (batch_size, seq_len) - 有効な位置
        """
        # 順位を逆転して関連度スコアに変換（1着=高スコア）
        # max_rank = labels.max(dim=1, keepdim=True)[0]
        # relevance = (max_rank - labels + 1).float()

        # 順位を確率分布に変換
        relevance = (20 - labels).float()  # 1着=19, 2着=18, ...
        relevance = relevance.masked_fill(~mask, 0)

        # Softmax確率
        pred_probs = torch.softmax(scores, dim=1)
        target_probs = torch.softmax(relevance, dim=1)

        # Cross Entropy Loss
        loss = -(target_probs * torch.log(pred_probs + 1e-10)).sum(dim=1)

        return loss.mean()


# =====================================================
# 訓練・評価関数
# =====================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0.0

    for features, labels, masks in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward
        scores = model(features, masks)
        loss = criterion(scores, labels, masks)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """モデル評価"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for features, labels, masks in dataloader:
            features = features.to(device)

            scores = model(features, masks)

            all_predictions.append(scores.cpu())
            all_labels.append(labels)
            all_masks.append(masks)

    # レース単位で評価
    metrics = {}

    ndcg_scores = []
    top1_correct = 0
    top3_correct = 0
    total_races = 0

    for scores, labels, masks in zip(all_predictions, all_labels, all_masks):
        for race_scores, race_labels, race_mask in zip(scores, labels, masks):
            # 有効な馬のみ
            valid_indices = race_mask.bool()
            race_scores = race_scores[valid_indices].numpy()
            race_labels = race_labels[valid_indices].numpy()

            if len(race_scores) < 2:
                continue

            # NDCG
            relevance = 20 - race_labels
            try:
                ndcg = ndcg_score([relevance], [race_scores], k=3)
                ndcg_scores.append(ndcg)
            except:
                pass

            # Top1 Accuracy
            pred_winner = np.argmax(race_scores)
            actual_winner = np.argmin(race_labels)
            if pred_winner == actual_winner:
                top1_correct += 1

            # Top3 Hit Rate
            if len(race_scores) >= 3:
                pred_top3 = set(np.argsort(race_scores)[-3:])
                actual_top3 = set(np.argsort(race_labels)[:3])
                if len(pred_top3 & actual_top3) > 0:
                    top3_correct += 1

            total_races += 1

    metrics['ndcg@3'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    metrics['top1_accuracy'] = top1_correct / total_races if total_races > 0 else 0.0
    metrics['top3_hit_rate'] = top3_correct / total_races if total_races > 0 else 0.0

    return metrics


# =====================================================
# メイン訓練関数
# =====================================================

def train_model(model_type, train_file, output_dir, epochs=50, batch_size=32, lr=0.001):
    """
    Args:
        model_type: 'transformer' or 'attention'
        train_file: 訓練データファイル
        output_dir: モデル保存ディレクトリ
        epochs: エポック数
        batch_size: バッチサイズ
        lr: 学習率
    """
    logging.info("="*60)
    logging.info(f"{model_type.upper()}モデル訓練開始")
    logging.info("="*60)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用デバイス: {device}")

    # データ読み込み
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    logging.info(f"データ読み込み: {len(df)} records, {df['race_id'].nunique()} races")

    # 特徴量（オッズ除外）
    feature_columns = [
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight', 'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size', 'race_avg_weight',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]

    available_features = [col for col in feature_columns if col in df.columns]
    logging.info(f"使用する特徴量: {len(available_features)}個")

    X = df[available_features].copy()
    y = df['finish_position_numeric'].copy()
    race_groups = df['race_id'].copy()

    # データセット作成
    dataset = HorseRaceDataset(X, y, race_groups)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # モデル作成
    input_dim = len(available_features)

    if model_type == 'transformer':
        model = TransformerRankingModel(
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1
        ).to(device)
    elif model_type == 'attention':
        model = AttentionRankingModel(
            input_dim=input_dim,
            hidden_dim=64,
            num_attention_heads=4,
            dropout=0.1
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logging.info(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # 損失関数・最適化
    criterion = ListwiseLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # 訓練ループ
    best_ndcg = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)

        if epoch % 5 == 0:
            metrics = evaluate_model(model, dataloader, device)

            logging.info(f"Epoch {epoch}/{epochs}")
            logging.info(f"  Loss: {train_loss:.4f}")
            logging.info(f"  NDCG@3: {metrics['ndcg@3']:.4f}")
            logging.info(f"  Top1 Acc: {metrics['top1_accuracy']:.4f}")
            logging.info(f"  Top3 Hit: {metrics['top3_hit_rate']:.4f}")

            scheduler.step(metrics['ndcg@3'])

            # ベストモデル保存
            if metrics['ndcg@3'] > best_ndcg:
                best_ndcg = metrics['ndcg@3']

                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_file = output_path / f'{model_type}_model_best_{timestamp}.pth'

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': metrics,
                    'input_dim': input_dim,
                    'feature_columns': available_features
                }, model_file)

                logging.info(f"  新しいベストモデル保存: {model_file}")

    # 最終評価
    final_metrics = evaluate_model(model, dataloader, device)

    logging.info("\n" + "="*60)
    logging.info(f"{model_type.upper()}モデル訓練完了!")
    logging.info("="*60)
    logging.info(f"最終性能:")
    logging.info(f"  NDCG@3: {final_metrics['ndcg@3']:.4f}")
    logging.info(f"  1着的中率: {final_metrics['top1_accuracy']:.2%}")
    logging.info(f"  3着内的中率: {final_metrics['top3_hit_rate']:.2%}")

    return model, final_metrics


# =====================================================
# 比較実行
# =====================================================

def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("Transformer & Attention モデル訓練スクリプト")
    logging.info("="*60)

    train_file = 'data/processed/keibalab_g1_2000_2024_processed.csv'

    # Transformerモデル
    logging.info("\n1. Transformerモデル訓練")
    transformer_model, transformer_metrics = train_model(
        model_type='transformer',
        train_file=train_file,
        output_dir='models/transformer',
        epochs=50,
        batch_size=32,
        lr=0.001
    )

    # Attentionモデル
    logging.info("\n2. Attentionモデル訓練")
    attention_model, attention_metrics = train_model(
        model_type='attention',
        train_file=train_file,
        output_dir='models/attention',
        epochs=50,
        batch_size=32,
        lr=0.001
    )

    # 比較結果
    logging.info("\n" + "="*60)
    logging.info("モデル比較結果")
    logging.info("="*60)

    logging.info("\nTransformer:")
    for metric, value in transformer_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    logging.info("\nAttention:")
    for metric, value in attention_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    # LightGBMとの比較（既に訓練済みの結果を参照）
    logging.info("\nLightGBM (参考):")
    logging.info("  NDCG@3: 0.9246")
    logging.info("  1着的中率: 0.10%")
    logging.info("  3着内的中率: 4.55%")


if __name__ == "__main__":
    main()

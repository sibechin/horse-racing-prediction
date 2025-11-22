"""
Transformer + Attention による TOP1予測モデル

特徴:
- Multi-head Self-Attention で特徴量間の関係を学習
- 勝ち馬予測に特化（Binary Classification）
- Attention重みの可視化が可能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention層"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # 可視化用

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # 保存
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)

        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerTop1Predictor(nn.Module):
    """Transformer-based TOP1予測モデル"""

    def __init__(
        self,
        input_dim,
        d_model=128,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        dropout=0.1
    ):
        super().__init__()

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output layer (binary classification)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output (probability of being winner)
        output = self.output_layer(x)

        return output.squeeze(-1)

    def get_attention_weights(self):
        """Get attention weights from last transformer block"""
        return self.transformer_blocks[-1].attention.attention_weights


class HorseRacingDataset(Dataset):
    """競馬データセット"""

    def __init__(self, X, y, race_ids):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.race_ids = race_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.race_ids[idx]


class TransformerTop1Trainer:
    """Transformer TOP1モデルの訓練"""

    def __init__(
        self,
        input_dim,
        d_model=128,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        dropout=0.1,
        lr=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = TransformerTop1Predictor(
            input_dim, d_model, num_heads, num_layers, d_ff, dropout
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = StandardScaler()

        self.feature_cols = [
            'year', 'month', 'day_of_week', 'venue_code', 'race_number',
            'sex_encoded', 'age', 'weight', 'last_3f', 'horse_weight_kg',
            'horse_weight_change', 'field_size', 'race_avg_odds', 'race_avg_weight',
            'trainer_encoded', 'trainer_frequency', 'jockey_encoded', 'jockey_frequency',
            'race_type_encoded'
        ]

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_top1_acc': []
        }

    def prepare_data(self, df):
        """データ準備"""
        # 特徴量
        available_features = [col for col in self.feature_cols if col in df.columns]
        X = df[available_features].values

        # ターゲット: 1着=1, それ以外=0
        if 'finish_position_numeric' in df.columns:
            y = (df['finish_position_numeric'] == 1).astype(float).values
        elif 'finish_position' in df.columns:
            y = (pd.to_numeric(df['finish_position'], errors='coerce') == 1).astype(float).values
        else:
            raise ValueError("finish_position カラムが見つかりません")

        race_ids = df['race_id'].values

        return X, y, race_ids

    def train_epoch(self, train_loader, pos_weight):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0

        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            outputs = self.model(X_batch.unsqueeze(1))  # Add sequence dimension

            # Loss (weighted BCE for class imbalance)
            loss = F.binary_cross_entropy(
                outputs, y_batch,
                weight=torch.where(y_batch == 1, pos_weight, torch.tensor(1.0).to(self.device))
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader, df_val):
        """評価"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_race_ids = []

        with torch.no_grad():
            for X_batch, y_batch, race_ids in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch.unsqueeze(1))
                loss = F.binary_cross_entropy(outputs, y_batch)

                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_race_ids.extend(race_ids.numpy())

        # Top1精度（レースごと）
        df_predictions = pd.DataFrame({
            'race_id': all_race_ids,
            'prediction': all_predictions
        })

        top1_correct = 0
        total_races = df_val['race_id'].nunique()

        for race_id in df_val['race_id'].unique():
            race_data = df_val[df_val['race_id'] == race_id]
            race_preds = df_predictions[df_predictions['race_id'] == race_id]['prediction'].values

            if len(race_preds) == 0:
                continue

            # 予測された勝ち馬
            pred_winner_idx = np.argmax(race_preds)

            # 実際の勝ち馬
            if 'finish_position_numeric' in race_data.columns:
                actual_positions = race_data['finish_position_numeric'].values
            else:
                actual_positions = pd.to_numeric(race_data['finish_position'], errors='coerce').values

            actual_winner_idx = np.argmin(actual_positions)

            if pred_winner_idx == actual_winner_idx:
                top1_correct += 1

        top1_accuracy = top1_correct / total_races if total_races > 0 else 0

        return total_loss / len(val_loader), top1_accuracy

    def fit(self, df_train, epochs=50, batch_size=32, validation_split=0.2):
        """訓練"""
        logging.info("="*60)
        logging.info("Transformer TOP1モデル 訓練")
        logging.info("="*60)

        # データ準備
        X, y, race_ids = self.prepare_data(df_train)

        # 正規化
        X = self.scaler.fit_transform(X)

        # Positive weight (class imbalance対策)
        pos_weight = torch.tensor((y == 0).sum() / (y == 1).sum()).to(self.device)
        logging.info(f"Positive weight (for class imbalance): {pos_weight.item():.2f}")

        # Train/Val split (by race)
        unique_races = np.unique(race_ids)
        n_val = int(len(unique_races) * validation_split)
        np.random.shuffle(unique_races)
        val_races = set(unique_races[:n_val])

        train_mask = np.array([rid not in val_races for rid in race_ids])
        val_mask = ~train_mask

        # Datasets
        train_dataset = HorseRacingDataset(X[train_mask], y[train_mask], race_ids[train_mask])
        val_dataset = HorseRacingDataset(X[val_mask], y[val_mask], race_ids[val_mask])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        df_val = df_train[val_mask]

        logging.info(f"訓練サンプル: {len(train_dataset)}")
        logging.info(f"検証サンプル: {len(val_dataset)}")
        logging.info(f"勝ち馬割合: {y.mean():.3%}")

        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, pos_weight)
            val_loss, val_top1_acc = self.evaluate(val_loader, df_val)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_top1_acc'].append(val_top1_acc)

            logging.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Top1 Acc: {val_top1_acc:.2%}"
            )

            # Early stopping
            if val_top1_acc > best_val_acc:
                best_val_acc = val_top1_acc
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        logging.info(f"\n最良検証Top1精度: {best_val_acc:.2%}")

        return self

    def predict(self, df):
        """予測"""
        X, _, race_ids = self.prepare_data(df)
        X = self.scaler.transform(X)

        dataset = HorseRacingDataset(X, np.zeros(len(X)), race_ids)
        loader = DataLoader(dataset, batch_size=32)

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for X_batch, _, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch.unsqueeze(1))
                all_predictions.extend(outputs.cpu().numpy())

        return np.array(all_predictions)

    def save(self, path):
        """モデル保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'history': self.history
        }, path)
        logging.info(f"✓ モデル保存: {path}")

    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """モデル読み込み"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        input_dim = len(checkpoint['feature_cols'])
        trainer = cls(input_dim, device=device)

        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.scaler = checkpoint['scaler']
        trainer.feature_cols = checkpoint['feature_cols']
        trainer.history = checkpoint['history']

        return trainer


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("Transformer + Attention TOP1予測")
    logging.info("="*60)

    # データ読み込み
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")

    if not train_file.exists():
        logging.error(f"訓練データが見つかりません: {train_file}")
        return

    logging.info(f"\n訓練データ: {train_file}")
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')

    logging.info(f"テストデータ: {test_file}")
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')

    # finish_position_numericの確認
    for df in [df_train, df_test]:
        if 'finish_position_numeric' not in df.columns:
            df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

    # モデル訓練
    trainer = TransformerTop1Trainer(
        input_dim=19,
        d_model=128,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        dropout=0.2,
        lr=0.001
    )

    trainer.fit(df_train, epochs=100, batch_size=64, validation_split=0.2)

    # テストデータで評価
    logging.info("\n" + "="*60)
    logging.info("テストデータ評価")
    logging.info("="*60)

    predictions = trainer.predict(df_test)

    # Top1精度
    top1_correct = 0
    total_races = df_test['race_id'].nunique()

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]
        race_indices = race_data.index
        race_preds = predictions[race_indices]

        pred_winner_idx = np.argmax(race_preds)
        actual_winner_idx = np.argmin(race_data['finish_position_numeric'].values)

        if pred_winner_idx == actual_winner_idx:
            top1_correct += 1

    top1_accuracy = top1_correct / total_races

    logging.info(f"\nテストTop1精度: {top1_accuracy:.2%}")
    logging.info(f"評価レース数: {total_races}")

    # モデル保存
    output_dir = Path("models/transformer_top1")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_dir / f"transformer_top1_{timestamp}.pth"

    trainer.save(model_path)

    # 結果保存
    results = {
        "trained_at": datetime.now().isoformat(),
        "model_architecture": {
            "d_model": 128,
            "num_heads": 8,
            "num_layers": 3,
            "d_ff": 512,
            "dropout": 0.2
        },
        "training": {
            "epochs": len(trainer.history['train_loss']),
            "best_val_top1_acc": max(trainer.history['val_top1_acc'])
        },
        "test_results": {
            "top1_accuracy": top1_accuracy,
            "total_races": total_races
        }
    }

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n✓ 結果保存: {results_path}")
    logging.info("\n完了!")


if __name__ == "__main__":
    main()

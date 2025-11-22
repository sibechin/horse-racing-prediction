"""
Transformer + Attention TOP1予測モデル v2

改善点:
- 学習率を下げる (0.0001)
- シンプルなアーキテクチャ
- Focal Loss でクラス不均衡対策
- レース単位でのランキング損失追加
- データシャッフル
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FocalLoss(nn.Module):
    """Focal Loss - クラス不均衡に強い"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class SimpleAttention(nn.Module):
    """シンプルなAttention層"""
    def __init__(self, d_model, num_heads=4, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, self.attn_weights = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x


class TransformerTop1V2(nn.Module):
    """シンプルなTransformer TOP1予測モデル"""
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention layers
        self.attention_layers = nn.ModuleList([
            SimpleAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Output
        self.output = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)

        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        x = self.ff(x)

        # Output per horse
        output = self.output(x).squeeze(-1)  # (batch, seq_len)

        return output


class RaceDataset(Dataset):
    """レース単位のデータセット"""
    def __init__(self, races_data, max_horses=18):
        self.races = races_data
        self.max_horses = max_horses

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        race = self.races[idx]
        X = race['X']
        y = race['y']
        n_horses = len(X)

        # パディング
        if n_horses < self.max_horses:
            pad_size = self.max_horses - n_horses
            X = np.vstack([X, np.zeros((pad_size, X.shape[1]))])
            y = np.concatenate([y, np.zeros(pad_size)])
            mask = np.concatenate([np.ones(n_horses), np.zeros(pad_size)])
        else:
            X = X[:self.max_horses]
            y = y[:self.max_horses]
            mask = np.ones(self.max_horses)

        return {
            'X': torch.FloatTensor(X),
            'y': torch.FloatTensor(y),
            'mask': torch.FloatTensor(mask),
            'n_horses': n_horses
        }


class TransformerTop1TrainerV2:
    """改良版Trainer"""
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dropout=0.3,
                 lr=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.input_dim = input_dim

        self.model = TransformerTop1V2(
            input_dim, d_model, num_heads, num_layers, dropout
        ).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        self.scaler = StandardScaler()
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

        self.feature_cols = [
            'year', 'month', 'day_of_week', 'venue_code', 'race_number',
            'sex_encoded', 'age', 'weight', 'last_3f', 'horse_weight_kg',
            'horse_weight_change', 'field_size', 'race_avg_odds', 'race_avg_weight',
            'trainer_encoded', 'trainer_frequency', 'jockey_encoded', 'jockey_frequency',
            'race_type_encoded'
        ]

        self.history = {'train_loss': [], 'val_loss': [], 'val_top1_acc': []}

    def prepare_races(self, df, fit_scaler=False):
        """レース単位でデータ準備"""
        available_features = [col for col in self.feature_cols if col in df.columns]

        races = []
        for race_id in df['race_id'].unique():
            race_data = df[df['race_id'] == race_id].copy()

            # シャッフル（重要！）
            race_data = race_data.sample(frac=1, random_state=None).reset_index(drop=True)

            X = race_data[available_features].values
            y = (race_data['finish_position_numeric'] == 1).astype(float).values

            races.append({'X': X, 'y': y, 'race_id': race_id})

        # スケーリング
        if fit_scaler:
            all_X = np.vstack([r['X'] for r in races])
            self.scaler.fit(all_X)

        for race in races:
            race['X'] = self.scaler.transform(race['X'])

        return races

    def compute_loss(self, outputs, targets, mask):
        """損失計算（Focal Loss + ランキング損失）"""
        # Sigmoid
        probs = torch.sigmoid(outputs)

        # Focal Loss (マスク適用)
        focal = self.focal_loss(probs * mask, targets * mask)

        # Pairwise Ranking Loss
        rank_loss = 0
        batch_size = outputs.size(0)

        for b in range(batch_size):
            valid_mask = mask[b] > 0
            valid_outputs = outputs[b][valid_mask]
            valid_targets = targets[b][valid_mask]

            if valid_targets.sum() > 0 and (1 - valid_targets).sum() > 0:
                pos_idx = (valid_targets == 1).nonzero(as_tuple=True)[0]
                neg_idx = (valid_targets == 0).nonzero(as_tuple=True)[0]

                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    pos_scores = valid_outputs[pos_idx]
                    neg_scores = valid_outputs[neg_idx]

                    # マージンランキング損失
                    margin = 1.0
                    for ps in pos_scores:
                        for ns in neg_scores:
                            rank_loss += F.relu(margin - (ps - ns))

        rank_loss = rank_loss / batch_size

        return focal + 0.1 * rank_loss

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)
            mask = batch['mask'].to(self.device)

            outputs = self.model(X)
            loss = self.compute_loss(outputs, y, mask)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader, val_races):
        self.model.eval()
        total_loss = 0
        top1_correct = 0
        total_races = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                mask = batch['mask'].to(self.device)
                n_horses = batch['n_horses']

                outputs = self.model(X)
                loss = self.compute_loss(outputs, y, mask)
                total_loss += loss.item()

                # Top1精度
                probs = torch.sigmoid(outputs)
                for i in range(X.size(0)):
                    n = n_horses[i].item()
                    pred = probs[i, :n].cpu().numpy()
                    actual = y[i, :n].cpu().numpy()

                    pred_winner = np.argmax(pred)
                    actual_winner = np.argmax(actual)

                    if pred_winner == actual_winner:
                        top1_correct += 1
                    total_races += 1

        return total_loss / len(val_loader), top1_correct / total_races if total_races > 0 else 0

    def fit(self, df_train, epochs=100, batch_size=16, validation_split=0.2):
        logging.info("="*60)
        logging.info("Transformer TOP1 v2 訓練")
        logging.info("="*60)

        # レースデータ準備
        races = self.prepare_races(df_train, fit_scaler=True)
        logging.info(f"総レース数: {len(races)}")

        # Train/Val分割
        np.random.shuffle(races)
        n_val = int(len(races) * validation_split)
        val_races = races[:n_val]
        train_races = races[n_val:]

        train_dataset = RaceDataset(train_races)
        val_dataset = RaceDataset(val_races)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        logging.info(f"訓練レース: {len(train_races)}")
        logging.info(f"検証レース: {len(val_races)}")

        best_val_acc = 0
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_top1_acc = self.evaluate(val_loader, val_races)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_top1_acc'].append(val_top1_acc)

            self.scheduler.step(val_top1_acc)

            logging.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Top1: {val_top1_acc:.2%}"
            )

            if val_top1_acc > best_val_acc:
                best_val_acc = val_top1_acc
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(self.best_model_state)
        logging.info(f"\n最良検証Top1精度: {best_val_acc:.2%}")

        return self

    def predict_race(self, race_data):
        """1レースの予測"""
        available_features = [col for col in self.feature_cols if col in race_data.columns]
        X = race_data[available_features].values
        X = self.scaler.transform(X)

        n_horses = len(X)
        max_horses = 18

        if n_horses < max_horses:
            pad_size = max_horses - n_horses
            X = np.vstack([X, np.zeros((pad_size, X.shape[1]))])

        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs[0, :n_horses])

        return probs.cpu().numpy()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'history': self.history,
            'input_dim': self.input_dim
        }, path)
        logging.info(f"✓ モデル保存: {path}")

    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        input_dim = checkpoint['input_dim']
        trainer = cls(input_dim, device=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.scaler = checkpoint['scaler']
        trainer.feature_cols = checkpoint['feature_cols']
        trainer.history = checkpoint['history']
        return trainer


def main():
    logging.info("="*60)
    logging.info("Transformer + Attention TOP1予測 v2")
    logging.info("="*60)

    # データ
    train_file = Path("data/processed/keibalab_g1_2000_2024_cleaned.csv")
    test_file = Path("data/processed/keibalab_g1_2025_cleaned.csv")

    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')

    for df in [df_train, df_test]:
        if 'finish_position_numeric' not in df.columns:
            df['finish_position_numeric'] = pd.to_numeric(df['finish_position'], errors='coerce')

    # 訓練
    trainer = TransformerTop1TrainerV2(
        input_dim=19,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        lr=0.0001
    )

    trainer.fit(df_train, epochs=100, batch_size=16, validation_split=0.2)

    # テスト評価
    logging.info("\n" + "="*60)
    logging.info("テストデータ評価")
    logging.info("="*60)

    top1_correct = 0
    top3_hit = 0
    total_races = 0

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]
        predictions = trainer.predict_race(race_data)

        actual_positions = race_data['finish_position_numeric'].values

        pred_winner = np.argmax(predictions)
        actual_winner = np.argmin(actual_positions)

        if pred_winner == actual_winner:
            top1_correct += 1

        if len(predictions) >= 3:
            pred_top3 = set(np.argsort(predictions)[-3:])
            actual_top3 = set(np.argsort(actual_positions)[:3])
            if len(pred_top3 & actual_top3) > 0:
                top3_hit += 1

        total_races += 1

    logging.info(f"\nテスト結果:")
    logging.info(f"  Top1精度: {top1_correct/total_races:.2%} ({top1_correct}/{total_races})")
    logging.info(f"  Top3適中率: {top3_hit/total_races:.2%} ({top3_hit}/{total_races})")

    # 保存
    output_dir = Path("models/transformer_top1_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_dir / f"transformer_top1_v2_{timestamp}.pth"
    trainer.save(model_path)

    results = {
        "test_top1_accuracy": top1_correct / total_races,
        "test_top3_hit_rate": top3_hit / total_races,
        "total_races": total_races
    }

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"\n✓ 結果保存: {results_path}")


if __name__ == "__main__":
    main()

"""
Transformer + Odds TOP1予測モデル

オッズ情報を活用した予測
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
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TransformerWithOdds(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.output(x).squeeze(-1)


class RaceDataset(Dataset):
    def __init__(self, races_data, max_horses=18):
        self.races = races_data
        self.max_horses = max_horses

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        race = self.races[idx]
        X, y = race['X'], race['y']
        n_horses = len(X)

        if n_horses < self.max_horses:
            pad_size = self.max_horses - n_horses
            X = np.vstack([X, np.zeros((pad_size, X.shape[1]))])
            y = np.concatenate([y, np.zeros(pad_size)])
            mask = np.concatenate([np.zeros(n_horses), np.ones(pad_size)])  # 1 = masked
        else:
            X, y = X[:self.max_horses], y[:self.max_horses]
            mask = np.zeros(self.max_horses)

        return {
            'X': torch.FloatTensor(X),
            'y': torch.FloatTensor(y),
            'mask': torch.BoolTensor(mask.astype(bool)),
            'n_horses': n_horses
        }


class TransformerOddsTrainer:
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dropout=0.3,
                 lr=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.input_dim = input_dim

        self.model = TransformerWithOdds(input_dim, d_model, num_heads, num_layers, dropout).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.scaler = StandardScaler()
        self.focal_loss = FocalLoss()

        # オッズを含む特徴量
        self.feature_cols = [
            'odds', 'popularity_rank', 'weight', 'age',
            'jockey_win_rate', 'jockey_top3_rate', 'jockey_track_win_rate',
            'horse_win_rate', 'horse_top3_rate', 'horse_race_count', 'horse_dist_win_rate',
            'field_size', 'race_avg_odds', 'distance',
            'track_type_encoded', 'track_condition_encoded', 'weather_encoded',
            'sex_encoded', 'distance_category_encoded',
            'year', 'month', 'day_of_week'
        ]

        self.history = {'train_loss': [], 'val_loss': [], 'val_top1_acc': []}

    def prepare_races(self, df, fit_scaler=False):
        available = [c for c in self.feature_cols if c in df.columns]
        logging.info(f"使用特徴量: {len(available)}/{len(self.feature_cols)}")

        races = []
        for race_id in df['race_id'].unique():
            race_data = df[df['race_id'] == race_id].copy()
            race_data = race_data.sample(frac=1).reset_index(drop=True)  # シャッフル

            X = race_data[available].fillna(0).values

            if 'finish_position' in race_data.columns:
                positions = pd.to_numeric(race_data['finish_position'], errors='coerce').values
            else:
                positions = race_data['finish_position_numeric'].values

            y = (positions == 1).astype(float)
            races.append({'X': X, 'y': y, 'race_id': race_id})

        if fit_scaler:
            all_X = np.vstack([r['X'] for r in races])
            self.scaler.fit(all_X)

        for race in races:
            race['X'] = self.scaler.transform(race['X'])

        self.input_dim = len(available)
        return races

    def compute_loss(self, outputs, targets, mask):
        probs = torch.sigmoid(outputs)
        valid_mask = ~mask
        focal = self.focal_loss(probs * valid_mask, targets * valid_mask)

        # Ranking loss
        rank_loss = 0
        for b in range(outputs.size(0)):
            valid = ~mask[b]
            valid_out = outputs[b][valid]
            valid_tgt = targets[b][valid]

            if valid_tgt.sum() > 0 and (1 - valid_tgt).sum() > 0:
                pos_idx = (valid_tgt == 1).nonzero(as_tuple=True)[0]
                neg_idx = (valid_tgt == 0).nonzero(as_tuple=True)[0]

                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    for ps in valid_out[pos_idx]:
                        for ns in valid_out[neg_idx]:
                            rank_loss += F.relu(1.0 - (ps - ns))

        return focal + 0.1 * rank_loss / outputs.size(0)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            X = batch['X'].to(self.device)
            y = batch['y'].to(self.device)
            mask = batch['mask'].to(self.device)

            outputs = self.model(X, mask)
            loss = self.compute_loss(outputs, y, mask)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        top1_correct = total_races = 0

        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(self.device)
                y = batch['y'].to(self.device)
                mask = batch['mask'].to(self.device)
                n_horses = batch['n_horses']

                outputs = self.model(X, mask)
                loss = self.compute_loss(outputs, y, mask)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                for i in range(X.size(0)):
                    n = n_horses[i].item()
                    pred = probs[i, :n].cpu().numpy()
                    actual = y[i, :n].cpu().numpy()

                    if np.argmax(pred) == np.argmax(actual):
                        top1_correct += 1
                    total_races += 1

        return total_loss / len(val_loader), top1_correct / total_races if total_races > 0 else 0

    def fit(self, df_train, epochs=100, batch_size=8, validation_split=0.2):
        logging.info("="*60)
        logging.info("Transformer + Odds 訓練")
        logging.info("="*60)

        races = self.prepare_races(df_train, fit_scaler=True)

        # Update model input dim
        self.model = TransformerWithOdds(self.input_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)

        np.random.shuffle(races)
        n_val = int(len(races) * validation_split)
        val_races, train_races = races[:n_val], races[n_val:]

        train_loader = DataLoader(RaceDataset(train_races), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(RaceDataset(val_races), batch_size=batch_size)

        logging.info(f"訓練: {len(train_races)} レース, 検証: {len(val_races)} レース")

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_top1_acc'].append(val_acc)

            self.scheduler.step(val_acc)

            logging.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Top1: {val_acc:.2%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(self.best_state)
        logging.info(f"\n最良検証Top1: {best_val_acc:.2%}")
        return self

    def predict_race(self, race_data):
        available = [c for c in self.feature_cols if c in race_data.columns]
        X = race_data[available].fillna(0).values
        X = self.scaler.transform(X)
        n = len(X)

        if n < 18:
            X = np.vstack([X, np.zeros((18 - n, X.shape[1]))])

        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs[0, :n])

        return probs.cpu().numpy()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'input_dim': self.input_dim,
            'history': self.history
        }, path)
        logging.info(f"保存: {path}")

    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        trainer = cls(checkpoint['input_dim'], device=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.scaler = checkpoint['scaler']
        trainer.feature_cols = checkpoint['feature_cols']
        trainer.history = checkpoint['history']
        return trainer


def main():
    logging.info("="*60)
    logging.info("Transformer + Odds モデル訓練")
    logging.info("="*60)

    # データ読み込み
    df_train = pd.read_csv("data/processed/g1_races_historical_processed.csv", encoding='utf-8-sig')
    df_test = pd.read_csv("data/processed/g1_races_2025_processed.csv", encoding='utf-8-sig')

    logging.info(f"訓練データ: {len(df_train)} records, {df_train['race_id'].nunique()} races")
    logging.info(f"テストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races")

    # 訓練
    trainer = TransformerOddsTrainer(input_dim=22)
    trainer.fit(df_train, epochs=100, batch_size=8)

    # テスト評価
    logging.info("\n" + "="*60)
    logging.info("テスト評価")
    logging.info("="*60)

    results = {'top1': 0, 'top2_any': 0, 'top3_any': 0, 'top3_exact': 0}
    total = 0

    for race_id in df_test['race_id'].unique():
        race_data = df_test[df_test['race_id'] == race_id]
        scores = trainer.predict_race(race_data)
        positions = pd.to_numeric(race_data['finish_position'], errors='coerce').values

        pred_rank = np.argsort(scores)[::-1]
        actual_rank = np.argsort(positions)

        if pred_rank[0] == actual_rank[0]: results['top1'] += 1
        if set(pred_rank[:2]) == set(actual_rank[:2]): results['top2_any'] += 1
        if set(pred_rank[:3]) == set(actual_rank[:3]): results['top3_any'] += 1
        if list(pred_rank[:3]) == list(actual_rank[:3]): results['top3_exact'] += 1
        total += 1

    logging.info(f"\n結果 ({total}レース):")
    logging.info(f"  単勝:   {results['top1']/total:.1%} ({results['top1']}/{total})")
    logging.info(f"  馬連:   {results['top2_any']/total:.1%} ({results['top2_any']}/{total})")
    logging.info(f"  三連複: {results['top3_any']/total:.1%} ({results['top3_any']}/{total})")
    logging.info(f"  三連単: {results['top3_exact']/total:.1%} ({results['top3_exact']}/{total})")

    # 保存
    output_dir = Path("models/transformer_odds")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trainer.save(output_dir / f"transformer_odds_{timestamp}.pth")

    with open(output_dir / f"results_{timestamp}.json", 'w') as f:
        json.dump({k: v/total for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()

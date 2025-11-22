"""
ディープラーニングモデル（LSTM, Transformer）
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import yaml
from pathlib import Path
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceDataset(Dataset):
    """競馬データセット"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_features: np.ndarray = None):
        """
        Args:
            features: 静的特徴量 (n_samples, n_features)
            targets: ターゲット (n_samples,)
            sequence_features: 系列特徴量 (n_samples, seq_len, n_seq_features)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        self.sequence_features = None

        if sequence_features is not None:
            self.sequence_features = torch.FloatTensor(sequence_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.sequence_features is not None:
            return self.features[idx], self.sequence_features[idx], self.targets[idx]
        return self.features[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM based競馬予測モデル"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 18,
                 dropout: float = 0.3, bidirectional: bool = True):
        """
        Args:
            input_size: 入力特徴量の次元数
            hidden_size: LSTM隠れ層のサイズ
            num_layers: LSTMの層数
            num_classes: 出力クラス数（着順数）
            dropout: ドロップアウト率
            bidirectional: 双方向LSTMを使用するか
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 全結合層
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        """
        Args:
            x: 入力テンソル (batch_size, seq_len, input_size)
            hidden: 隠れ状態（オプション）

        Returns:
            出力テンソル (batch_size, num_classes)
        """
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # 最後のタイムステップの出力を使用
        lstm_out = lstm_out[:, -1, :]

        # 全結合層
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerModel(nn.Module):
    """Transformer based競馬予測モデル"""

    def __init__(self, input_size: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 512, num_classes: int = 18,
                 dropout: float = 0.1):
        """
        Args:
            input_size: 入力特徴量の次元数
            d_model: Transformerの次元数
            nhead: マルチヘッドアテンションのヘッド数
            num_layers: Transformerエンコーダの層数
            dim_feedforward: フィードフォワード層の次元数
            num_classes: 出力クラス数（着順数）
            dropout: ドロップアウト率
        """
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # 入力埋め込み層
        self.input_embedding = nn.Linear(input_size, d_model)

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformerエンコーダ
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 出力層
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 入力テンソル (batch_size, seq_len, input_size)

        Returns:
            出力テンソル (batch_size, num_classes)
        """
        # 埋め込み
        x = self.input_embedding(x) * np.sqrt(self.d_model)

        # 位置エンコーディング
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer_encoder(x)

        # 平均プーリング
        x = torch.mean(x, dim=1)

        # 出力層
        x = self.dropout(x)
        out = self.fc(x)

        return out


class PositionalEncoding(nn.Module):
    """位置エンコーディング"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置エンコーディングを計算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DeepLearningTrainer:
    """ディープラーニングモデルのトレーナー"""

    def __init__(self, model: nn.Module, config_path: str = "configs/config.yaml",
                 device: str = None):
        """
        Args:
            model: PyTorchモデル
            config_path: 設定ファイルのパス
            device: デバイス（"cuda" or "cpu"）
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.training_config = self.config['training']
        self.batch_size = self.training_config['batch_size']
        self.epochs = self.training_config['epochs']
        self.learning_rate = self.training_config['learning_rate']

        logger.info(f"Using device: {self.device}")

    def train(self, train_dataset: RaceDataset, val_dataset: RaceDataset = None) -> Dict:
        """
        モデルを訓練

        Args:
            train_dataset: 訓練データセット
            val_dataset: 検証データセット

        Returns:
            訓練履歴
        """
        logger.info("Starting deep learning model training")

        # データローダー
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

        # 損失関数と最適化手法
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 学習率スケジューラ
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 訓練ループ
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.training_config['early_stopping_patience']

        for epoch in range(self.epochs):
            # 訓練フェーズ
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 検証フェーズ
            if val_loader:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # 学習率スケジューリング
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # ベストモデルを保存
                    self.save_checkpoint('best_model_checkpoint.pth')
                else:
                    patience_counter += 1

                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )

        logger.info("Training completed")
        return history

    def _train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            if len(batch) == 3:  # 系列データがある場合
                features, seq_features, targets = batch
                features = features.to(self.device)
                seq_features = seq_features.to(self.device)
                targets = targets.to(self.device)

                # 順伝播
                outputs = self.model(seq_features)
            else:  # 系列データがない場合
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)

                # 順伝播（系列として扱う）
                features = features.unsqueeze(1)
                outputs = self.model(features)

            # 損失計算
            loss = criterion(outputs, targets)

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 統計
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    features, seq_features, targets = batch
                    features = features.to(self.device)
                    seq_features = seq_features.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(seq_features)
                else:
                    features, targets = batch
                    features = features.to(self.device)
                    targets = targets.to(self.device)

                    features = features.unsqueeze(1)
                    outputs = self.model(features)

                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def predict(self, dataset: RaceDataset) -> np.ndarray:
        """予測"""
        self.model.eval()
        predictions = []

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    features, seq_features, _ = batch
                    seq_features = seq_features.to(self.device)
                    outputs = self.model(seq_features)
                else:
                    features, _ = batch
                    features = features.to(self.device)
                    features = features.unsqueeze(1)
                    outputs = self.model(features)

                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions) + 1  # 0始まりから1始まりに変換

    def predict_proba(self, dataset: RaceDataset) -> np.ndarray:
        """確率を予測"""
        self.model.eval()
        probabilities = []

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    features, seq_features, _ = batch
                    seq_features = seq_features.to(self.device)
                    outputs = self.model(seq_features)
                else:
                    features, _ = batch
                    features = features.to(self.device)
                    features = features.unsqueeze(1)
                    outputs = self.model(features)

                proba = F.softmax(outputs, dim=1)
                probabilities.extend(proba.cpu().numpy())

        return np.array(probabilities)

    def save_checkpoint(self, filepath: str):
        """チェックポイントを保存"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config
        }, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded from {filepath}")


if __name__ == "__main__":
    # 使用例
    # ダミーデータで動作確認
    input_size = 50
    seq_len = 5
    num_samples = 1000
    num_classes = 18

    # ダミーデータ生成
    X_train = np.random.randn(num_samples, seq_len, input_size)
    y_train = np.random.randint(0, num_classes, num_samples)

    X_val = np.random.randn(200, seq_len, input_size)
    y_val = np.random.randint(0, num_classes, 200)

    # データセット作成
    train_dataset = RaceDataset(
        np.random.randn(num_samples, input_size),
        y_train,
        X_train
    )
    val_dataset = RaceDataset(
        np.random.randn(200, input_size),
        y_val,
        X_val
    )

    # LSTMモデル
    lstm_model = LSTMModel(input_size=input_size, hidden_size=128, num_classes=num_classes)
    lstm_trainer = DeepLearningTrainer(lstm_model)
    history = lstm_trainer.train(train_dataset, val_dataset)

    print("LSTM Training completed")

    # Transformerモデル
    # transformer_model = TransformerModel(input_size=input_size, d_model=128, num_classes=num_classes)
    # transformer_trainer = DeepLearningTrainer(transformer_model)
    # history = transformer_trainer.train(train_dataset, val_dataset)

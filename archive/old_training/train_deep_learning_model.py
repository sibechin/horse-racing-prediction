"""
深層学習モデル: PyTorch Ranking Neural Network

G1専用データでニューラルネットワークを訓練
- Multi-layer Feed-Forward Network
- Pairwise Ranking Loss
- Group K-Fold Cross-Validation
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RaceDataset(Dataset):
    """レースデータセット"""

    def __init__(self, X, y, groups):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.groups = groups

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RankingNet(nn.Module):
    """ランキング用ニューラルネットワーク"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(RankingNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 出力層: ランキングスコア
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class PairwiseRankingLoss(nn.Module):
    """Pairwise Ranking Loss"""

    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, labels, groups):
        """
        scores: モデル出力 (N,)
        labels: 実際の順位 (N,)
        groups: レースID (N,)
        """
        loss = 0.0
        n_pairs = 0

        unique_groups = torch.unique(groups)

        for group_id in unique_groups:
            mask = (groups == group_id)
            group_scores = scores[mask]
            group_labels = labels[mask]

            # ペアワイズ比較
            n_horses = len(group_scores)
            for i in range(n_horses):
                for j in range(i + 1, n_horses):
                    # より良い順位(小さい値)の馬がより高いスコアを持つべき
                    if group_labels[i] < group_labels[j]:
                        # i の方が順位が良い → scores[i] > scores[j] であるべき
                        pair_loss = torch.clamp(self.margin - (group_scores[i] - group_scores[j]), min=0)
                    elif group_labels[i] > group_labels[j]:
                        # j の方が順位が良い → scores[j] > scores[i] であるべき
                        pair_loss = torch.clamp(self.margin - (group_scores[j] - group_scores[i]), min=0)
                    else:
                        continue  # 同順位の場合はスキップ

                    loss += pair_loss
                    n_pairs += 1

        return loss / max(n_pairs, 1)


class DeepRankingModel:
    """深層学習ランキングモデル"""

    def __init__(self, output_dir='models/deep_learning'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = []
        self.scalers = []

        self.feature_names = [
            'distance', 'venue_code', 'horse_number', 'weight', 'popularity', 'odds',
            'time_seconds', 'age', 'year', 'jockey_win_rate', 'jockey_top3_rate',
            'jockey_track_win_rate', 'horse_win_rate', 'horse_top3_rate',
            'horse_race_count', 'horse_dist_win_rate', 'field_size', 'race_avg_odds',
            'popularity_rank', 'month', 'day_of_week', 'track_type_encoded',
            'track_condition_encoded', 'weather_encoded', 'sex_encoded',
            'distance_category_encoded', 'season_encoded', 'track_combined_encoded',
            'time_weight'
        ]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用デバイス: {self.device}")

    def train_all_models(self, data_path: str, n_splits=5):
        """全モデル訓練"""
        logging.info("="*60)
        logging.info("深層学習モデル訓練開始")
        logging.info("="*60)

        # データ読み込み
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        logging.info(f"\nG1データ: {len(df)} records, {df['race_id'].nunique()} races")

        # finish_positionフィルタリング
        df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
        df = df[df['finish_position'].notna()].copy()
        logging.info(f"有効なデータ: {len(df)} records")

        # 特徴量準備
        X = df[self.feature_names].copy()
        y = df['finish_position'].values
        groups = df['race_id'].values

        # 欠損値処理
        for col in X.columns:
            if X[col].isnull().any():
                X.loc[:, col] = X[col].fillna(X[col].median())

        X = X.values

        # Group K-Fold
        gkf = GroupKFold(n_splits=n_splits)

        logging.info(f"\n{n_splits}-Fold Cross-Validation")
        logging.info("Architecture: [29] -> [128] -> [64] -> [32] -> [1]")
        logging.info("Loss: Pairwise Ranking Loss")
        logging.info("Optimizer: Adam with ReduceLROnPlateau")

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Fold {fold_idx}/{n_splits}")
            logging.info(f"{'='*60}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]
            groups_val = groups[val_idx]

            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            self.scalers.append(scaler)

            # モデル訓練
            model = self._train_single_model(
                X_train_scaled, y_train, groups_train,
                X_val_scaled, y_val, groups_val,
                fold_idx
            )

            self.models.append(model)

            # Validation評価
            self._evaluate_fold(model, X_val_scaled, y_val, groups_val, fold_idx)

        logging.info(f"\n{'='*60}")
        logging.info("全モデル訓練完了!")
        logging.info(f"{'='*60}")
        logging.info(f"訓練モデル数: {len(self.models)}")

        # モデル保存
        self._save_models()

    def _train_single_model(self, X_train, y_train, groups_train,
                           X_val, y_val, groups_val, fold_idx):
        """単一モデル訓練"""

        # モデル初期化
        model = RankingNet(
            input_dim=len(self.feature_names),
            hidden_dims=[128, 64, 32],
            dropout=0.3
        ).to(self.device)

        # 損失関数とオプティマイザ
        criterion = PairwiseRankingLoss(margin=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # データローダー (バッチサイズ=全データ)
        train_dataset = RaceDataset(X_train, y_train, groups_train)
        train_loader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=False)

        # 訓練ループ
        n_epochs = 200
        best_val_ndcg = 0.0
        patience_counter = 0
        early_stop_patience = 30

        for epoch in range(n_epochs):
            model.train()

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                groups_tensor = torch.LongTensor(groups_train).to(self.device)

                optimizer.zero_grad()

                scores = model(batch_X)
                loss = criterion(scores, batch_y, groups_tensor)

                loss.backward()
                optimizer.step()

            # Validation評価 (10エポックごと)
            if (epoch + 1) % 10 == 0:
                val_ndcg = self._compute_ndcg(model, X_val, y_val, groups_val)
                scheduler.step(val_ndcg)

                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stop_patience:
                    logging.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        logging.info(f"  訓練完了: Best Val NDCG@1 = {best_val_ndcg:.4f}")

        return model

    def _compute_ndcg(self, model, X_val, y_val, groups_val):
        """NDCG@1計算"""
        model.eval()

        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            predictions = model(X_val_tensor).cpu().numpy()

        unique_races = pd.Series(groups_val).unique()
        ndcg_scores = []

        for race_id in unique_races:
            race_mask = (groups_val == race_id)
            race_y = y_val[race_mask]
            race_pred = predictions[race_mask]

            if len(race_y) > 0:
                predicted_order = np.argsort(-race_pred)
                true_winner_idx = race_y.argmin()
                ndcg_at_1 = 1.0 if predicted_order[0] == true_winner_idx else 0.0
                ndcg_scores.append(ndcg_at_1)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def _evaluate_fold(self, model, X_val, y_val, groups_val, fold_idx):
        """Fold評価"""
        model.eval()

        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            predictions = model(X_val_tensor).cpu().numpy()

        unique_races = pd.Series(groups_val).unique()
        ndcg_scores = []

        for race_id in unique_races:
            race_mask = (groups_val == race_id)
            race_y = y_val[race_mask]
            race_pred = predictions[race_mask]

            if len(race_y) > 0:
                predicted_order = np.argsort(-race_pred)
                true_winner_idx = race_y.argmin()
                ndcg_at_1 = 1.0 if predicted_order[0] == true_winner_idx else 0.0
                ndcg_scores.append(ndcg_at_1)

        mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        logging.info(f"\nFold {fold_idx} Validation NDCG@1: {mean_ndcg:.4f}")

    def _save_models(self):
        """モデル保存"""
        logging.info(f"\nモデル保存中...")

        for i, (model, scaler) in enumerate(zip(self.models, self.scalers), 1):
            # PyTorchモデル
            model_path = self.output_dir / f'ranking_net_fold{i}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_dim': len(self.feature_names),
                    'hidden_dims': [128, 64, 32],
                    'dropout': 0.3
                }
            }, str(model_path))

            # Scaler
            import joblib
            scaler_path = self.output_dir / f'scaler_fold{i}.pkl'
            joblib.dump(scaler, str(scaler_path))

        logging.info(f"全モデル保存完了: {self.output_dir}")

    def predict_ensemble(self, X):
        """アンサンブル予測"""
        predictions = []

        for model, scaler in zip(self.models, self.scalers):
            model.eval()

            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()

            predictions.append(pred)

        # 平均予測
        ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("深層学習モデル訓練 (PyTorch Ranking NN)")
    logging.info("="*60)
    logging.info("\nアーキテクチャ:")
    logging.info("  入力層: 29 features")
    logging.info("  隠れ層1: 128 units (ReLU + BatchNorm + Dropout 0.3)")
    logging.info("  隠れ層2: 64 units (ReLU + BatchNorm + Dropout 0.3)")
    logging.info("  隠れ層3: 32 units (ReLU + BatchNorm + Dropout 0.3)")
    logging.info("  出力層: 1 unit (ranking score)")
    logging.info("\n損失関数: Pairwise Ranking Loss (margin=1.0)")
    logging.info("最適化: Adam (lr=0.001, weight_decay=1e-5)")
    logging.info("学習率スケジューリング: ReduceLROnPlateau")
    logging.info("Early Stopping: Patience=30 epochs")

    # モデル初期化
    dl_model = DeepRankingModel(output_dir='models/deep_learning')

    # データパス
    data_path = "data/processed/g1_races_historical_processed.csv"
    logging.info(f"\n使用データ: {data_path}")

    # 全モデル訓練
    dl_model.train_all_models(data_path, n_splits=5)

    # 結果保存
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'PyTorch Ranking Neural Network',
        'architecture': '[29] -> [128] -> [64] -> [32] -> [1]',
        'loss_function': 'Pairwise Ranking Loss',
        'optimizer': 'Adam (lr=0.001, weight_decay=1e-5)',
        'training_data': 'G1 Races 2016-2024 (~1,900 records, 102 races)',
        'n_folds': 5,
        'device': str(dl_model.device)
    }

    results_file = dl_model.output_dir / 'training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n結果保存: {results_file}")
    logging.info("\n✅ 深層学習モデル訓練完了!")
    logging.info("\n次のステップ:")
    logging.info("  python test_deep_learning_model.py")
    logging.info("  (2025年G1データでテスト)")


if __name__ == "__main__":
    main()

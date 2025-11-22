"""
全モデル(LightGBM, Transformer, Attention)を2025年データでテスト
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import torch
import torch.nn as nn
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Transformer & Attentionモデルの定義（train_transformer_attention_models.pyから）
class TransformerRankingModel(nn.Module):
    """Transformerベースのランキングモデル"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerRankingModel, self).__init__()

        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

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

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, mask=None):
        x = self.input_embedding(x)

        if mask is not None:
            mask = ~mask

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        scores = self.output_layer(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool() if mask.dtype == torch.bool else mask.bool(), float('-inf'))

        return scores


class AttentionRankingModel(nn.Module):
    """Attentionベースのランキングモデル"""
    def __init__(self, input_dim, hidden_dim=64, num_attention_heads=4, dropout=0.1):
        super(AttentionRankingModel, self).__init__()

        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        x = self.input_embedding(x)

        key_padding_mask = ~mask if mask is not None else None

        attn_output, _ = self.multihead_attention(
            x, x, x,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + attn_output)

        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        scores = self.output_layer(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        return scores


def evaluate_ranking_model(y_true, y_pred, race_groups):
    """ランキングモデルの評価"""
    from sklearn.metrics import ndcg_score

    metrics = {}

    # NDCG Score
    ndcg_scores = []
    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() > 1:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]
            race_y_true_rel = race_y_true - 1

            try:
                ndcg = ndcg_score([race_y_true_rel], [race_y_pred], k=3)
                ndcg_scores.append(ndcg)
            except:
                pass

    metrics['ndcg@3'] = np.mean(ndcg_scores) if ndcg_scores else 0.0

    # Top1 Accuracy
    top1_correct = 0
    total_races = 0

    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() > 0:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]

            predicted_winner_idx = np.argmax(race_y_pred)
            actual_winner_idx = np.argmin(race_y_true)

            if predicted_winner_idx == actual_winner_idx:
                top1_correct += 1

            total_races += 1

    metrics['top1_accuracy'] = top1_correct / total_races if total_races > 0 else 0.0

    # Top3 Hit Rate
    top3_correct = 0
    total_races = 0

    for race_id in race_groups.unique():
        mask = race_groups == race_id
        if mask.sum() >= 3:
            race_y_true = y_true[mask].values
            race_y_pred = y_pred[mask]

            predicted_top3 = set(np.argsort(race_y_pred)[-3:])
            actual_top3 = set(np.argsort(race_y_true)[:3])

            if len(predicted_top3 & actual_top3) > 0:
                top3_correct += 1

            total_races += 1

    metrics['top3_hit_rate'] = top3_correct / total_races if total_races > 0 else 0.0

    return metrics


def test_lightgbm_model(model_file, test_file):
    """LightGBMモデルをテスト"""
    logging.info("\n" + "="*60)
    logging.info("LightGBMモデルのテスト")
    logging.info("="*60)

    # モデル読み込み
    model = joblib.load(model_file)
    logging.info(f"モデル読み込み: {model_file}")

    # テストデータ読み込み
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    logging.info(f"テストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races")

    # 特徴量（オッズ除外）
    feature_columns = [
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight',
        'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size',
        'race_avg_weight',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]

    available_features = [col for col in feature_columns if col in df_test.columns]
    X_test = df_test[available_features].copy()
    y_test = df_test['finish_position_numeric'].copy()
    race_groups_test = df_test['race_id'].copy()

    # 予測
    y_pred = model.predict(X_test)

    # 評価
    metrics = evaluate_ranking_model(y_test, y_pred, race_groups_test)

    logging.info("\n評価結果:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    return metrics, y_pred


def test_pytorch_model(model, model_file, test_file, model_name):
    """PyTorchモデル(Transformer/Attention)をテスト"""
    logging.info("\n" + "="*60)
    logging.info(f"{model_name}モデルのテスト")
    logging.info("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデル読み込み
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logging.info(f"モデル読み込み: {model_file}")

    # テストデータ読み込み
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    logging.info(f"テストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races")

    # 特徴量（オッズ除外）
    feature_columns = [
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight',
        'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size',
        'race_avg_weight',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]

    available_features = [col for col in feature_columns if col in df_test.columns]
    X_test = df_test[available_features].copy()
    y_test = df_test['finish_position_numeric'].copy()
    race_groups_test = df_test['race_id'].copy()

    # レース単位で予測
    all_predictions = []

    with torch.no_grad():
        for race_id in race_groups_test.unique():
            mask = race_groups_test == race_id
            race_features = X_test[mask].values

            # Tensor化
            race_tensor = torch.FloatTensor(race_features).unsqueeze(0).to(device)
            race_mask = torch.ones(1, len(race_features), dtype=torch.bool).to(device)

            # 予測
            scores = model(race_tensor, race_mask)
            scores = scores.cpu().numpy()[0]

            all_predictions.extend(scores)

    y_pred = np.array(all_predictions)

    # 評価
    metrics = evaluate_ranking_model(y_test, y_pred, race_groups_test)

    logging.info("\n評価結果:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.4f}")

    return metrics, y_pred


def main():
    """メイン実行"""
    logging.info("="*60)
    logging.info("全モデル(LightGBM, Transformer, Attention)の2025年テスト")
    logging.info("="*60)

    # ファイルパス
    test_file = 'data/processed/keibalab_g1_2025_processed.csv'

    # 訓練済みモデルのパス
    lightgbm_model = 'models/keibalab_g1_no_odds/no_odds_model_2000_2024_20251120_194630.pkl'
    transformer_model = 'models/transformer/transformer_model_best_20251120_195259.pth'
    attention_model = 'models/attention/attention_model_best_20251120_195443.pth'

    # ファイルの存在確認
    if not Path(test_file).exists():
        logging.error(f"テストデータが見つかりません: {test_file}")
        return

    # 特徴量数を取得（テストデータから）
    df_test = pd.read_csv(test_file, encoding='utf-8-sig')
    feature_columns = [
        'year', 'month', 'day_of_week',
        'venue_code', 'race_number',
        'sex_encoded', 'age',
        'weight',
        'last_3f', 'horse_weight_kg', 'horse_weight_change',
        'field_size',
        'race_avg_weight',
        'trainer_encoded', 'trainer_frequency',
        'jockey_encoded', 'jockey_frequency',
        'race_type_encoded'
    ]
    available_features = [col for col in feature_columns if col in df_test.columns]
    input_dim = len(available_features)

    logging.info(f"\nテストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races")
    logging.info(f"特徴量数: {input_dim}")

    # 全モデルのテスト結果を保存
    results = {}

    # 1. LightGBMテスト
    if Path(lightgbm_model).exists():
        lgb_metrics, _ = test_lightgbm_model(lightgbm_model, test_file)
        results['LightGBM'] = lgb_metrics
    else:
        logging.warning(f"LightGBMモデルが見つかりません: {lightgbm_model}")

    # 2. Transformerテスト
    if Path(transformer_model).exists():
        transformer = TransformerRankingModel(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
        trans_metrics, _ = test_pytorch_model(transformer, transformer_model, test_file, "Transformer")
        results['Transformer'] = trans_metrics
    else:
        logging.warning(f"Transformerモデルが見つかりません: {transformer_model}")

    # 3. Attentionテスト
    if Path(attention_model).exists():
        attention = AttentionRankingModel(input_dim=input_dim, hidden_dim=64, num_attention_heads=4)
        attn_metrics, _ = test_pytorch_model(attention, attention_model, test_file, "Attention")
        results['Attention'] = attn_metrics
    else:
        logging.warning(f"Attentionモデルが見つかりません: {attention_model}")

    # 結果の比較
    logging.info("\n" + "="*60)
    logging.info("2025年テストデータ性能比較")
    logging.info("="*60)

    comparison = pd.DataFrame(results).T
    comparison['NDCG@3'] = comparison['ndcg@3']
    comparison['1着的中率'] = comparison['top1_accuracy'] * 100
    comparison['3着内的中率'] = comparison['top3_hit_rate'] * 100
    comparison = comparison[['NDCG@3', '1着的中率', '3着内的中率']]

    print("\n")
    print(comparison.to_string())

    # 最良モデルの特定
    best_ndcg = comparison['NDCG@3'].idxmax()
    best_top1 = comparison['1着的中率'].idxmax()
    best_top3 = comparison['3着内的中率'].idxmax()

    logging.info(f"\n最良モデル:")
    logging.info(f"  NDCG@3: {best_ndcg} ({comparison.loc[best_ndcg, 'NDCG@3']:.4f})")
    logging.info(f"  1着的中率: {best_top1} ({comparison.loc[best_top1, '1着的中率']:.2f}%)")
    logging.info(f"  3着内的中率: {best_top3} ({comparison.loc[best_top3, '3着内的中率']:.2f}%)")

    # 結果をファイルに保存
    output_dir = Path('models/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'2025_test_results_{timestamp}.txt'

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("2025年G1データテスト結果\n")
        f.write("="*60 + "\n\n")
        f.write(f"テストデータ: {len(df_test)} records, {df_test['race_id'].nunique()} races\n\n")
        f.write("モデル性能比較:\n")
        f.write(comparison.to_string())
        f.write("\n\n")
        f.write(f"最良モデル:\n")
        f.write(f"  NDCG@3: {best_ndcg} ({comparison.loc[best_ndcg, 'NDCG@3']:.4f})\n")
        f.write(f"  1着的中率: {best_top1} ({comparison.loc[best_top1, '1着的中率']:.2f}%)\n")
        f.write(f"  3着内的中率: {best_top3} ({comparison.loc[best_top3, '3着内的中率']:.2f}%)\n")

    logging.info(f"\n結果保存: {results_file}")

    logging.info("\n" + "="*60)
    logging.info("テスト完了!")
    logging.info("="*60)


if __name__ == "__main__":
    main()

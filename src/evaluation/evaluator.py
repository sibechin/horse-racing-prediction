"""
モデル評価モジュール
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RacePredictor Evaluator:
    """競馬予測モデルの評価"""

    def __init__(self, output_dir: str = "logs"):
        """
        Args:
            output_dir: 評価結果の出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: np.ndarray = None) -> Dict:
        """
        モデルを評価

        Args:
            y_true: 実際の着順
            y_pred: 予測された着順
            y_proba: 予測確率（オプション）

        Returns:
            評価指標の辞書
        """
        logger.info("Starting model evaluation")

        metrics = {}

        # 1. 正確度（完全一致）
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        # 2. Top-3正確度（3着以内に入る確率）
        metrics['top3_accuracy'] = self._calculate_top_k_accuracy(y_true, y_pred, k=3)
        logger.info(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")

        # 3. 平均絶対誤差（MAE）
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        logger.info(f"MAE: {metrics['mae']:.4f}")

        # 4. 二乗平均平方根誤差（RMSE）
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        logger.info(f"RMSE: {metrics['rmse']:.4f}")

        # 5. NDCG（正規化割引累積利得）
        if y_proba is not None:
            # 真のランキングをバイナリ関連性スコアに変換
            # 1位=1.0, 2位以降=0.0
            y_true_binary = (y_true == 1).astype(int)

            # y_probaが2次元配列の場合、1位の確率を使用
            if y_proba.ndim == 2:
                y_score = y_proba[:, 0]  # 1位の確率
            else:
                y_score = y_proba

            try:
                # NDCGを計算（レースごとにグループ化して計算する必要があるが、簡易版）
                metrics['ndcg'] = ndcg_score([y_true_binary], [y_score])
                logger.info(f"NDCG: {metrics['ndcg']:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate NDCG: {e}")
                metrics['ndcg'] = None

        # 6. 着順別の精度
        metrics['position_accuracy'] = self._calculate_position_accuracy(y_true, y_pred)

        # 7. 人気別の的中率（オプション、人気データがある場合）

        logger.info("Evaluation completed")

        return metrics

    @staticmethod
    def _calculate_top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
        """
        Top-K正確度を計算

        Args:
            y_true: 実際の着順
            y_pred: 予測された着順
            k: 閾値

        Returns:
            Top-K正確度
        """
        # 実際の着順がk着以内で、予測もk着以内なら正解
        true_in_top_k = y_true <= k
        pred_in_top_k = y_pred <= k

        # 両方がTop-Kに入っているかどうか
        correct = true_in_top_k & pred_in_top_k

        return correct.sum() / len(y_true)

    @staticmethod
    def _calculate_position_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        着順別の精度を計算

        Args:
            y_true: 実際の着順
            y_pred: 予測された着順

        Returns:
            着順別の精度の辞書
        """
        position_acc = {}

        for pos in [1, 2, 3, 4, 5]:
            mask = y_true == pos
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
                position_acc[f'position_{pos}_accuracy'] = acc

        return position_acc

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             max_position: int = 10, save_path: str = None):
        """
        混同行列をプロット

        Args:
            y_true: 実際の着順
            y_pred: 予測された着順
            max_position: プロットする最大着順
            save_path: 保存パス
        """
        # 着順をmax_position以下に制限
        mask = (y_true <= max_position) & (y_pred <= max_position)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        # 混同行列を計算
        cm = confusion_matrix(y_true_filtered, y_pred_filtered,
                            labels=list(range(1, max_position + 1)))

        # プロット
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(1, max_position + 1),
                   yticklabels=range(1, max_position + 1))
        plt.title('Confusion Matrix (Finish Position)')
        plt.ylabel('True Position')
        plt.xlabel('Predicted Position')

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')

        plt.close()

    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: str = None):
        """
        予測誤差の分布をプロット

        Args:
            y_true: 実際の着順
            y_pred: 予測された着順
            save_path: 保存パス
        """
        errors = y_pred - y_true

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Prediction Error (Predicted - True)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error distribution plot saved to {save_path}")
        else:
            plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')

        plt.close()

    def plot_position_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                              max_position: int = 10, save_path: str = None):
        """
        着順別の精度をプロット

        Args:
            y_true: 実際の着順
            y_pred: 予測された着順
            max_position: プロットする最大着順
            save_path: 保存パス
        """
        positions = range(1, max_position + 1)
        accuracies = []

        for pos in positions:
            mask = y_true == pos
            if mask.sum() > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)

        plt.figure(figsize=(10, 6))
        plt.bar(positions, accuracies, edgecolor='black', alpha=0.7)
        plt.title('Accuracy by Finish Position')
        plt.xlabel('Finish Position')
        plt.ylabel('Accuracy')
        plt.xticks(positions)
        plt.grid(True, alpha=0.3, axis='y')

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Position accuracy plot saved to {save_path}")
        else:
            plt.savefig(self.output_dir / 'position_accuracy.png', dpi=300, bbox_inches='tight')

        plt.close()

    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: np.ndarray = None, model_name: str = "Model") -> str:
        """
        評価レポートを生成

        Args:
            y_true: 実際の着順
            y_pred: 予測された着順
            y_proba: 予測確率
            model_name: モデル名

        Returns:
            レポート文字列
        """
        metrics = self.evaluate(y_true, y_pred, y_proba)

        report = f"""
========================================
競馬予測モデル評価レポート
モデル名: {model_name}
========================================

基本評価指標:
- 正確度 (Accuracy): {metrics['accuracy']:.4f}
- Top-3正確度: {metrics['top3_accuracy']:.4f}
- 平均絶対誤差 (MAE): {metrics['mae']:.4f}
- 二乗平均平方根誤差 (RMSE): {metrics['rmse']:.4f}
"""

        if metrics.get('ndcg') is not None:
            report += f"- NDCG: {metrics['ndcg']:.4f}\n"

        report += "\n着順別精度:\n"
        for key, value in metrics['position_accuracy'].items():
            position = key.replace('position_', '').replace('_accuracy', '')
            report += f"- {position}位: {value:.4f}\n"

        # グラフを生成
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_error_distribution(y_true, y_pred)
        self.plot_position_accuracy(y_true, y_pred)

        report += f"\n評価グラフは {self.output_dir} に保存されました。\n"
        report += "========================================\n"

        # レポートをファイルに保存
        report_path = self.output_dir / f'{model_name}_evaluation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Evaluation report saved to {report_path}")

        return report


if __name__ == "__main__":
    # 使用例
    evaluator = RacePredictorEvaluator(output_dir="../../logs")

    # ダミーデータで動作確認
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.randint(1, 11, n_samples)
    y_pred = y_true + np.random.randint(-2, 3, n_samples)
    y_pred = np.clip(y_pred, 1, 18)

    # 評価
    report = evaluator.generate_report(y_true, y_pred, model_name="TestModel")
    print(report)

"""
高度な特徴量エンジニアリング

G1_MODEL_COMPARISON_REPORT.mdの推奨事項に基づく特徴量追加:
1. レース展開特徴（位置取り傾向、ペース適性）
2. 馬体状態特徴（馬体重推移、休養期間）
3. 相互作用特徴（騎手×コース、馬×距離×馬場状態）
4. 時系列特徴（調子の上昇/下降）

現在の29特徴量から50+特徴量に拡張
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AdvancedFeatureEngineer:
    """高度な特徴量エンジニアリング"""

    def __init__(self):
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全特徴量を生成

        Args:
            df: 元データ（raw data）

        Returns:
            特徴量追加後のDataFrame
        """
        logging.info("=" * 60)
        logging.info("高度な特徴量エンジニアリング開始")
        logging.info("=" * 60)
        logging.info(f"入力データ: {len(df)} records")
        logging.info("")

        # データのコピー
        df = df.copy()

        # 基本的な前処理
        df = self._preprocess_basic_fields(df)

        # 1. レース展開特徴
        logging.info("1/4: レース展開特徴を生成中...")
        df = self._add_race_pace_features(df)

        # 2. 馬体状態特徴
        logging.info("2/4: 馬体状態特徴を生成中...")
        df = self._add_horse_condition_features(df)

        # 3. 相互作用特徴
        logging.info("3/4: 相互作用特徴を生成中...")
        df = self._add_interaction_features(df)

        # 4. 時系列特徴
        logging.info("4/4: 時系列特徴を生成中...")
        df = self._add_time_series_features(df)

        logging.info("")
        logging.info("=" * 60)
        logging.info("特徴量エンジニアリング完了!")
        logging.info("=" * 60)
        logging.info(f"出力データ: {len(df)} records")
        logging.info(f"追加された特徴量数: {len(df.columns)} columns")
        logging.info("=" * 60)

        return df

    def _preprocess_basic_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本フィールドの前処理"""
        # 日付を datetime に変換
        if 'race_date' in df.columns:
            df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')

        # 数値フィールドの変換
        numeric_fields = ['finish_position', 'horse_number', 'popularity', 'odds',
                          'weight', 'distance', 'horse_weight']

        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')

        # 馬体重の変化を抽出（例: 480(+5) → 480, +5）
        if 'horse_weight' in df.columns and df['horse_weight'].dtype == 'object':
            df['horse_weight_raw'] = df['horse_weight'].astype(str)
            df['horse_weight_value'] = df['horse_weight_raw'].str.extract(r'(\d+)', expand=False).astype(float)
            df['horse_weight_change'] = df['horse_weight_raw'].str.extract(r'\(([+\-]?\d+)\)', expand=False).astype(float)
        else:
            df['horse_weight_value'] = df['horse_weight']
            df['horse_weight_change'] = np.nan

        return df

    def _add_race_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レース展開特徴を追加

        - 過去のaverage finish position（前走、前々走）
        - 逃げ/先行/差し/追込のスタイル推定
        - コーナー通過順位の傾向
        """
        # 馬ごとに過去の成績を集計
        if 'horse_name' in df.columns and 'race_date' in df.columns and 'finish_position' in df.columns:
            df = df.sort_values(['horse_name', 'race_date'])

            # 前走の着順
            df['prev_finish_position'] = df.groupby('horse_name')['finish_position'].shift(1)

            # 前々走の着順
            df['prev2_finish_position'] = df.groupby('horse_name')['finish_position'].shift(2)

            # 過去3走の平均着順
            df['avg_finish_position_last3'] = (
                df.groupby('horse_name')['finish_position']
                .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
            )

            # 過去5走の平均着順
            df['avg_finish_position_last5'] = (
                df.groupby('horse_name')['finish_position']
                .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
            )

            # 着順の標準偏差（安定性の指標）
            df['std_finish_position_last5'] = (
                df.groupby('horse_name')['finish_position']
                .transform(lambda x: x.rolling(window=5, min_periods=2).std().shift(1))
            )

            # 直近の調子（過去3走の着順トレンド: 改善傾向 or 悪化傾向）
            df['finish_position_trend'] = (
                df.groupby('horse_name')['finish_position']
                .transform(lambda x: x.rolling(window=3, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
                ).shift(1))
            )

        # 枠番による有利不利（内枠 vs 外枠）
        if 'frame_number' in df.columns:
            df['is_inner_frame'] = (df['frame_number'] <= 3).astype(int)  # 1-3枠 = 内枠
            df['is_outer_frame'] = (df['frame_number'] >= 6).astype(int)  # 6-8枠 = 外枠

        return df

    def _add_horse_condition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        馬体状態特徴を追加

        - 馬体重の推移（増加/減少トレンド）
        - 休養期間（前走からの日数）
        - 連闘フラグ
        """
        if 'horse_name' in df.columns and 'race_date' in df.columns:
            df = df.sort_values(['horse_name', 'race_date'])

            # 前走からの休養期間（日数）
            df['days_since_last_race'] = (
                df.groupby('horse_name')['race_date']
                .diff()
                .dt.days
            )

            # 連闘フラグ（2週間以内）
            df['is_quick_turnaround'] = (df['days_since_last_race'] <= 14).astype(int)

            # 長期休養フラグ（3ヶ月以上）
            df['is_long_rest'] = (df['days_since_last_race'] >= 90).astype(int)

            # 馬体重の推移
            if 'horse_weight_value' in df.columns:
                # 前走の馬体重
                df['prev_horse_weight'] = df.groupby('horse_name')['horse_weight_value'].shift(1)

                # 馬体重の変化（前走比）
                df['horse_weight_diff_from_prev'] = df['horse_weight_value'] - df['prev_horse_weight']

                # 過去3走の平均馬体重
                df['avg_horse_weight_last3'] = (
                    df.groupby('horse_name')['horse_weight_value']
                    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
                )

                # 馬体重のトレンド（増加傾向 or 減少傾向）
                df['horse_weight_trend'] = (
                    df.groupby('horse_name')['horse_weight_value']
                    .transform(lambda x: x.rolling(window=3, min_periods=2).apply(
                        lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
                    ).shift(1))
                )

        # 年齢による特徴
        if 'age' in df.columns:
            df['is_young_horse'] = (df['age'] == 3).astype(int)  # 3歳馬
            df['is_prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int)  # 4-6歳（全盛期）
            df['is_veteran'] = (df['age'] >= 7).astype(int)  # 7歳以上

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        相互作用特徴を追加

        - 騎手×コース の相性
        - 馬×距離×馬場状態 の適性
        - 馬×騎手 のコンビネーション
        """
        # 騎手×競馬場 の組み合わせ勝率
        if 'jockey' in df.columns and 'venue_name' in df.columns and 'finish_position' in df.columns:
            jockey_venue_stats = (
                df[df['finish_position'].notna()]
                .groupby(['jockey', 'venue_name'])
                .agg({
                    'finish_position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
                })
            )
            jockey_venue_stats.columns = ['jockey_venue_races', 'jockey_venue_wins', 'jockey_venue_top3']
            jockey_venue_stats['jockey_venue_win_rate'] = (
                jockey_venue_stats['jockey_venue_wins'] / jockey_venue_stats['jockey_venue_races']
            )
            jockey_venue_stats['jockey_venue_top3_rate'] = (
                jockey_venue_stats['jockey_venue_top3'] / jockey_venue_stats['jockey_venue_races']
            )

            df = df.merge(
                jockey_venue_stats[['jockey_venue_win_rate', 'jockey_venue_top3_rate']],
                left_on=['jockey', 'venue_name'],
                right_index=True,
                how='left'
            )

        # 馬×距離 の適性
        if 'horse_name' in df.columns and 'distance' in df.columns and 'finish_position' in df.columns:
            # 距離カテゴリを作成（短距離、マイル、中距離、長距離）
            df['distance_category_detail'] = pd.cut(
                df['distance'],
                bins=[0, 1400, 1800, 2200, 10000],
                labels=['sprint', 'mile', 'middle', 'long']
            )

            horse_distance_stats = (
                df[df['finish_position'].notna()]
                .groupby(['horse_name', 'distance_category_detail'])
                .agg({
                    'finish_position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
                })
            )
            horse_distance_stats.columns = ['horse_dist_races', 'horse_dist_wins', 'horse_dist_top3']
            horse_distance_stats['horse_distance_win_rate'] = (
                horse_distance_stats['horse_dist_wins'] / horse_distance_stats['horse_dist_races']
            )
            horse_distance_stats['horse_distance_top3_rate'] = (
                horse_distance_stats['horse_dist_top3'] / horse_distance_stats['horse_dist_races']
            )

            df = df.merge(
                horse_distance_stats[['horse_distance_win_rate', 'horse_distance_top3_rate']],
                left_on=['horse_name', 'distance_category_detail'],
                right_index=True,
                how='left'
            )

        # 馬×馬場状態 の適性
        if 'horse_name' in df.columns and 'track_condition' in df.columns and 'finish_position' in df.columns:
            horse_condition_stats = (
                df[df['finish_position'].notna()]
                .groupby(['horse_name', 'track_condition'])
                .agg({
                    'finish_position': ['count', lambda x: (x == 1).sum()]
                })
            )
            horse_condition_stats.columns = ['horse_cond_races', 'horse_cond_wins']
            horse_condition_stats['horse_condition_win_rate'] = (
                horse_condition_stats['horse_cond_wins'] / horse_condition_stats['horse_cond_races']
            )

            df = df.merge(
                horse_condition_stats[['horse_condition_win_rate']],
                left_on=['horse_name', 'track_condition'],
                right_index=True,
                how='left'
            )

        # 馬×騎手 のコンビネーション
        if 'horse_name' in df.columns and 'jockey' in df.columns and 'finish_position' in df.columns:
            horse_jockey_stats = (
                df[df['finish_position'].notna()]
                .groupby(['horse_name', 'jockey'])
                .agg({
                    'finish_position': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
                })
            )
            horse_jockey_stats.columns = ['horse_jockey_races', 'horse_jockey_wins', 'horse_jockey_top3']
            horse_jockey_stats['horse_jockey_win_rate'] = (
                horse_jockey_stats['horse_jockey_wins'] / horse_jockey_stats['horse_jockey_races']
            )
            horse_jockey_stats['horse_jockey_combination_count'] = horse_jockey_stats['horse_jockey_races']

            df = df.merge(
                horse_jockey_stats[['horse_jockey_win_rate', 'horse_jockey_combination_count']],
                left_on=['horse_name', 'jockey'],
                right_index=True,
                how='left'
            )

        return df

    def _add_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時系列特徴を追加

        - 連勝フラグ
        - 連敗フラグ
        - 直近のパフォーマンス（過去3走の平均オッズ）
        """
        if 'horse_name' in df.columns and 'race_date' in df.columns and 'finish_position' in df.columns:
            df = df.sort_values(['horse_name', 'race_date'])

            # 連勝数（1着の連続回数）
            df['is_win'] = (df['finish_position'] == 1).astype(int)
            df['consecutive_wins'] = (
                df.groupby('horse_name')['is_win']
                .apply(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
                .reset_index(level=0, drop=True)
            )

            # 連敗数（1着以外の連続回数）
            df['is_loss'] = (df['finish_position'] > 1).astype(int)
            df['consecutive_losses'] = (
                df.groupby('horse_name')['is_loss']
                .apply(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
                .reset_index(level=0, drop=True)
            )

            # 過去3走の平均オッズ
            if 'odds' in df.columns:
                df['avg_odds_last3'] = (
                    df.groupby('horse_name')['odds']
                    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
                )

            # 過去3走の平均人気
            if 'popularity' in df.columns:
                df['avg_popularity_last3'] = (
                    df.groupby('horse_name')['popularity']
                    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
                )

        # レース内の相対的な特徴量
        if 'race_id' in df.columns:
            # レース内での人気順位の相対位置（例: 18頭中3番人気 → 3/18 = 0.167）
            df['popularity_rank_ratio'] = df.groupby('race_id')['popularity'].rank(pct=True)

            # レース内での オッズの相対位置
            if 'odds' in df.columns:
                df['odds_rank_ratio'] = df.groupby('race_id')['odds'].rank(pct=True)

            # レース内での馬体重の相対位置
            if 'horse_weight_value' in df.columns:
                df['horse_weight_rank_ratio'] = df.groupby('race_id')['horse_weight_value'].rank(pct=True)

        return df


def main():
    """メイン実行"""
    logging.info("=" * 60)
    logging.info("高度な特徴量エンジニアリング")
    logging.info("=" * 60)
    logging.info("")
    logging.info("追加される特徴量:")
    logging.info("  1. レース展開特徴 (前走成績、着順トレンド、枠番有利不利)")
    logging.info("  2. 馬体状態特徴 (馬体重推移、休養期間、連闘/長期休養)")
    logging.info("  3. 相互作用特徴 (騎手×コース、馬×距離、馬×騎手)")
    logging.info("  4. 時系列特徴 (連勝/連敗、直近オッズ/人気)")
    logging.info("")
    logging.info("=" * 60)
    logging.info("")

    # データ読み込み
    data_path = "data/processed/g1_races_historical_processed.csv"
    if not Path(data_path).exists():
        logging.error(f"データが見つかりません: {data_path}")
        logging.info("まず collect_historical_g1_extended.py を実行してください")
        return

    df = pd.read_csv(data_path, encoding='utf-8-sig')
    logging.info(f"元データ読み込み: {data_path}")
    logging.info(f"レコード数: {len(df)}")
    logging.info(f"カラム数: {len(df.columns)}")
    logging.info("")

    # 特徴量エンジニアリング
    engineer = AdvancedFeatureEngineer()
    df_enhanced = engineer.engineer_features(df)

    # 保存
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'g1_races_enhanced_features_{timestamp}.csv'

    df_enhanced.to_csv(output_file, index=False, encoding='utf-8-sig')

    logging.info(f"\n保存先: {output_file}")
    logging.info(f"出力レコード数: {len(df_enhanced)}")
    logging.info(f"出力カラム数: {len(df_enhanced.columns)}")
    logging.info("")

    # 新規追加された特徴量をリスト表示
    original_columns = set(df.columns)
    new_columns = set(df_enhanced.columns) - original_columns

    logging.info("=" * 60)
    logging.info(f"新規追加特徴量: {len(new_columns)}個")
    logging.info("=" * 60)
    for i, col in enumerate(sorted(new_columns), 1):
        logging.info(f"  {i:2d}. {col}")

    logging.info("")
    logging.info("✅ 特徴量エンジニアリング完了!")
    logging.info("")
    logging.info("次のステップ:")
    logging.info("  python train_deep_learning_model.py --data {output_file}")


if __name__ == "__main__":
    main()

"""
収集データの詳細分析スクリプト
データ品質チェックと特徴量エンジニアリングの準備
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

def load_latest_data():
    """最新の収集データを読み込み"""
    files = glob.glob('data/raw/netkeiba_races_*.csv')
    if not files:
        print("CSVファイルが見つかりません")
        return None

    latest_file = max(files, key=os.path.getctime)
    print(f"読み込み: {latest_file}\n")

    df = pd.read_csv(latest_file, encoding='utf-8-sig')
    return df

def analyze_data_quality(df):
    """データ品質の詳細分析"""
    print("="*60)
    print("データ品質分析")
    print("="*60)

    print(f"\n1. 基本情報:")
    print(f"   総レース数: {df['race_id'].nunique()}")
    print(f"   総データ数: {len(df)}頭")
    print(f"   カラム数: {len(df.columns)}")

    print(f"\n2. 欠損値の詳細:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    for col in df.columns:
        if missing[col] > 0:
            print(f"   {col:20s}: {missing[col]:4d}件 ({missing_pct[col]:5.2f}%)")

    print(f"\n3. データ型:")
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        print(f"   {col:20s}: {str(dtype):10s} (ユニーク値: {unique_count})")

    print(f"\n4. 数値カラムの統計:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"\n   {col}:")
        print(f"      平均: {df[col].mean():.2f}")
        print(f"      中央値: {df[col].median():.2f}")
        print(f"      最小: {df[col].min()}")
        print(f"      最大: {df[col].max()}")

def analyze_features(df):
    """特徴量の詳細分析"""
    print("\n" + "="*60)
    print("特徴量分析")
    print("="*60)

    print("\n1. レース条件の分布:")
    print("\n   距離:")
    distance_dist = df['distance'].value_counts().head(10)
    for dist, count in distance_dist.items():
        print(f"      {dist}m: {count}件")

    print("\n   馬場タイプ:")
    for track_type, count in df['track_type'].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {track_type}: {count}件 ({pct:.1f}%)")

    print("\n   馬場状態:")
    for condition, count in df['track_condition'].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {condition}: {count}件 ({pct:.1f}%)")

    print("\n   天候:")
    for weather, count in df['weather'].value_counts().items():
        pct = count / len(df) * 100
        print(f"      {weather}: {count}件 ({pct:.1f}%)")

    print("\n2. 馬の情報:")
    print(f"   ユニーク馬数: {df['horse_id'].nunique()}")

    # 性齢の分布
    if 'sex_age' in df.columns and not df['sex_age'].isnull().all():
        print("\n   性齢分布:")
        sex_age_dist = df['sex_age'].value_counts().head(10)
        for sex_age, count in sex_age_dist.items():
            print(f"      {sex_age}: {count}件")

    # 斤量の分布
    if 'weight' in df.columns and not df['weight'].isnull().all():
        print("\n   斤量:")
        print(f"      平均: {df['weight'].mean():.1f}kg")
        print(f"      範囲: {df['weight'].min():.0f}kg - {df['weight'].max():.0f}kg")

    print("\n3. 騎手情報:")
    print(f"   ユニーク騎手数: {df['jockey_name'].nunique()}")
    print("\n   トップ騎手:")
    top_jockeys = df['jockey_name'].value_counts().head(10)
    for jockey, count in top_jockeys.items():
        print(f"      {jockey}: {count}レース")

    print("\n4. 着順分布:")
    finish_dist = df['finish_position'].value_counts().head(10)
    for pos, count in finish_dist.items():
        print(f"      {pos}: {count}頭")

def analyze_target_variable(df):
    """目的変数（着順）の分析"""
    print("\n" + "="*60)
    print("目的変数（着順）分析")
    print("="*60)

    # 着順を数値に変換可能かチェック
    valid_positions = df[df['finish_position'].astype(str).str.isdigit()].copy()
    valid_positions['finish_position_int'] = valid_positions['finish_position'].astype(int)

    print(f"\n数値着順データ: {len(valid_positions)}件 ({len(valid_positions)/len(df)*100:.1f}%)")
    print(f"除外データ: {len(df) - len(valid_positions)}件")

    if len(valid_positions) > 0:
        print(f"\n着順統計:")
        print(f"   平均着順: {valid_positions['finish_position_int'].mean():.2f}")
        print(f"   中央値: {valid_positions['finish_position_int'].median():.0f}")
        print(f"   範囲: {valid_positions['finish_position_int'].min()} - {valid_positions['finish_position_int'].max()}")

        # 1-3着の割合
        top3 = (valid_positions['finish_position_int'] <= 3).sum()
        print(f"\n3着以内: {top3}頭 ({top3/len(valid_positions)*100:.1f}%)")

def suggest_next_steps(df):
    """次のステップの提案"""
    print("\n" + "="*60)
    print("次のステップ提案")
    print("="*60)

    print("\n1. データクリーニングが必要な項目:")

    # 欠損値が多いカラム
    missing = df.isnull().sum()
    high_missing = missing[missing > 0]
    if len(high_missing) > 0:
        print("\n   欠損値処理:")
        for col, count in high_missing.items():
            pct = count / len(df) * 100
            print(f"      - {col}: {pct:.1f}%")

    print("\n2. 特徴量エンジニアリング候補:")
    print("   - 距離カテゴリ (短距離/マイル/中距離/長距離)")
    print("   - 馬場コンディション複合 (芝-良、ダート-重など)")
    print("   - 騎手勝率 (過去データから算出)")
    print("   - 馬の成績履歴 (このデータセットからは取得不可)")
    print("   - オッズランキング (人気順位)")

    print("\n3. モデル学習の準備:")
    print("   - 着順を数値に変換")
    print("   - カテゴリカル変数のエンコーディング")
    print("   - レースIDでのグループ化（ランキング学習用）")
    print("   - 時系列分割（テスト用データの確保）")

    print("\n4. 追加データ収集の検討:")
    current_races = df['race_id'].nunique()
    if current_races < 500:
        print(f"   現在のレース数: {current_races}")
        print("   推奨: 500レース以上（より robust なモデルのため）")
        print("   方法: SmartRaceDiscoverer で2024年の他の月も収集")

def main():
    """メイン処理"""
    print("="*60)
    print("収集データ詳細分析")
    print("="*60)
    print()

    # データ読み込み
    df = load_latest_data()
    if df is None:
        return

    # 分析実行
    analyze_data_quality(df)
    analyze_features(df)
    analyze_target_variable(df)
    suggest_next_steps(df)

    print("\n" + "="*60)
    print("分析完了")
    print("="*60)

if __name__ == "__main__":
    main()

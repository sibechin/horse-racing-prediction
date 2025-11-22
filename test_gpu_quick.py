"""GPU加速テスト - 簡易版"""
import torch
import time
import pandas as pd
import numpy as np
from comprehensive_feature_study import prepare_races, train_evaluate

if __name__ == '__main__':
    print("="*70)
    print("GPU加速テスト")
    print("="*70)

    # GPU情報
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\nGPU not available")
        exit(1)

    # データ読み込み
    print("\nデータ読み込み中...")
    df = pd.read_csv('data/processed/keibalab_g1_2000_2024_cleaned.csv', encoding='utf-8-sig')
    df_test = pd.read_csv('data/processed/keibalab_g1_2025_cleaned.csv', encoding='utf-8-sig')

    # テスト: 2つの特徴量で訓練
    features = ['popularity', 'odds_numeric']
    print(f"\nテスト特徴量: {features}")

    train_races = prepare_races(df, features)
    test_races = prepare_races(df_test, features)

    print(f"訓練レース数: {len(train_races)}")
    print(f"テストレース数: {len(test_races)}")

    # GPU訓練時間測定
    print("\nGPU訓練開始...")
    torch.cuda.synchronize()
    start = time.time()

    res = train_evaluate(train_races, test_races, len(features), epochs=10, batch_size=256)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"\n結果:")
    print(f"  訓練時間: {elapsed:.2f}秒")
    print(f"  単勝精度: {res['top1']:.1%}")
    print(f"  三連複精度: {res['top3']:.1%}")

    # メモリ使用量
    print(f"\nGPUメモリ使用量:")
    print(f"  割り当て: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  予約済み: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    print("\nGPUテスト完了！")

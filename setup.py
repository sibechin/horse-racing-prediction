"""
セットアップスクリプト
プロジェクトの初期セットアップを行う
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories():
    """必要なディレクトリを作成"""
    directories = [
        'data/raw',
        'data/processed',
        'data/models',
        'logs',
        'notebooks'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_init_files():
    """__init__.pyファイルを作成"""
    init_dirs = [
        'src',
        'src/data_collection',
        'src/preprocessing',
        'src/features',
        'src/models',
        'src/evaluation'
    ]

    for directory in init_dirs:
        init_file = Path(directory) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            logger.info(f"Created __init__.py in {directory}")


def main():
    """メイン処理"""
    logger.info("="*60)
    logger.info("競馬予測プロジェクトのセットアップ")
    logger.info("="*60)

    # ディレクトリの作成
    logger.info("\n[1] ディレクトリの作成")
    create_directories()

    # __init__.pyファイルの作成
    logger.info("\n[2] __init__.pyファイルの作成")
    create_init_files()

    logger.info("\n="*60)
    logger.info("セットアップが完了しました！")
    logger.info("="*60)
    logger.info("\n次のステップ:")
    logger.info("1. pip install -r requirements.txt")
    logger.info("2. configs/config.yaml を確認・編集")
    logger.info("3. python src/data_collection/netkeiba_scraper.py でデータ収集")
    logger.info("4. python train_model.py でモデル訓練")
    logger.info("\n詳細は README.md を参照してください。")


if __name__ == "__main__":
    main()

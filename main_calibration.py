#!/usr/bin/env python3
"""
環境キャリブレーションツール起動スクリプト

使用例:
    python main_calibration.py
    python main_calibration.py --port 8083 --image-dir demo_images
"""

import argparse
from ui.calibration_ui import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JetRacer 環境キャリブレーション"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8083,
        help="ポート番号 (デフォルト: 8083)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="demo_images",
        help="画像ディレクトリ (デフォルト: demo_images)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("JetRacer 環境キャリブレーションツールを起動しています...")
    print(f"アクセスURL: http://localhost:{args.port}")
    print(f"画像ディレクトリ: {args.image_dir}")
    print("="*60 + "\n")

    main(port=args.port)

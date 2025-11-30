#!/usr/bin/env python3
"""
深度推定テストツール起動スクリプト

使用例:
    python main_depth_test.py
    python main_depth_test.py --port 8085
"""

import argparse
from ui.depth_test_ui import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JetRacer 深度推定テストツール"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8085,
        help="ポート番号 (デフォルト: 8085)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("JetRacer 深度推定テストツールを起動しています...")
    print(f"アクセスURL: http://localhost:{args.port}")
    print("="*60 + "\n")

    main(port=args.port)

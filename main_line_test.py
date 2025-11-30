#!/usr/bin/env python3
"""
ライン検出テストツール起動スクリプト

使用例:
    python main_line_test.py
    python main_line_test.py --port 8086
"""

import argparse
from ui.line_test_ui import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JetRacer ライン検出テストツール"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8086,
        help="ポート番号 (デフォルト: 8086)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("JetRacer ライン検出テストツールを起動しています...")
    print(f"アクセスURL: http://localhost:{args.port}")
    print("="*60 + "\n")

    main(port=args.port)

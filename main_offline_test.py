#!/usr/bin/env python3
"""
オフライン検証ツール起動スクリプト

使用例:
    # デモ画像を使用（セッション未指定）
    python main_offline_test.py

    # 録画セッションを指定
    python main_offline_test.py --session output/recordings/20251130_120000

    # ポート番号を指定
    python main_offline_test.py --port 8085
"""

import argparse
from pathlib import Path
from ui.offline_viewer_ui import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JetRacer オフライン検証ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main_offline_test.py
  python main_offline_test.py --session output/recordings/20251130_120000
  python main_offline_test.py --port 8085
        """
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8084,
        help="ポート番号 (デフォルト: 8084)"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="録画セッションパス (例: output/recordings/20251130_120000)"
    )

    args = parser.parse_args()

    # セッションパスの検証
    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            print(f"警告: セッションパスが存在しません: {args.session}")
            print("UIからセッションを選択してください。")
        else:
            print(f"セッションを読み込みます: {args.session}")

    # UIを起動
    print("\n" + "="*60)
    print("JetRacer オフライン検証ツールを起動しています...")
    print(f"アクセスURL: http://localhost:{args.port}")
    print("="*60 + "\n")

    main(session_path=args.session, port=args.port)

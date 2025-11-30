#!/usr/bin/env python3
"""
Phase 4 統合テスト

セグメンテーションとライン検出の統合判定をテストする。
"""

import cv2
import numpy as np
from pathlib import Path
import yaml

from core.image_loader import ImageLoader
from core.onnx_inference import ONNXSegmenter
from core.line_detection import LineDetector
from core.driving_decision import IntegratedDecisionMaker, ProcessingMode
from core.calibration import EnvironmentManager


def test_segmentation_only():
    """セグメンテーションのみのテスト"""
    print("\n" + "="*60)
    print("Phase 4 統合テスト: セグメンテーション")
    print("="*60 + "\n")

    # 設定を読み込み
    env_manager = EnvironmentManager()
    config = env_manager.load_current()

    # セグメンテーションモデルを読み込み
    seg_config = config.get('processing', {}).get('segmentation', {})
    model_path = seg_config.get('model_path', 'output/models/road_segmentation.onnx')
    input_size = tuple(seg_config.get('input_size', [320, 240]))

    if not Path(model_path).exists():
        print(f"✗ セグメンテーションモデルが見つかりません: {model_path}")
        return False

    print(f"✓ モデルパス: {model_path}")
    print(f"✓ 入力サイズ: {input_size}")

    # Use CPU backend for testing to avoid CUDA issues
    segmenter = ONNXSegmenter(
        model_path=model_path,
        input_size=input_size,
        use_cuda=False
    )
    # Model is loaded in __init__

    # デモセッションを読み込み
    session_path = "output/recordings/demo_session"
    loader = ImageLoader(session_path)

    if not loader.load_session():
        print(f"✗ セッションの読み込みに失敗しました: {session_path}")
        return False

    print(f"✓ セッション読み込み: {len(loader)} フレーム\n")

    # 最初のフレームでテスト
    frame_data = loader.get_frame(0)
    if frame_data is None:
        print("✗ フレームの取得に失敗しました")
        return False

    ground_image = frame_data["ground"]

    # セグメンテーション実行
    print("セグメンテーション実行中...")
    mask, seg_time = segmenter.inference(ground_image)

    # クラスごとの面積を計算
    class_areas = {}
    total_pixels = mask.shape[0] * mask.shape[1]

    for class_id in range(3):
        class_areas[class_id] = np.sum(mask == class_id)

    print(f"✓ 推論時間: {seg_time:.2f}ms")
    print(f"✓ マスクサイズ: {mask.shape}")
    print("\nクラス別面積:")
    for class_id, area in class_areas.items():
        ratio = area / total_pixels * 100
        print(f"  クラス {class_id}: {area} ピクセル ({ratio:.1f}%)")

    # セグメンテーション結果を構築
    segmentation_result = {
        'mask': mask,
        'class_areas': class_areas,
        'inference_time_ms': seg_time
    }

    # 統合判定（ライン検出なしの簡易版）
    decision_maker = IntegratedDecisionMaker(config)

    # ダミーのライン検出結果を作成
    dummy_line_result = {
        'white_mask': np.zeros_like(mask),
        'yellow_mask': np.zeros_like(mask),
        'combined_mask': np.zeros_like(mask),
        'lines': [],
        'target_point': None,
        'steering_offset': 0.0,
        'confidence': 0.0
    }

    print("\n統合判定実行中...")
    decision = decision_maker.decide_line_following(
        segmentation_result,
        dummy_line_result
    )

    print("\n判定結果:")
    print(f"  コマンド: {decision.command.value}")
    print(f"  速度: {decision.speed:.2f}")
    print(f"  ステアリング: {decision.steering:.2f}")
    print(f"  信頼度: {decision.confidence:.2f}")
    print(f"  地面安全: {'✓' if decision.ground_safe else '✗'}")
    print(f"  前方クリア: {'✓' if decision.front_clear else '✗'}")
    print(f"  理由: {decision.reason}")

    print("\n✓ セグメンテーション統合テスト完了！\n")
    return True


def test_line_detection():
    """ライン検出のテスト"""
    print("\n" + "="*60)
    print("Phase 4 統合テスト: ライン検出")
    print("="*60 + "\n")

    # 設定を読み込み
    env_manager = EnvironmentManager()
    config = env_manager.load_current()

    # ライン検出器を初期化
    line_config = config.get('processing', {}).get('line_detection', {})
    line_detector = LineDetector(line_config)

    # デモセッションを読み込み
    session_path = "output/recordings/demo_session"
    loader = ImageLoader(session_path)

    if not loader.load_session():
        print(f"✗ セッションの読み込みに失敗しました: {session_path}")
        return False

    print(f"✓ セッション読み込み: {len(loader)} フレーム\n")

    # 最初のフレームでテスト
    frame_data = loader.get_frame(0)
    if frame_data is None:
        print("✗ フレームの取得に失敗しました")
        return False

    front_image = frame_data["front"]

    # ライン検出実行
    print("ライン検出実行中...")
    line_result = line_detector.detect(front_image)

    print(f"✓ 検出ライン数: {len(line_result['lines'])}")
    print(f"✓ 目標点: {line_result['target_point']}")
    print(f"✓ ステアリングオフセット: {line_result['steering_offset']:.2f}")
    print(f"✓ 信頼度: {line_result['confidence']:.2f}")

    print("\n✓ ライン検出テスト完了！\n")
    return True


if __name__ == "__main__":
    print("\nPhase 4 統合テストを開始します...\n")

    success = True

    # Test 1: セグメンテーション統合
    if not test_segmentation_only():
        success = False

    # Test 2: ライン検出
    if not test_line_detection():
        success = False

    if success:
        print("\n" + "="*60)
        print("✓ すべてのテストが成功しました！")
        print("="*60 + "\n")
        print("次のステップ:")
        print("1. オフラインビューアを起動:")
        print("   python main_offline_test.py --session output/recordings/demo_session")
        print("\n2. ブラウザで http://localhost:8084 にアクセス")
        print("\n3. '処理実行' ボタンを押して統合判定を確認")
        print("\n")
    else:
        print("\n✗ テストに失敗しました\n")

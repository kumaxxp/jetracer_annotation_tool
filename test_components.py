#!/usr/bin/env python3
"""
個別コンポーネントの動作確認テスト
"""

import sys
from pathlib import Path
import numpy as np
import cv2

print("\n" + "="*60)
print("コンポーネント動作確認テスト")
print("="*60 + "\n")

# Test 1: Image Loader
print("Test 1: Image Loader")
print("-" * 40)
try:
    from core.image_loader import ImageLoader
    loader = ImageLoader("output/recordings/demo_session")
    if loader.load_session():
        frame = loader.get_frame(0)
        if frame and frame['ground'] is not None and frame['front'] is not None:
            print(f"✓ Image Loader: OK")
            print(f"  - Ground image shape: {frame['ground'].shape}")
            print(f"  - Front image shape: {frame['front'].shape}")
        else:
            print("✗ Image Loader: フレーム取得失敗")
            sys.exit(1)
    else:
        print("✗ Image Loader: セッション読み込み失敗")
        sys.exit(1)
except Exception as e:
    print(f"✗ Image Loader: {e}")
    sys.exit(1)

# Test 2: Segmentation
print("\nTest 2: Segmentation (CPU backend)")
print("-" * 40)
try:
    from core.onnx_inference import ONNXSegmenter
    segmenter = ONNXSegmenter(
        model_path="output/models/road_segmentation.onnx",
        input_size=(320, 240),
        use_cuda=False
    )
    mask, time_ms = segmenter.inference(frame['ground'])
    print(f"✓ Segmentation: OK")
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Inference time: {time_ms:.2f}ms")
    print(f"  - Unique classes: {np.unique(mask)}")
except Exception as e:
    print(f"✗ Segmentation: {e}")
    sys.exit(1)

# Test 3: Line Detection
print("\nTest 3: Line Detection")
print("-" * 40)
try:
    from core.line_detection import LineDetector
    from core.calibration import EnvironmentManager

    env_manager = EnvironmentManager()
    config = env_manager.load_current()
    line_config = config.get('processing', {}).get('line_detection', {})

    detector = LineDetector(line_config)
    result = detector.detect(frame['front'])

    print(f"✓ Line Detection: OK")
    print(f"  - Lines detected: {len(result['lines'])}")
    print(f"  - Target point: {result['target_point']}")
    print(f"  - Steering offset: {result['steering_offset']:.3f}")
    print(f"  - Confidence: {result['confidence']:.3f}")
except Exception as e:
    print(f"✗ Line Detection: {e}")
    sys.exit(1)

# Test 4: Driving Decision
print("\nTest 4: Driving Decision")
print("-" * 40)
try:
    from core.driving_decision import IntegratedDecisionMaker

    decision_maker = IntegratedDecisionMaker(config)

    # セグメンテーション結果を構築
    class_areas = {i: np.sum(mask == i) for i in range(3)}
    seg_result = {
        'mask': mask,
        'class_areas': class_areas,
        'inference_time_ms': time_ms
    }

    # Mode B: ライントレースでテスト
    decision = decision_maker.decide_line_following(seg_result, result)

    print(f"✓ Driving Decision: OK")
    print(f"  - Command: {decision.command.value}")
    print(f"  - Speed: {decision.speed:.2f}")
    print(f"  - Steering: {decision.steering:.2f}")
    print(f"  - Confidence: {decision.confidence:.2f}")
    print(f"  - Ground safe: {decision.ground_safe}")
    print(f"  - Reason: {decision.reason}")
except Exception as e:
    print(f"✗ Driving Decision: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Configuration Management
print("\nTest 5: Configuration Management")
print("-" * 40)
try:
    from core.calibration import EnvironmentManager

    env_manager = EnvironmentManager()
    config = env_manager.load_current()

    # 設定の確認
    has_processing = 'processing' in config
    has_segmentation = 'segmentation' in config.get('processing', {})
    has_line_detection = 'line_detection' in config.get('processing', {})

    print(f"✓ Configuration Management: OK")
    print(f"  - Processing config: {has_processing}")
    print(f"  - Segmentation config: {has_segmentation}")
    print(f"  - Line detection config: {has_line_detection}")

    # 環境一覧
    envs = env_manager.list_environments()
    print(f"  - Available environments: {len(envs)}")
    if envs:
        print(f"    {', '.join(envs)}")

except Exception as e:
    print(f"✗ Configuration Management: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ すべてのコンポーネントテストが成功しました！")
print("="*60 + "\n")

print("動作確認済みコンポーネント:")
print("  1. ✓ Image Loader (画像読み込み)")
print("  2. ✓ Segmentation (セグメンテーション)")
print("  3. ✓ Line Detection (ライン検出)")
print("  4. ✓ Driving Decision (統合判定)")
print("  5. ✓ Configuration Management (設定管理)")
print("\nPhase 1-4 のコア機能はすべて正常に動作しています。\n")

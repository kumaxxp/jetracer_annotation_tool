#!/usr/bin/env python3
"""
テストデータ用の動作確認GUI（OpenCVベース）

機能:
- テストデータフォルダ（demo_session）の画像を読み込み
- セグメンテーション、ライン検出、統合判定を実行
- 結果を画面に表示
- キーボードで操作
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import yaml

from core.image_loader import ImageLoader
from core.onnx_inference import ONNXSegmenter
from core.line_detection import LineDetector
from core.driving_decision import IntegratedDecisionMaker, ProcessingMode


class TestGUI:
    """テストデータ用の動作確認GUI"""

    def __init__(self, session_path: str = "output/recordings/demo_session"):
        self.session_path = Path(session_path)
        self.current_frame_index = 0
        self.mode = ProcessingMode.LINE_FOLLOWING
        self.auto_process = False  # 自動処理モード

        # 設定を読み込み
        with open("config/default.yaml", 'r') as f:
            self.config = yaml.safe_load(f)

        # 画像ローダーを初期化
        self.loader = ImageLoader(str(self.session_path))
        if not self.loader.load_session():
            raise RuntimeError(f"セッションの読み込みに失敗: {session_path}")

        print(f"✓ セッション読み込み完了: {len(self.loader)} フレーム")

        # 処理モジュールを初期化
        self._init_processing_modules()

        # 処理結果キャッシュ
        self.segmentation_result: Optional[Dict] = None
        self.line_result: Optional[Dict] = None
        self.decision: Optional[Dict] = None

        # ウィンドウ名
        self.window_ground = "Ground Camera (Segmentation)"
        self.window_front = "Front Camera (Line Detection)"
        self.window_info = "Decision & Info"

    def _init_processing_modules(self):
        """処理モジュールを初期化"""
        # セグメンテーション
        seg_config = self.config.get('processing', {}).get('segmentation', {})
        model_path = seg_config.get('model_path', 'models/segmentation_model.onnx')
        input_size = tuple(seg_config.get('input_size', [224, 224]))
        use_cuda = seg_config.get('use_cuda', False)

        if Path(model_path).exists():
            try:
                self.segmenter = ONNXSegmenter(
                    model_path=model_path,
                    input_size=input_size,
                    use_cuda=use_cuda
                )
                print(f"✓ セグメンテーションモデル読み込み完了: {model_path}")
            except Exception as e:
                print(f"✗ セグメンテーションモデル読み込みエラー: {e}")
                self.segmenter = None
        else:
            print(f"✗ セグメンテーションモデルが見つかりません: {model_path}")
            self.segmenter = None

        # ライン検出
        line_config = self.config.get('processing', {}).get('line_detection', {})
        self.line_detector = LineDetector(line_config)
        print("✓ ライン検出器初期化完了")

        # 統合判定
        self.decision_maker = IntegratedDecisionMaker(self.config)
        print("✓ 統合判定器初期化完了")

    def process_frame(self):
        """現在のフレームを処理"""
        # フレームを取得
        frame_data = self.loader.get_frame(self.current_frame_index)
        if frame_data is None:
            print("✗ フレームの取得に失敗")
            return None, None

        ground_image = frame_data["ground"]
        front_image = frame_data["front"]

        if ground_image is None or front_image is None:
            print("✗ 画像が見つかりません")
            return None, None

        # 1. セグメンテーション処理（足元カメラ）
        if self.segmenter is not None:
            mask, seg_time = self.segmenter.inference(ground_image)

            # クラスごとの面積を計算
            total_pixels = mask.size
            class_areas = {}
            class_percentages = {}
            for class_id in range(3):  # 0, 1, 2
                area = np.sum(mask == class_id)
                class_areas[class_id] = area
                class_percentages[class_id] = (area / total_pixels) * 100

            self.segmentation_result = {
                'mask': mask,
                'class_areas': class_areas,
                'class_percentages': class_percentages,
                'inference_time_ms': seg_time
            }

            # セグメンテーション結果をオーバーレイ表示
            ground_display = self._create_segmentation_overlay(ground_image, mask)
        else:
            ground_display = ground_image.copy()
            self.segmentation_result = None

        # 2. ライン検出（正面カメラ）
        if self.line_detector is not None:
            self.line_result = self.line_detector.detect(front_image)

            # ライン検出結果を可視化
            front_display = self._visualize_line_detection(front_image, self.line_result)
        else:
            front_display = front_image.copy()
            self.line_result = None

        # 3. 統合判定
        if self.decision_maker is not None and self.segmentation_result is not None:
            if self.mode == ProcessingMode.LINE_FOLLOWING and self.line_result is not None:
                self.decision = self.decision_maker.decide_line_following(
                    self.segmentation_result,
                    self.line_result
                )

        return ground_display, front_display

    def _create_segmentation_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """セグメンテーションマスクをオーバーレイ表示"""
        overlay = image.copy()

        # カラーマップ（クラスごとの色）
        colors = {
            0: [0, 0, 255],    # クラス0: 赤（車体）
            1: [255, 0, 0],    # クラス1: 青（障害物）
            2: [0, 255, 0]     # クラス2: 緑（通行可能）
        }

        # マスクをカラー画像に変換
        h, w = image.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in colors.items():
            colored_mask[mask_resized == class_id] = color

        # アルファブレンド
        overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)

        return overlay

    def _visualize_line_detection(self, image: np.ndarray, line_result: Dict) -> np.ndarray:
        """ライン検出結果を可視化"""
        vis = image.copy()

        # ホワイトマスク（緑で表示）
        white_mask = line_result.get('white_mask')
        if white_mask is not None and white_mask.size > 0:
            h, w = vis.shape[:2]
            mask_resized = cv2.resize(white_mask, (w, h))
            # マスク領域を緑色でブレンド
            green_overlay = np.zeros_like(vis)
            green_overlay[mask_resized > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, green_overlay, 0.3, 0)

        # イエローマスク（黄色で表示）
        yellow_mask = line_result.get('yellow_mask')
        if yellow_mask is not None and yellow_mask.size > 0:
            h, w = vis.shape[:2]
            mask_resized = cv2.resize(yellow_mask, (w, h))
            # マスク領域を黄色でブレンド
            yellow_overlay = np.zeros_like(vis)
            yellow_overlay[mask_resized > 0] = [0, 255, 255]
            vis = cv2.addWeighted(vis, 0.7, yellow_overlay, 0.3, 0)

        # 検出ラインを描画
        lines = line_result.get('lines', [])
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 目標点を描画
        target_point = line_result.get('target_point')
        if target_point:
            cx, cy = target_point
            cv2.circle(vis, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(vis, (cx, cy), 15, (255, 0, 0), 2)

            # 画像中央との線
            center_x = vis.shape[1] // 2
            cv2.line(vis, (center_x, cy), (cx, cy), (255, 0, 0), 2)

        # ステアリング情報を表示
        steering_offset = line_result.get('steering_offset', 0.0)
        confidence = line_result.get('confidence', 0.0)

        cv2.putText(vis, f"Offset: {steering_offset:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Confidence: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def _create_info_panel(self) -> np.ndarray:
        """情報パネルを作成"""
        # 640x480の黒い画像
        panel = np.zeros((480, 640, 3), dtype=np.uint8)

        y_offset = 30
        line_height = 30

        # タイトル
        cv2.putText(panel, "JetRacer Vision System - Test GUI", (10, int(y_offset)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height * 2

        # フレーム情報
        cv2.putText(panel, f"Frame: {self.current_frame_index + 1} / {len(self.loader)}",
                   (10, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height

        # モード情報
        mode_text = "Line Following" if self.mode == ProcessingMode.LINE_FOLLOWING else "Obstacle Avoidance"
        cv2.putText(panel, f"Mode: {mode_text}", (10, int(y_offset)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height * 1.5

        # セグメンテーション結果
        if self.segmentation_result:
            cv2.putText(panel, "=== Segmentation ===", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += line_height

            percentages = self.segmentation_result.get('class_percentages', {})
            cv2.putText(panel, f"Vehicle:     {percentages.get(0, 0):.1f}%", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += line_height

            cv2.putText(panel, f"Obstacle:    {percentages.get(1, 0):.1f}%", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            y_offset += line_height

            cv2.putText(panel, f"Passable:    {percentages.get(2, 0):.1f}%", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height

            seg_time = self.segmentation_result.get('inference_time_ms', 0)
            cv2.putText(panel, f"Inference:   {seg_time:.1f}ms", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height * 1.5

        # ライン検出結果
        if self.line_result:
            cv2.putText(panel, "=== Line Detection ===", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += line_height

            steering = self.line_result.get('steering_offset', 0.0)
            confidence = self.line_result.get('confidence', 0.0)

            cv2.putText(panel, f"Steering:    {steering:.2f}", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height

            cv2.putText(panel, f"Confidence:  {confidence:.2f}", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height * 1.5

        # 統合判定結果
        if self.decision:
            cv2.putText(panel, "=== Decision ===", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += line_height

            command = self.decision.command.value.upper()
            command_color = self._get_command_color(self.decision.command.value)
            cv2.putText(panel, f"Command:     {command}", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, command_color, 2)
            y_offset += line_height

            cv2.putText(panel, f"Speed:       {self.decision.speed:.2f}", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height

            cv2.putText(panel, f"Steering:    {self.decision.steering:.2f}", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height

            cv2.putText(panel, f"Confidence:  {self.decision.confidence:.2f}", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height

            ground_safe = "YES" if self.decision.ground_safe else "NO"
            ground_color = (0, 255, 0) if self.decision.ground_safe else (0, 0, 255)
            cv2.putText(panel, f"Ground Safe: {ground_safe}", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ground_color, 1)
            y_offset += line_height * 1.5

            # 理由を表示（長いので折り返し）
            cv2.putText(panel, "Reason:", (10, int(y_offset)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height

            reason = self.decision.reason
            # 最大40文字ずつで折り返し
            max_chars = 40
            reason_lines = [reason[i:i+max_chars] for i in range(0, len(reason), max_chars)]
            for line in reason_lines[:3]:  # 最大3行
                cv2.putText(panel, f"  {line}", (10, int(y_offset)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                y_offset += line_height - 5

        # キー操作ヘルプ
        y_offset = 450
        cv2.putText(panel, "Keys: [n]Next [p]Prev [SPACE]Process [q]Quit",
                   (10, int(y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        return panel

    def _get_command_color(self, command: str):
        """コマンドに応じた色を返す"""
        colors = {
            'go': (0, 255, 0),           # 緑
            'slow': (0, 255, 255),       # 黄
            'stop': (0, 165, 255),       # オレンジ
            'turn_left': (255, 255, 0),  # シアン
            'turn_right': (255, 255, 0), # シアン
            'emergency_stop': (0, 0, 255) # 赤
        }
        return colors.get(command, (255, 255, 255))

    def run(self):
        """GUIメインループ"""
        # ウィンドウを作成
        cv2.namedWindow(self.window_ground, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_front, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_info, cv2.WINDOW_NORMAL)

        # ウィンドウサイズを設定
        cv2.resizeWindow(self.window_ground, 640, 480)
        cv2.resizeWindow(self.window_front, 640, 480)
        cv2.resizeWindow(self.window_info, 640, 480)

        print("\n" + "="*60)
        print("テストGUI起動")
        print("="*60)
        print("操作方法:")
        print("  [n]      : 次のフレーム")
        print("  [p]      : 前のフレーム")
        print("  [SPACE]  : 現在のフレームを処理")
        print("  [a]      : 自動処理モード ON/OFF")
        print("  [q]      : 終了")
        print("="*60 + "\n")

        # 最初のフレームを表示
        frame_data = self.loader.get_frame(self.current_frame_index)
        if frame_data:
            cv2.imshow(self.window_ground, frame_data["ground"])
            cv2.imshow(self.window_front, frame_data["front"])
            cv2.imshow(self.window_info, self._create_info_panel())

        while True:
            key = cv2.waitKey(100) & 0xFF

            if key == ord('q'):
                print("終了します")
                break

            elif key == ord('n'):
                # 次のフレーム
                if self.current_frame_index < len(self.loader) - 1:
                    self.current_frame_index += 1
                    print(f"フレーム {self.current_frame_index + 1}/{len(self.loader)}")

                    # 処理結果をクリア
                    self.segmentation_result = None
                    self.line_result = None
                    self.decision = None

                    # フレームを表示
                    frame_data = self.loader.get_frame(self.current_frame_index)
                    if frame_data:
                        cv2.imshow(self.window_ground, frame_data["ground"])
                        cv2.imshow(self.window_front, frame_data["front"])
                        cv2.imshow(self.window_info, self._create_info_panel())

                    if self.auto_process:
                        ground_display, front_display = self.process_frame()
                        if ground_display is not None:
                            cv2.imshow(self.window_ground, ground_display)
                            cv2.imshow(self.window_front, front_display)
                            cv2.imshow(self.window_info, self._create_info_panel())

            elif key == ord('p'):
                # 前のフレーム
                if self.current_frame_index > 0:
                    self.current_frame_index -= 1
                    print(f"フレーム {self.current_frame_index + 1}/{len(self.loader)}")

                    # 処理結果をクリア
                    self.segmentation_result = None
                    self.line_result = None
                    self.decision = None

                    # フレームを表示
                    frame_data = self.loader.get_frame(self.current_frame_index)
                    if frame_data:
                        cv2.imshow(self.window_ground, frame_data["ground"])
                        cv2.imshow(self.window_front, frame_data["front"])
                        cv2.imshow(self.window_info, self._create_info_panel())

                    if self.auto_process:
                        ground_display, front_display = self.process_frame()
                        if ground_display is not None:
                            cv2.imshow(self.window_ground, ground_display)
                            cv2.imshow(self.window_front, front_display)
                            cv2.imshow(self.window_info, self._create_info_panel())

            elif key == ord(' '):
                # 処理実行
                print("処理を実行中...")
                ground_display, front_display = self.process_frame()
                if ground_display is not None:
                    cv2.imshow(self.window_ground, ground_display)
                    cv2.imshow(self.window_front, front_display)
                    cv2.imshow(self.window_info, self._create_info_panel())
                    print("✓ 処理完了")

            elif key == ord('a'):
                # 自動処理モードの切り替え
                self.auto_process = not self.auto_process
                mode = "ON" if self.auto_process else "OFF"
                print(f"自動処理モード: {mode}")

                if self.auto_process:
                    ground_display, front_display = self.process_frame()
                    if ground_display is not None:
                        cv2.imshow(self.window_ground, ground_display)
                        cv2.imshow(self.window_front, front_display)
                        cv2.imshow(self.window_info, self._create_info_panel())

        cv2.destroyAllWindows()


    def run_headless(self, output_dir: str = "test_results"):
        """ヘッドレスモード（画像保存）"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("\n" + "="*60)
        print("ヘッドレスモード - 全フレームを処理して保存")
        print("="*60)

        total_frames = len(self.loader)

        for frame_idx in range(total_frames):
            self.current_frame_index = frame_idx

            print(f"\nフレーム {frame_idx + 1}/{total_frames} を処理中...")

            # 処理実行
            ground_display, front_display = self.process_frame()

            if ground_display is None:
                print(f"  ✗ フレーム {frame_idx} の処理に失敗")
                continue

            # 情報パネルを作成
            info_panel = self._create_info_panel()

            # 画像を保存
            cv2.imwrite(str(output_path / f"frame_{frame_idx:04d}_ground.jpg"), ground_display)
            cv2.imwrite(str(output_path / f"frame_{frame_idx:04d}_front.jpg"), front_display)
            cv2.imwrite(str(output_path / f"frame_{frame_idx:04d}_info.jpg"), info_panel)

            # 結果を表示
            if self.decision:
                print(f"  Command: {self.decision.command.value}")
                print(f"  Speed: {self.decision.speed:.2f}")
                print(f"  Steering: {self.decision.steering:.2f}")
                print(f"  Ground Safe: {self.decision.ground_safe}")
                if self.segmentation_result:
                    percentages = self.segmentation_result.get('class_percentages', {})
                    print(f"  Passable: {percentages.get(2, 0):.1f}%")

        print("\n" + "="*60)
        print(f"✓ 処理完了: {total_frames} フレーム")
        print(f"✓ 保存先: {output_path.absolute()}")
        print("="*60 + "\n")


def main():
    """メインエントリポイント"""
    import argparse

    parser = argparse.ArgumentParser(description='JetRacer Vision System - Test GUI')
    parser.add_argument('--session', type=str, default='output/recordings/demo_session',
                       help='Path to the test session directory')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (save images instead of displaying)')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for headless mode')

    args = parser.parse_args()

    try:
        gui = TestGUI(session_path=args.session)

        if args.headless:
            gui.run_headless(output_dir=args.output)
        else:
            gui.run()

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

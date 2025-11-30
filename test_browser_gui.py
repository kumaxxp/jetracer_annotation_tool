#!/usr/bin/env python3
"""
ブラウザベースのテストGUI（NiceGUI）

機能:
- テストデータフォルダの画像を読み込み
- セグメンテーション、ライン検出、統合判定を実行
- 結果をブラウザに表示
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import yaml
import base64
import io
from PIL import Image

from nicegui import ui

from core.image_loader import ImageLoader
from core.onnx_inference import ONNXSegmenter
from core.line_detection import LineDetector
from core.driving_decision import IntegratedDecisionMaker, ProcessingMode


class BrowserTestGUI:
    """ブラウザベースのテストGUI"""

    def __init__(self, session_path: str = "output/recordings/demo_session"):
        self.session_path = Path(session_path)
        self.current_frame_index = 0
        self.mode = ProcessingMode.LINE_FOLLOWING

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

        # UI要素（後で初期化）
        self.ground_display = None
        self.front_display = None
        self.info_text = None
        self.frame_slider = None
        self.frame_label = None

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
                print(f"✓ セグメンテーションモデル読み込み完了")
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

    def _cv2_to_base64(self, image: np.ndarray) -> str:
        """OpenCV画像をBase64エンコード"""
        # BGR → RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # PIL Imageに変換
        pil_image = Image.fromarray(rgb_image)

        # Base64エンコード
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    def _create_segmentation_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """セグメンテーションマスクをオーバーレイ表示"""
        overlay = image.copy()

        # カラーマップ
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
            green_overlay = np.zeros_like(vis)
            green_overlay[mask_resized > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, green_overlay, 0.3, 0)

        # イエローマスク（黄色で表示）
        yellow_mask = line_result.get('yellow_mask')
        if yellow_mask is not None and yellow_mask.size > 0:
            h, w = vis.shape[:2]
            mask_resized = cv2.resize(yellow_mask, (w, h))
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

    def process_frame(self):
        """現在のフレームを処理"""
        # フレームを取得
        frame_data = self.loader.get_frame(self.current_frame_index)
        if frame_data is None:
            ui.notify("フレームの取得に失敗", type='negative')
            return

        ground_image = frame_data["ground"]
        front_image = frame_data["front"]

        if ground_image is None or front_image is None:
            ui.notify("画像が見つかりません", type='negative')
            return

        # 1. セグメンテーション処理
        if self.segmenter is not None:
            mask, seg_time = self.segmenter.inference(ground_image)

            # クラスごとの面積を計算
            total_pixels = mask.size
            class_areas = {}
            class_percentages = {}
            for class_id in range(3):
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
            ground_img = self._cv2_to_base64(ground_display)
            self.ground_display.source = f"data:image/jpeg;base64,{ground_img}"
        else:
            ground_img = self._cv2_to_base64(ground_image)
            self.ground_display.source = f"data:image/jpeg;base64,{ground_img}"
            self.segmentation_result = None

        # 2. ライン検出
        if self.line_detector is not None:
            self.line_result = self.line_detector.detect(front_image)

            # ライン検出結果を可視化
            front_display = self._visualize_line_detection(front_image, self.line_result)
            front_img = self._cv2_to_base64(front_display)
            self.front_display.source = f"data:image/jpeg;base64,{front_img}"
        else:
            front_img = self._cv2_to_base64(front_image)
            self.front_display.source = f"data:image/jpeg;base64,{front_img}"
            self.line_result = None

        # 3. 統合判定
        if self.decision_maker is not None and self.segmentation_result is not None:
            if self.mode == ProcessingMode.LINE_FOLLOWING and self.line_result is not None:
                self.decision = self.decision_maker.decide_line_following(
                    self.segmentation_result,
                    self.line_result
                )

        # 4. 判定結果を表示
        self._update_decision_display()

        ui.notify("✓ 処理完了", type='positive')

    def _update_decision_display(self):
        """判定結果を更新"""
        if self.decision is None:
            self.info_text.set_content("処理を実行してください")
            return

        percentages = self.segmentation_result.get('class_percentages', {})
        seg_time = self.segmentation_result.get('inference_time_ms', 0)
        steering = self.line_result.get('steering_offset', 0.0)
        confidence = self.line_result.get('confidence', 0.0)

        command_colors = {
            'go': 'positive',
            'slow': 'warning',
            'stop': 'warning',
            'turn_left': 'info',
            'turn_right': 'info',
            'emergency_stop': 'negative'
        }

        command_value = self.decision.command.value
        command_color = command_colors.get(command_value, 'grey')

        info_html = f"""
        <div style="padding: 10px;">
            <h3>セグメンテーション結果</h3>
            <ul>
                <li><span style="color: red;">車体:</span> {percentages.get(0, 0):.1f}%</li>
                <li><span style="color: blue;">障害物:</span> {percentages.get(1, 0):.1f}%</li>
                <li><span style="color: green;">通行可能:</span> {percentages.get(2, 0):.1f}%</li>
                <li>推論時間: {seg_time:.1f}ms</li>
            </ul>

            <h3>ライン検出結果</h3>
            <ul>
                <li>ステアリング: {steering:.2f}</li>
                <li>信頼度: {confidence:.2f}</li>
            </ul>

            <h3>統合判定結果</h3>
            <div style="background-color: {'#ef5350' if command_value == 'emergency_stop' else '#66bb6a' if command_value == 'go' else '#ffa726'};
                        padding: 10px; margin: 10px 0; border-radius: 5px; color: white; font-weight: bold;">
                コマンド: {command_value.upper()}
            </div>
            <ul>
                <li>速度: {self.decision.speed:.2f}</li>
                <li>ステアリング: {self.decision.steering:.2f}</li>
                <li>信頼度: {self.decision.confidence:.2f}</li>
                <li>地面安全: {'✓ YES' if self.decision.ground_safe else '✗ NO'}</li>
            </ul>
            <p><strong>理由:</strong> {self.decision.reason}</p>
        </div>
        """

        self.info_text.set_content(info_html)

    def on_slider_change(self, e):
        """スライダー変更時"""
        self.current_frame_index = int(e.value)
        self.frame_label.set_text(f"フレーム: {self.current_frame_index + 1} / {len(self.loader)}")

        # 処理結果をクリア
        self.segmentation_result = None
        self.line_result = None
        self.decision = None

        # 元画像を表示
        frame_data = self.loader.get_frame(self.current_frame_index)
        if frame_data:
            ground_img = self._cv2_to_base64(frame_data["ground"])
            front_img = self._cv2_to_base64(frame_data["front"])
            self.ground_display.source = f"data:image/jpeg;base64,{ground_img}"
            self.front_display.source = f"data:image/jpeg;base64,{front_img}"
            self.info_text.set_content("処理を実行してください")

    def prev_frame(self):
        """前のフレーム"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.frame_slider.value = self.current_frame_index
            self.on_slider_change(type('obj', (object,), {'value': self.current_frame_index})())

    def next_frame(self):
        """次のフレーム"""
        if self.current_frame_index < len(self.loader) - 1:
            self.current_frame_index += 1
            self.frame_slider.value = self.current_frame_index
            self.on_slider_change(type('obj', (object,), {'value': self.current_frame_index})())

    def create_ui(self):
        """UIを構築"""
        with ui.header().classes('items-center justify-between bg-blue-600'):
            ui.label("JetRacer Vision System - Browser Test GUI").classes("text-h4 text-white")

        with ui.row().classes("w-full gap-4 p-4"):
            # 左カラム: 画像表示
            with ui.column().classes("flex-1"):
                # 足元カメラ
                with ui.card().classes("w-full"):
                    ui.label("足元カメラ (セグメンテーション)").classes("text-h6 mb-2")
                    self.ground_display = ui.image().classes('w-full')

                # 正面カメラ
                with ui.card().classes("w-full"):
                    ui.label("正面カメラ (ライン検出)").classes("text-h6 mb-2")
                    self.front_display = ui.image().classes('w-full')

            # 右カラム: 操作パネル
            with ui.column().classes("w-96"):
                # フレーム操作
                with ui.card().classes("w-full p-4"):
                    ui.label("フレーム操作").classes("text-h6 mb-2")

                    self.frame_label = ui.label(f"フレーム: 1 / {len(self.loader)}").classes("mb-2")

                    with ui.row().classes("w-full gap-2 mb-2"):
                        ui.button("← 前", on_click=self.prev_frame, icon='navigate_before').classes('flex-1')
                        ui.button("次 →", on_click=self.next_frame, icon='navigate_next').classes('flex-1')

                    self.frame_slider = ui.slider(
                        min=0,
                        max=len(self.loader) - 1,
                        value=0,
                        on_change=self.on_slider_change
                    ).classes('w-full')

                # 処理実行
                with ui.card().classes("w-full p-4"):
                    ui.button(
                        "処理実行",
                        on_click=self.process_frame,
                        icon='play_arrow',
                        color='primary'
                    ).classes('w-full text-lg')

                # 判定結果
                with ui.card().classes("w-full p-4"):
                    ui.label("処理結果").classes("text-h6 mb-2")
                    self.info_text = ui.html("処理を実行してください", sanitize=False)

        # 初期画像を表示
        frame_data = self.loader.get_frame(0)
        if frame_data:
            ground_img = self._cv2_to_base64(frame_data["ground"])
            front_img = self._cv2_to_base64(frame_data["front"])
            self.ground_display.source = f"data:image/jpeg;base64,{ground_img}"
            self.front_display.source = f"data:image/jpeg;base64,{front_img}"


if __name__ in {"__main__", "__mp_main__"}:
    import argparse

    parser = argparse.ArgumentParser(description='JetRacer Vision System - Browser Test GUI')
    parser.add_argument('--session', type=str, default='output/recordings/demo_session',
                       help='Path to the test session directory')
    parser.add_argument('--port', type=int, default=8090,
                       help='Port number for the web server')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ブラウザベースのテストGUIを起動します")
    print("="*60 + "\n")

    try:
        gui = BrowserTestGUI(session_path=args.session)
        gui.create_ui()

        print(f"✓ UIセットアップ完了")
        print(f"✓ アクセスURL: http://localhost:{args.port}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        ui.label(f"エラー: {str(e)}").classes("text-negative")

    ui.run(port=args.port, title="JetRacer Test GUI", show=False)

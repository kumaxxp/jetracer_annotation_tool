"""
オフライン検証ビューアUI

録画済みJPEGファイルに対して処理を実行し、結果を確認する。
"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np
import yaml
import base64
import io
from PIL import Image

from core.image_loader import ImageLoader
from core.onnx_inference import ONNXSegmenter
from core.depth_estimation import DepthEstimator, ObstacleAnalyzer
from core.line_detection import LineDetector
from core.driving_decision import IntegratedDecisionMaker, ProcessingMode
from core.calibration import EnvironmentManager


class OfflineViewerUI:
    """
    オフライン検証ビューア

    機能:
        1. 録画セッション選択
        2. フレーム単位での処理・表示
        3. 足元/正面カメラの同時表示
        4. 処理結果（セグメンテーション、深度、ライン）のオーバーレイ (Phase 2-4で実装)
        5. モード切り替え（障害物回避/ライントレース） (Phase 4で実装)
        6. パラメータ調整 (Phase 2-4で実装)
    """

    def __init__(self, session_path: Optional[str] = None, config_path: str = "config/default.yaml"):
        self.session_path: Optional[Path] = Path(session_path) if session_path else None
        self.loader: Optional[ImageLoader] = None
        self.current_frame_index: int = 0
        self.mode: ProcessingMode = ProcessingMode.OBSTACLE_AVOIDANCE

        # 設定を読み込み
        self.env_manager = EnvironmentManager()
        self.config = self.env_manager.load_current()

        # 処理モジュールを初期化
        self._init_processing_modules()

        # 処理結果キャッシュ
        self.segmentation_result: Optional[Dict] = None
        self.depth_result: Optional[Dict] = None
        self.line_result: Optional[Dict] = None
        self.decision: Optional[Dict] = None

        # 表示用
        self.ground_display = None
        self.front_display = None
        self.result_panel = None
        self.frame_slider = None
        self.frame_info_label = None
        self.mode_toggle = None

    def _init_processing_modules(self):
        """処理モジュールを初期化"""
        try:
            # セグメンテーション
            seg_config = self.config.get('processing', {}).get('segmentation', {})
            model_path = seg_config.get('model_path', 'models/segmentation_model.onnx')
            input_size = tuple(seg_config.get('input_size', [224, 224]))
            use_cuda = seg_config.get('use_cuda', True)

            if Path(model_path).exists():
                try:
                    self.segmenter = ONNXSegmenter(
                        model_path=model_path,
                        input_size=input_size,
                        use_cuda=use_cuda
                    )
                    # Model is loaded in __init__
                    print(f"✓ Segmentation model loaded: {model_path}")
                except Exception as e:
                    print(f"Error loading segmentation model: {e}")
                    self.segmenter = None
            else:
                print(f"Warning: Segmentation model not found: {model_path}")
                self.segmenter = None

            # 深度推定
            depth_config = self.config.get('processing', {}).get('depth', {})
            depth_model_path = depth_config.get('model_path', 'models/depth_model.onnx')
            depth_input_size = tuple(depth_config.get('input_size', [518, 518]))

            if Path(depth_model_path).exists():
                try:
                    self.depth_estimator = DepthEstimator(
                        model_path=depth_model_path,
                        input_size=depth_input_size,
                        use_cuda=use_cuda
                    )
                    # Model is loaded in __init__
                    print(f"✓ Depth model loaded: {depth_model_path}")
                except Exception as e:
                    print(f"Error loading depth model: {e}")
                    self.depth_estimator = None
            else:
                print(f"Warning: Depth model not found: {depth_model_path}")
                self.depth_estimator = None

            # ライン検出
            line_config = self.config.get('processing', {}).get('line_detection', {})
            self.line_detector = LineDetector(line_config)
            print("✓ Line detector initialized")

            # 統合判定
            self.decision_maker = IntegratedDecisionMaker(self.config)
            print("✓ Decision maker initialized")

        except Exception as e:
            print(f"Error initializing processing modules: {e}")
            self.segmenter = None
            self.depth_estimator = None
            self.line_detector = None
            self.decision_maker = None

    def create_ui(self):
        """UIを構築"""

        with ui.header().classes('items-center justify-between'):
            ui.label("JetRacer オフライン検証").classes("text-h4")

            # モード切り替え
            with ui.row().classes('items-center gap-4'):
                ui.label("Mode:")
                self.mode_toggle = ui.toggle(
                    ["障害物回避", "ライントレース"],
                    value="障害物回避",
                    on_change=self.on_mode_change
                ).classes('text-sm')

        with ui.row().classes("w-full gap-2 p-2"):
            # 左: 足元カメラ
            with ui.card().classes("flex-1"):
                ui.label("足元カメラ (セグメンテーション)").classes("text-h6 mb-2")
                self.ground_display = ui.image().classes('w-full')

            # 右: 正面カメラ
            with ui.card().classes("flex-1"):
                ui.label("正面カメラ").classes("text-h6 mb-2")
                self.front_display = ui.image().classes('w-full')

        with ui.row().classes("w-full gap-2 p-2"):
            # フレーム操作
            with ui.card().classes("flex-1"):
                ui.label("フレーム操作").classes("text-h6 mb-2")

                with ui.row().classes('w-full items-center gap-2'):
                    ui.button("◀ 前", on_click=self.prev_frame, icon='skip_previous')
                    ui.button("次 ▶", on_click=self.next_frame, icon='skip_next')

                # フレーム情報
                self.frame_info_label = ui.label("フレーム: 0 / 0").classes('text-sm')

                # スライダー
                self.frame_slider = ui.slider(
                    min=0,
                    max=1,
                    value=0,
                    on_change=self.on_slider_change
                ).classes('w-full')

            # 判定結果
            with ui.card().classes("flex-1"):
                ui.label("走行判定").classes("text-h6 mb-2")

                # 処理実行ボタン
                ui.button(
                    "処理実行",
                    on_click=self.process_frame,
                    icon='play_arrow',
                    color='primary'
                ).classes('w-full mb-2')

                self.result_panel = ui.column().classes('w-full')
                with self.result_panel:
                    ui.label("処理実行ボタンを押してください").classes('text-sm text-grey')

        # セッション読み込みボタン (Phase 1)
        with ui.card().classes("w-full p-2"):
            ui.label("セッション選択").classes("text-h6 mb-2")
            with ui.row().classes('w-full gap-2'):
                if self.session_path:
                    ui.label(f"指定されたセッション: {self.session_path}").classes('text-sm')
                    ui.button(
                        "このセッションを読み込む",
                        on_click=lambda: self.load_session(str(self.session_path)),
                        icon='folder_open',
                        color='primary'
                    )
                else:
                    ui.label("セッションが指定されていません").classes('text-sm text-grey')

    def load_session(self, path: str):
        """セッションを読み込み"""
        self.session_path = Path(path)

        if not self.session_path.exists():
            ui.notify(f"エラー: セッションが見つかりません: {path}", type='negative')
            return

        # ImageLoaderで読み込み
        self.loader = ImageLoader(str(self.session_path))
        success = self.loader.load_session()

        if not success:
            ui.notify("エラー: セッションの読み込みに失敗しました", type='negative')
            return

        # フレーム数を更新
        frame_count = len(self.loader)
        if frame_count > 0:
            self.frame_slider.max = frame_count - 1
            ui.notify(f"✓ セッション読み込み完了: {frame_count} フレーム", type='positive')

            # 最初のフレームを表示
            self.current_frame_index = 0
            self.update_display()
        else:
            ui.notify("エラー: フレームが見つかりません", type='negative')

    def update_display(self):
        """表示を更新"""
        if self.loader is None:
            return

        # フレームを取得
        frame_data = self.loader.get_frame(self.current_frame_index)
        if frame_data is None:
            return

        # フレーム情報を更新
        self.frame_info_label.text = f"フレーム: {self.current_frame_index + 1} / {len(self.loader)}"
        self.frame_slider.value = self.current_frame_index

        # 処理結果をクリア
        self.segmentation_result = None
        self.depth_result = None
        self.line_result = None
        self.decision = None

        # 足元カメラ画像を表示（元画像）
        if frame_data["ground"] is not None:
            ground_img = self._cv2_to_base64(frame_data["ground"])
            self.ground_display.source = f"data:image/jpeg;base64,{ground_img}"
        else:
            self.ground_display.source = ""

        # 正面カメラ画像を表示（元画像）
        if frame_data["front"] is not None:
            front_img = self._cv2_to_base64(frame_data["front"])
            self.front_display.source = f"data:image/jpeg;base64,{front_img}"
        else:
            self.front_display.source = ""

        # 判定結果パネルをクリア
        self.result_panel.clear()
        with self.result_panel:
            ui.label("処理実行ボタンを押してください").classes('text-sm text-grey')

    def process_frame(self):
        """現在のフレームを処理"""
        if self.loader is None:
            ui.notify("エラー: セッションが読み込まれていません", type='warning')
            return

        # フレームを取得
        frame_data = self.loader.get_frame(self.current_frame_index)
        if frame_data is None:
            ui.notify("エラー: フレームの取得に失敗しました", type='negative')
            return

        ground_image = frame_data["ground"]
        front_image = frame_data["front"]

        if ground_image is None or front_image is None:
            ui.notify("エラー: 画像が見つかりません", type='negative')
            return

        # 1. セグメンテーション処理（足元カメラ）
        if self.segmenter is not None:
            mask, seg_time = self.segmenter.inference(ground_image)

            # クラスごとの面積を計算
            class_areas = {}
            for class_id in range(3):  # 0, 1, 2
                class_areas[class_id] = np.sum(mask == class_id)

            self.segmentation_result = {
                'mask': mask,
                'class_areas': class_areas,
                'inference_time_ms': seg_time
            }

            # セグメンテーション結果をオーバーレイ表示
            overlay = self._create_segmentation_overlay(ground_image, mask)
            overlay_img = self._cv2_to_base64(overlay)
            self.ground_display.source = f"data:image/jpeg;base64,{overlay_img}"

        # 2. モードに応じて前方カメラを処理
        if self.mode == ProcessingMode.OBSTACLE_AVOIDANCE:
            # Mode A: 深度推定
            if self.depth_estimator is not None:
                depth_map, depth_time = self.depth_estimator.inference(front_image)

                # 障害物分析
                depth_config = self.config.get('processing', {}).get('depth', {})
                analyzer = ObstacleAnalyzer(depth_config)
                obstacle_analysis = analyzer.analyze(depth_map)

                self.depth_result = {
                    'depth_map': depth_map,
                    'obstacle_analysis': obstacle_analysis,
                    'inference_time_ms': depth_time
                }

                # 深度マップを可視化
                depth_vis = self._visualize_depth(front_image, depth_map, obstacle_analysis)
                depth_img = self._cv2_to_base64(depth_vis)
                self.front_display.source = f"data:image/jpeg;base64,{depth_img}"

        elif self.mode == ProcessingMode.LINE_FOLLOWING:
            # Mode B: ライン検出
            if self.line_detector is not None:
                self.line_result = self.line_detector.detect(front_image)

                # ライン検出結果を可視化
                line_vis = self._visualize_line_detection(front_image, self.line_result)
                line_img = self._cv2_to_base64(line_vis)
                self.front_display.source = f"data:image/jpeg;base64,{line_img}"

        # 3. 統合判定
        if self.decision_maker is not None and self.segmentation_result is not None:
            if self.mode == ProcessingMode.OBSTACLE_AVOIDANCE and self.depth_result is not None:
                self.decision = self.decision_maker.decide_obstacle_avoidance(
                    self.segmentation_result,
                    self.depth_result
                )
            elif self.mode == ProcessingMode.LINE_FOLLOWING and self.line_result is not None:
                self.decision = self.decision_maker.decide_line_following(
                    self.segmentation_result,
                    self.line_result
                )

        # 4. 判定結果を表示
        self._update_decision_display()

        ui.notify("✓ 処理完了", type='positive')

    def _cv2_to_base64(self, image: np.ndarray) -> str:
        """OpenCV画像をBase64エンコード"""
        # BGR → RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # PIL Imageに変換
        pil_image = Image.fromarray(rgb_image)

        # Base64エンコード
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    def prev_frame(self):
        """前のフレームへ"""
        if self.loader is None:
            return

        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.update_display()

    def next_frame(self):
        """次のフレームへ"""
        if self.loader is None:
            return

        if self.current_frame_index < len(self.loader) - 1:
            self.current_frame_index += 1
            self.update_display()

    def on_slider_change(self, e):
        """スライダー変更時"""
        if self.loader is None:
            return

        self.current_frame_index = int(e.value)
        self.update_display()

    def on_mode_change(self, e):
        """モード変更時"""
        if e.value == "障害物回避":
            self.mode = ProcessingMode.OBSTACLE_AVOIDANCE
        else:
            self.mode = ProcessingMode.LINE_FOLLOWING

        ui.notify(f"モード変更: {e.value}", type='info')

        # 現在のフレームを再処理（処理済みの場合のみ）
        if self.loader is not None and self.segmentation_result is not None:
            self.process_frame()

    def _create_segmentation_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """セグメンテーションマスクをオーバーレイ表示"""
        overlay = image.copy()

        # カラーマップ（クラスごとの色）
        colors = {
            0: [255, 0, 0],    # クラス0: 赤（車体）
            1: [0, 0, 255],    # クラス1: 青（障害物）
            2: [0, 255, 0]     # クラス2: 緑（通行可能）
        }

        # マスクをRGBに変換
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color

        # アルファブレンド
        overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)

        return overlay

    def _visualize_depth(self, image: np.ndarray, depth_map: np.ndarray, obstacle_analysis: Dict) -> np.ndarray:
        """深度マップを可視化"""
        # 深度マップをカラーマップで可視化
        depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

        # 元画像とブレンド
        h, w = image.shape[:2]
        depth_resized = cv2.resize(depth_colored, (w, h))
        vis = cv2.addWeighted(image, 0.5, depth_resized, 0.5, 0)

        # グリッド線を描画
        grid = obstacle_analysis.get('grid', np.zeros((3, 3)))
        cell_h = h // 3
        cell_w = w // 3

        for i in range(3):
            for j in range(3):
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = (j + 1) * cell_w
                y2 = (i + 1) * cell_h

                # グリッド枠を描画
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # 距離を表示
                dist = grid[i, j]
                if dist > 0:
                    text = f"{dist:.1f}cm"
                    cv2.putText(vis, text, (x1 + 10, y1 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 推奨方向を表示
        recommended_dir = obstacle_analysis.get('recommended_direction', 'center')
        warning_level = obstacle_analysis.get('warning_level', 0)

        warning_colors = {
            0: (0, 255, 0),   # 緑
            1: (0, 255, 255), # 黄
            2: (0, 165, 255), # オレンジ
            3: (0, 0, 255)    # 赤
        }

        warning_color = warning_colors.get(warning_level, (255, 255, 255))

        cv2.putText(vis, f"Warning: Level {warning_level}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
        cv2.putText(vis, f"Direction: {recommended_dir}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)

        return vis

    def _visualize_line_detection(self, image: np.ndarray, line_result: Dict) -> np.ndarray:
        """ライン検出結果を可視化"""
        vis = image.copy()

        # ラインを描画
        lines = line_result.get('lines', [])
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 目標点を描画
        target_point = line_result.get('target_point')
        if target_point:
            cx, cy = target_point
            cv2.circle(vis, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(vis, (cx, cy), 15, (255, 0, 0), 3)

            # 画像中央との線
            center_x = vis.shape[1] // 2
            cv2.line(vis, (center_x, cy), (cx, cy), (255, 0, 0), 2)

        # ステアリング情報を表示
        steering_offset = line_result.get('steering_offset', 0.0)
        confidence = line_result.get('confidence', 0.0)

        cv2.putText(vis, f"Offset: {steering_offset:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"Confidence: {confidence:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return vis

    def _update_decision_display(self):
        """判定結果パネルを更新"""
        self.result_panel.clear()

        if self.decision is None:
            with self.result_panel:
                ui.label("判定結果なし").classes('text-sm text-grey')
            return

        with self.result_panel:
            # コマンド
            command_colors = {
                'go': 'positive',
                'slow': 'warning',
                'stop': 'negative',
                'turn_left': 'info',
                'turn_right': 'info',
                'emergency_stop': 'negative'
            }

            command_value = self.decision.command.value
            command_color = command_colors.get(command_value, 'grey')

            with ui.card().classes(f'w-full p-2 bg-{command_color}'):
                ui.label(f"コマンド: {command_value.upper()}").classes('text-h6 font-bold text-white')

            # 詳細情報
            with ui.grid(columns=2).classes('w-full gap-2 mt-2'):
                ui.label("速度:").classes('font-bold')
                ui.label(f"{self.decision.speed:.2f}")

                ui.label("ステアリング:").classes('font-bold')
                ui.label(f"{self.decision.steering:.2f}")

                ui.label("信頼度:").classes('font-bold')
                ui.label(f"{self.decision.confidence:.2f}")

                ui.label("地面安全:").classes('font-bold')
                ui.label("✓" if self.decision.ground_safe else "✗")

                ui.label("前方クリア:").classes('font-bold')
                ui.label("✓" if self.decision.front_clear else "✗")

            # 理由
            ui.separator()
            ui.label("理由:").classes('font-bold text-sm')
            ui.label(self.decision.reason).classes('text-sm')

            # 処理時間
            if self.segmentation_result:
                ui.separator()
                ui.label("処理時間:").classes('font-bold text-sm')
                seg_time = self.segmentation_result.get('inference_time_ms', 0)
                ui.label(f"Segmentation: {seg_time:.1f}ms").classes('text-xs')

                if self.depth_result:
                    depth_time = self.depth_result.get('inference_time_ms', 0)
                    ui.label(f"Depth: {depth_time:.1f}ms").classes('text-xs')

                if self.line_result:
                    ui.label("Line: < 1ms").classes('text-xs')


def main(session_path: Optional[str] = None, port: int = 8084):
    """メインエントリポイント"""
    try:
        viewer = OfflineViewerUI(session_path=session_path)
        viewer.create_ui()
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        # Create minimal error UI
        ui.label(f"Error: {str(e)}").classes("text-negative")

    ui.run(port=port, title="JetRacer Offline Viewer")

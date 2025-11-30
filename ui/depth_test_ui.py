"""
深度推定テストUI

深度推定の動作確認とキャリブレーション調整。
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

from core.depth_estimation import DepthEstimator, DepthCalibrator, ObstacleAnalyzer
from core.image_loader import SingleFolderLoader


class DepthTestUI:
    """
    深度推定テストUI

    機能:
        1. 画像選択・深度推定実行
        2. 深度マップの可視化（カラーマップ）
        3. 車体基準領域の設定
        4. キャリブレーション実行
        5. 距離閾値の調整・プレビュー
    """

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)
        self.config: Optional[Dict] = None
        self.load_config()

        # 画像関連
        self.image_folder = Path("demo_images")
        self.current_image: Optional[np.ndarray] = None
        self.current_image_path: Optional[Path] = None
        self.image_paths: list = []

        # 深度推定
        self.estimator: Optional[DepthEstimator] = None
        self.calibrator: Optional[DepthCalibrator] = None
        self.analyzer: Optional[ObstacleAnalyzer] = None
        self.depth_map: Optional[np.ndarray] = None
        self.analysis_result: Optional[Dict] = None

        # UI要素
        self.original_display = None
        self.depth_display = None
        self.result_panel = None
        self.calibration_button = None
        self.inference_time_label = None

    def load_config(self):
        """設定ファイルを読み込み"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ Loaded config from {self.config_path}")
        else:
            print(f"Warning: Config file not found: {self.config_path}")
            self.config = {}

    def create_ui(self):
        """UIを構築"""

        with ui.header().classes('items-center justify-between'):
            ui.label("JetRacer 深度推定テスト").classes("text-h4")

        with ui.row().classes("w-full gap-2 p-2"):
            # 左: 元画像
            with ui.card().classes("flex-1"):
                ui.label("元画像").classes("text-h6 mb-2")
                self.original_display = ui.image().classes('w-full')

                # 画像選択
                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.button("画像を選択", on_click=self.load_demo_images, icon='folder')
                    ui.button("深度推定実行", on_click=self.run_inference, icon='play_arrow', color='primary')

                self.inference_time_label = ui.label("推論時間: - ms").classes('text-sm text-grey')

            # 中央: 深度マップ
            with ui.card().classes("flex-1"):
                ui.label("深度マップ").classes("text-h6 mb-2")
                self.depth_display = ui.image().classes('w-full')

                # カラーマップ選択
                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.label("カラーマップ:").classes('text-sm')
                    ui.select(
                        ['TURBO', 'JET', 'HOT', 'VIRIDIS'],
                        value='TURBO',
                        on_change=self.update_depth_visualization
                    ).classes('w-32')

            # 右: 設定・結果
            with ui.card().classes("flex-1"):
                ui.label("設定・結果").classes("text-h6 mb-2")

                # モデルステータス
                with ui.column().classes('w-full gap-2'):
                    model_status = ui.label("モデル: 未読み込み").classes('text-sm')

                    # モデル読み込みボタン
                    ui.button("モデル読み込み", on_click=self.load_model, icon='download')

                ui.separator()

                # 車体基準ROI設定
                ui.label("車体基準ROI").classes('text-subtitle1 mt-2')
                with ui.grid(columns=4).classes('w-full gap-2'):
                    ui.label("x1:")
                    self.roi_x1 = ui.number(value=240, min=0, max=640).classes('w-20')
                    ui.label("y1:")
                    self.roi_y1 = ui.number(value=420, min=0, max=480).classes('w-20')
                    ui.label("x2:")
                    self.roi_x2 = ui.number(value=400, min=0, max=640).classes('w-20')
                    ui.label("y2:")
                    self.roi_y2 = ui.number(value=480, min=0, max=480).classes('w-20')

                with ui.row().classes('w-full gap-2'):
                    ui.label("基準距離 (cm):")
                    self.ref_distance = ui.number(value=5.0, min=1.0, max=100.0, step=0.5).classes('w-24')

                # キャリブレーションボタン
                self.calibration_button = ui.button(
                    "キャリブレーション実行",
                    on_click=self.run_calibration,
                    icon='settings',
                    color='orange'
                )
                self.calibration_button.disable()

                ui.separator()

                # 結果表示
                self.result_panel = ui.column().classes('w-full')
                with self.result_panel:
                    ui.label("距離分析結果").classes('text-subtitle1')
                    ui.label("深度推定を実行してください").classes('text-sm text-grey')

    def load_demo_images(self):
        """デモ画像を読み込み"""
        loader = SingleFolderLoader(str(self.image_folder))
        self.image_paths = loader.load_paths()

        if self.image_paths:
            # 最初の画像を表示
            self.current_image_path = self.image_paths[0]
            self.current_image = cv2.imread(str(self.current_image_path))

            if self.current_image is not None:
                self._display_image(self.original_display, self.current_image)
                ui.notify(f"✓ {len(self.image_paths)} 枚の画像を読み込みました", type='positive')
            else:
                ui.notify("エラー: 画像の読み込みに失敗しました", type='negative')
        else:
            ui.notify("エラー: 画像が見つかりません", type='negative')

    def load_model(self):
        """深度推定モデルを読み込み"""
        if self.config is None:
            ui.notify("エラー: 設定ファイルが読み込まれていません", type='negative')
            return

        depth_config = self.config.get('processing', {}).get('depth', {})
        model_path = depth_config.get('model_path', 'output/models/depth_anything_v2_small.onnx')
        input_size = tuple(depth_config.get('input_size', [320, 240]))
        use_cuda = depth_config.get('use_cuda', True)

        # モデルの存在確認
        if not Path(model_path).exists():
            ui.notify(
                f"警告: モデルファイルが見つかりません: {model_path}\n"
                "Phase 2では実際のモデルが必要です。",
                type='warning'
            )
            return

        # DepthEstimatorを初期化
        self.estimator = DepthEstimator(
            model_path=model_path,
            input_size=input_size,
            use_cuda=use_cuda
        )

        # モデル読み込み
        success = self.estimator.load_model()

        if success:
            ui.notify("✓ モデルを読み込みました", type='positive')
            self.calibration_button.enable()
        else:
            ui.notify("エラー: モデルの読み込みに失敗しました", type='negative')

    def run_inference(self):
        """深度推定を実行"""
        if self.current_image is None:
            ui.notify("エラー: 画像が読み込まれていません", type='warning')
            return

        if self.estimator is None:
            ui.notify("エラー: モデルが読み込まれていません。先にモデルを読み込んでください。", type='warning')
            return

        try:
            # 深度推定実行
            self.depth_map, inference_time = self.estimator.inference(self.current_image)

            # 推論時間を表示
            self.inference_time_label.text = f"推論時間: {inference_time:.1f} ms"

            # 深度マップを可視化
            self.update_depth_visualization()

            ui.notify(f"✓ 深度推定完了 ({inference_time:.1f} ms)", type='positive')

        except Exception as e:
            ui.notify(f"エラー: {e}", type='negative')
            import traceback
            traceback.print_exc()

    def update_depth_visualization(self, e=None):
        """深度マップの可視化を更新"""
        if self.depth_map is None:
            return

        # カラーマップを適用
        colormap_name = e.value if e else 'TURBO'
        colormap = getattr(cv2, f'COLORMAP_{colormap_name}', cv2.COLORMAP_TURBO)

        # 0-255にスケール
        depth_vis = (self.depth_map * 255).astype(np.uint8)

        # カラーマップ適用
        depth_colored = cv2.applyColorMap(depth_vis, colormap)

        # ROIグリッドを描画
        self._draw_roi_grid(depth_colored)

        # 表示
        self._display_image(self.depth_display, depth_colored)

    def _draw_roi_grid(self, image: np.ndarray):
        """ROIグリッド（3x3 + 車体基準）を描画"""
        h, w = image.shape[:2]

        # 3x3グリッド
        exclude_ratio = 0.125
        effective_h = int(h * (1 - exclude_ratio))

        row_step = effective_h // 3
        col_step = w // 3

        # グリッド線を描画
        for i in range(1, 3):
            # 縦線
            cv2.line(image, (i * col_step, 0), (i * col_step, effective_h), (0, 255, 0), 2)
            # 横線
            cv2.line(image, (0, i * row_step), (w, i * row_step), (0, 255, 0), 2)

        # 車体基準領域を描画
        x1 = int(self.roi_x1.value)
        y1 = int(self.roi_y1.value)
        x2 = int(self.roi_x2.value)
        y2 = int(self.roi_y2.value)

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(image, "Vehicle Ref", (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def run_calibration(self):
        """キャリブレーションを実行"""
        if self.depth_map is None:
            ui.notify("エラー: 深度マップがありません。先に深度推定を実行してください。", type='warning')
            return

        # ROIと基準距離を取得
        roi = (
            int(self.roi_x1.value),
            int(self.roi_y1.value),
            int(self.roi_x2.value),
            int(self.roi_y2.value)
        )
        ref_distance = float(self.ref_distance.value)

        # キャリブレーター作成
        self.calibrator = DepthCalibrator(
            reference_roi=roi,
            reference_distance_cm=ref_distance
        )

        # キャリブレーション実行
        ref_depth = self.calibrator.calibrate(self.depth_map)

        # 障害物分析器を作成
        obstacle_config = self.config.get('decision', {}).get('obstacle_avoidance', {})
        self.analyzer = ObstacleAnalyzer(self.calibrator, obstacle_config)

        # 分析実行
        self.analysis_result = self.analyzer.analyze(self.depth_map)

        # 結果を表示
        self.display_analysis_result()

        ui.notify("✓ キャリブレーション完了", type='positive')

    def display_analysis_result(self):
        """分析結果を表示"""
        if self.analysis_result is None:
            return

        self.result_panel.clear()

        with self.result_panel:
            ui.label("距離分析結果").classes('text-subtitle1')

            # 最小距離と警告レベル
            min_dist = self.analysis_result['min_distance_cm']
            warning = self.analysis_result['warning_level']
            direction = self.analysis_result['recommended_direction']

            # 警告レベルに応じた色
            warning_colors = {
                'SAFE': 'positive',
                'SLOW': 'warning',
                'STOP': 'negative',
                'EMERGENCY': 'negative'
            }
            color = warning_colors.get(warning, 'grey')

            with ui.card().classes('w-full'):
                ui.label(f"最小距離: {min_dist:.1f} cm").classes('text-lg font-bold')
                ui.badge(warning, color=color).classes('text-sm')
                ui.label(f"推奨: {direction}").classes('text-sm')

            # 各領域の距離
            ui.label("領域別距離").classes('text-subtitle2 mt-2')
            regions = self.analysis_result['regions']

            with ui.grid(columns=3).classes('w-full gap-1'):
                for row_name in ['top', 'mid', 'bot']:
                    for col_name in ['left', 'center', 'right']:
                        region_key = f"{row_name}_{col_name}"
                        distance = regions.get(region_key, 0)

                        with ui.card().classes('p-2'):
                            ui.label(f"{distance:.1f} cm").classes('text-center text-sm')

    def _display_image(self, ui_element, image: np.ndarray):
        """画像をUI要素に表示"""
        # BGR → RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # PIL Imageに変換
        pil_image = Image.fromarray(rgb_image)

        # Base64エンコード
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        ui_element.source = f"data:image/jpeg;base64,{img_str}"


def main(port: int = 8085):
    """メインエントリポイント"""
    depth_ui = DepthTestUI()
    depth_ui.create_ui()
    ui.run(port=port, title="JetRacer Depth Test")


if __name__ in {"__main__", "__mp_main__"}:
    main()

"""
ライン検出テストUI

ライン検出の動作確認とパラメータ調整。
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

from core.line_detection import LineDetector
from core.image_loader import SingleFolderLoader


class LineTestUI:
    """
    ライン検出テストUI

    機能:
        1. 画像選択・ライン検出実行
        2. 白線/黄線マスクの表示
        3. 検出ラインの表示
        4. 走行目標点の表示
        5. HSVパラメータのリアルタイム調整
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

        # ライン検出
        self.detector: Optional[LineDetector] = None
        self.detection_result: Optional[Dict] = None

        # UI要素
        self.result_display = None
        self.white_mask_display = None
        self.yellow_mask_display = None
        self.combined_mask_display = None
        self.steering_bar = None
        self.confidence_label = None

        # HSVパラメータ（スライダー用）
        self.white_params = {}
        self.yellow_params = {}

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
            ui.label("JetRacer ライン検出テスト").classes("text-h4")

        with ui.row().classes("w-full gap-2 p-2"):
            # 左: 元画像 + 検出結果オーバーレイ
            with ui.card().classes("flex-1"):
                ui.label("検出結果").classes("text-h6 mb-2")
                self.result_display = ui.image().classes('w-full')

                # 画像選択と検出実行
                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.button("画像を選択", on_click=self.load_demo_images, icon='folder')
                    ui.button("ライン検出実行", on_click=self.run_detection, icon='play_arrow', color='primary')

            # 右: マスク表示
            with ui.card().classes("flex-1"):
                ui.label("マスク").classes("text-h6 mb-2")

                with ui.tabs().classes('w-full') as tabs:
                    white_tab = ui.tab("白線", icon='radio_button_checked')
                    yellow_tab = ui.tab("黄線", icon='radio_button_checked')
                    combined_tab = ui.tab("統合", icon='layers')

                with ui.tab_panels(tabs, value=white_tab).classes('w-full'):
                    with ui.tab_panel(white_tab):
                        self.white_mask_display = ui.image().classes('w-full')

                    with ui.tab_panel(yellow_tab):
                        self.yellow_mask_display = ui.image().classes('w-full')

                    with ui.tab_panel(combined_tab):
                        self.combined_mask_display = ui.image().classes('w-full')

        with ui.row().classes("w-full gap-2 p-2"):
            # パラメータ調整パネル
            with ui.card().classes("flex-1"):
                ui.label("白線パラメータ").classes("text-h6 mb-2")

                with ui.column().classes('w-full gap-2'):
                    # H range
                    ui.label("H (色相): 0-179").classes('text-sm')
                    with ui.row().classes('w-full gap-2'):
                        self.white_params['h_min'] = ui.slider(min=0, max=179, value=0, on_change=self.on_param_change).classes('flex-1')
                        self.white_params['h_max'] = ui.slider(min=0, max=179, value=180, on_change=self.on_param_change).classes('flex-1')

                    # S range
                    ui.label("S (彩度): 0-255").classes('text-sm')
                    with ui.row().classes('w-full gap-2'):
                        self.white_params['s_min'] = ui.slider(min=0, max=255, value=0, on_change=self.on_param_change).classes('flex-1')
                        self.white_params['s_max'] = ui.slider(min=0, max=255, value=30, on_change=self.on_param_change).classes('flex-1')

                    # V range
                    ui.label("V (明度): 0-255").classes('text-sm')
                    with ui.row().classes('w-full gap-2'):
                        self.white_params['v_min'] = ui.slider(min=0, max=255, value=200, on_change=self.on_param_change).classes('flex-1')
                        self.white_params['v_max'] = ui.slider(min=0, max=255, value=255, on_change=self.on_param_change).classes('flex-1')

            with ui.card().classes("flex-1"):
                ui.label("黄線パラメータ").classes("text-h6 mb-2")

                with ui.column().classes('w-full gap-2'):
                    # H range
                    ui.label("H (色相): 0-179").classes('text-sm')
                    with ui.row().classes('w-full gap-2'):
                        self.yellow_params['h_min'] = ui.slider(min=0, max=179, value=20, on_change=self.on_param_change).classes('flex-1')
                        self.yellow_params['h_max'] = ui.slider(min=0, max=179, value=40, on_change=self.on_param_change).classes('flex-1')

                    # S range
                    ui.label("S (彩度): 0-255").classes('text-sm')
                    with ui.row().classes('w-full gap-2'):
                        self.yellow_params['s_min'] = ui.slider(min=0, max=255, value=100, on_change=self.on_param_change).classes('flex-1')
                        self.yellow_params['s_max'] = ui.slider(min=0, max=255, value=255, on_change=self.on_param_change).classes('flex-1')

                    # V range
                    ui.label("V (明度): 0-255").classes('text-sm')
                    with ui.row().classes('w-full gap-2'):
                        self.yellow_params['v_min'] = ui.slider(min=0, max=255, value=100, on_change=self.on_param_change).classes('flex-1')
                        self.yellow_params['v_max'] = ui.slider(min=0, max=255, value=255, on_change=self.on_param_change).classes('flex-1')

        with ui.row().classes("w-full gap-2 p-2"):
            # ステアリング結果
            with ui.card().classes("w-full"):
                ui.label("ステアリング").classes("text-h6 mb-2")

                with ui.row().classes('w-full items-center gap-4'):
                    ui.label("Left").classes('text-sm')
                    self.steering_bar = ui.linear_progress(value=0.5, show_value=False).classes('flex-1')
                    ui.label("Right").classes('text-sm')

                with ui.row().classes('w-full gap-4 mt-2'):
                    ui.label("Offset:").classes('text-sm')
                    self.steering_offset_label = ui.label("0.00").classes('text-lg font-bold')

                    ui.label("Confidence:").classes('text-sm')
                    self.confidence_label = ui.label("0.00").classes('text-lg font-bold')

    def load_demo_images(self):
        """デモ画像を読み込み"""
        loader = SingleFolderLoader(str(self.image_folder))
        self.image_paths = loader.load_paths()

        if self.image_paths:
            # 最初の画像を表示
            self.current_image_path = self.image_paths[0]
            self.current_image = cv2.imread(str(self.current_image_path))

            if self.current_image is not None:
                self._display_image(self.result_display, self.current_image)
                ui.notify(f"✓ {len(self.image_paths)} 枚の画像を読み込みました", type='positive')
            else:
                ui.notify("エラー: 画像の読み込みに失敗しました", type='negative')
        else:
            ui.notify("エラー: 画像が見つかりません", type='negative')

    def run_detection(self):
        """ライン検出を実行"""
        if self.current_image is None:
            ui.notify("エラー: 画像が読み込まれていません", type='warning')
            return

        # 現在のパラメータでLineDetectorを作成
        line_config = self._build_line_config()

        self.detector = LineDetector(line_config)

        # 検出実行
        self.detection_result = self.detector.detect(self.current_image)

        # 結果を表示
        self.update_displays()

        ui.notify("✓ ライン検出完了", type='positive')

    def on_param_change(self, e=None):
        """パラメータ変更時に再検出"""
        if self.current_image is not None:
            self.run_detection()

    def _build_line_config(self) -> Dict:
        """現在のUIパラメータから設定を構築"""
        # デフォルト設定を取得
        if self.config:
            line_config = self.config.get('processing', {}).get('line_detection', {}).copy()
        else:
            line_config = {}

        # UIパラメータで上書き
        line_config['white_line'] = {
            'h_range': [int(self.white_params['h_min'].value), int(self.white_params['h_max'].value)],
            's_range': [int(self.white_params['s_min'].value), int(self.white_params['s_max'].value)],
            'v_range': [int(self.white_params['v_min'].value), int(self.white_params['v_max'].value)]
        }

        line_config['yellow_line'] = {
            'h_range': [int(self.yellow_params['h_min'].value), int(self.yellow_params['h_max'].value)],
            's_range': [int(self.yellow_params['s_min'].value), int(self.yellow_params['s_max'].value)],
            'v_range': [int(self.yellow_params['v_min'].value), int(self.yellow_params['v_max'].value)]
        }

        return line_config

    def update_displays(self):
        """検出結果を表示"""
        if self.detection_result is None:
            return

        # マスク表示
        self._display_mask(self.white_mask_display, self.detection_result['white_mask'])
        self._display_mask(self.yellow_mask_display, self.detection_result['yellow_mask'])
        self._display_mask(self.combined_mask_display, self.detection_result['combined_mask'])

        # 検出結果オーバーレイ
        result_image = self.current_image.copy()

        # ラインを描画
        for line in self.detection_result['lines']:
            x1, y1, x2, y2 = line
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 走行目標点を描画
        target_point = self.detection_result['target_point']
        if target_point:
            cx, cy = target_point
            cv2.circle(result_image, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(result_image, (cx, cy), 15, (255, 0, 0), 3)

            # 画像中央との線
            center_x = result_image.shape[1] // 2
            cv2.line(result_image, (center_x, cy), (cx, cy), (255, 0, 0), 2)

        self._display_image(self.result_display, result_image)

        # ステアリング情報を更新
        steering_offset = self.detection_result['steering_offset']
        confidence = self.detection_result['confidence']

        # ステアリングバー: 0.0 (左) ~ 1.0 (右)
        bar_value = (steering_offset + 1.0) / 2.0
        self.steering_bar.value = bar_value

        self.steering_offset_label.text = f"{steering_offset:.2f}"
        self.confidence_label.text = f"{confidence:.2f}"

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

    def _display_mask(self, ui_element, mask: np.ndarray):
        """マスクをUI要素に表示"""
        # グレースケールマスクをRGBに変換
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # PIL Imageに変換
        pil_image = Image.fromarray(mask_rgb)

        # Base64エンコード
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        ui_element.source = f"data:image/png;base64,{img_str}"


def main(port: int = 8086):
    """メインエントリポイント"""
    line_ui = LineTestUI()
    line_ui.create_ui()
    ui.run(port=port, title="JetRacer Line Test")


if __name__ in {"__main__", "__mp_main__"}:
    main()

"""
環境キャリブレーションUI

NiceGUIを使用した環境キャリブレーションツール。
ユーザーが画像上で対象物を選択し、色閾値を自動算出する。
"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import cv2
import numpy as np
import base64
import io
from PIL import Image

from core.line_detection import ColorCalibrator
from core.calibration import EnvironmentManager
from core.image_loader import SingleFolderLoader


class CalibrationUI:
    """
    環境キャリブレーションUI

    機能:
        1. 画像選択・表示
        2. 対象物の範囲選択（白線、黄線、床、車体基準）
        3. HSV範囲の自動算出
        4. 検出結果のプレビュー
        5. 環境設定の保存
    """

    def __init__(self, image_dir: str = "demo_images"):
        self.image_dir = Path(image_dir)
        self.current_image: Optional[np.ndarray] = None
        self.current_image_path: Optional[Path] = None
        self.image_paths: List[Path] = []

        # キャリブレーター
        self.calibrator = ColorCalibrator()
        self.env_manager = EnvironmentManager()

        # 選択中の対象
        self.selection_mode: str = "white_line"  # white_line, yellow_line, floor, vehicle_ref
        self.selections: Dict[str, Dict] = {}

        # ROI選択（簡易実装：テキスト入力）
        self.roi_inputs = {}

        # UI要素の参照
        self.image_display = None
        self.preview_display = None
        self.result_panel = None

    def create_ui(self):
        """UIを構築"""

        with ui.header().classes('items-center justify-between'):
            ui.label("JetRacer 環境キャリブレーション").classes("text-h4")

        with ui.row().classes("w-full gap-2 p-2"):
            # 左パネル: 画像選択・表示
            with ui.card().classes("flex-1"):
                ui.label("1. 画像を選択").classes("text-h6 mb-2")

                # 画像選択
                with ui.row().classes('w-full gap-2'):
                    ui.button("画像を選択", on_click=self.load_images, icon='folder')
                    ui.button("次の画像", on_click=self.next_image, icon='skip_next')

                # 画像表示
                self.image_display = ui.image().classes('w-full')

                ui.label("選択領域 (x1, y1, x2, y2)").classes('text-sm mt-2')
                with ui.grid(columns=4).classes('w-full gap-2'):
                    self.roi_inputs['x1'] = ui.number(value=100, min=0, max=640).classes('w-20')
                    self.roi_inputs['y1'] = ui.number(value=100, min=0, max=480).classes('w-20')
                    self.roi_inputs['x2'] = ui.number(value=200, min=0, max=640).classes('w-20')
                    self.roi_inputs['y2'] = ui.number(value=200, min=0, max=480).classes('w-20')

                ui.button("範囲を計算", on_click=self.calculate_hsv_range, icon='calculate', color='primary')

            # 右パネル: 対象選択・設定
            with ui.card().classes("flex-1"):
                ui.label("2. 対象を選択").classes("text-h6 mb-2")

                # 対象選択ラジオボタン
                with ui.column().classes('w-full gap-2'):
                    ui.radio(
                        ['白線', '黄線', '床', '車体基準'],
                        value='白線',
                        on_change=self.on_target_change
                    ).props('inline')

                ui.separator()

                # 選択済み一覧
                self.result_panel = ui.column().classes('w-full')
                with self.result_panel:
                    ui.label("HSV範囲を計算してください").classes('text-sm text-grey')

        with ui.row().classes("w-full gap-2 p-2"):
            # 下パネル: プレビュー・保存
            with ui.card().classes("w-1/2"):
                ui.label("3. プレビュー").classes("text-h6 mb-2")

                # 検出結果プレビュー
                self.preview_display = ui.image().classes('w-full')

                # プレビューボタン
                ui.button("プレビュー", on_click=self.update_preview, icon='preview')

            with ui.card().classes("w-1/2"):
                ui.label("4. 保存").classes("text-h6 mb-2")

                # 環境名入力
                with ui.row().classes('w-full gap-2'):
                    ui.label("環境名:")
                    self.env_name_input = ui.input(
                        value=f"環境_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        placeholder="例: 室内_蛍光灯"
                    ).classes('flex-1')

                # 保存ボタン
                ui.button("環境設定を保存", on_click=self.save_environment, icon='save', color='green')

                ui.separator()

                # 環境一覧
                ui.label("保存済み環境").classes('text-subtitle2 mt-2')
                self.env_list = ui.column().classes('w-full')
                self.update_env_list()

    def load_images(self):
        """画像を読み込み"""
        loader = SingleFolderLoader(str(self.image_dir))
        self.image_paths = loader.load_paths()

        if self.image_paths:
            self.current_image_path = self.image_paths[0]
            self.current_image = cv2.imread(str(self.current_image_path))

            if self.current_image is not None:
                self._display_image(self.image_display, self.current_image)
                ui.notify(f"✓ {len(self.image_paths)} 枚の画像を読み込みました", type='positive')
            else:
                ui.notify("エラー: 画像の読み込みに失敗しました", type='negative')
        else:
            ui.notify("エラー: 画像が見つかりません", type='negative')

    def next_image(self):
        """次の画像を表示"""
        if not self.image_paths:
            ui.notify("エラー: 画像が読み込まれていません", type='warning')
            return

        current_index = self.image_paths.index(self.current_image_path) if self.current_image_path in self.image_paths else 0
        next_index = (current_index + 1) % len(self.image_paths)

        self.current_image_path = self.image_paths[next_index]
        self.current_image = cv2.imread(str(self.current_image_path))

        if self.current_image is not None:
            self._display_image(self.image_display, self.current_image)
        else:
            ui.notify("エラー: 画像の読み込みに失敗しました", type='negative')

    def on_target_change(self, e):
        """対象変更時"""
        target_map = {
            '白線': 'white_line',
            '黄線': 'yellow_line',
            '床': 'floor',
            '車体基準': 'vehicle_ref'
        }
        self.selection_mode = target_map.get(e.value, 'white_line')

    def calculate_hsv_range(self):
        """HSV範囲を計算"""
        if self.current_image is None:
            ui.notify("エラー: 画像が読み込まれていません", type='warning')
            return

        # ROIを取得
        roi = (
            int(self.roi_inputs['x1'].value),
            int(self.roi_inputs['y1'].value),
            int(self.roi_inputs['x2'].value),
            int(self.roi_inputs['y2'].value)
        )

        # HSV範囲を計算
        hsv_range = self.calibrator.calculate_hsv_range(self.current_image, roi, margin_sigma=2.0)

        # 結果を保存
        self.selections[self.selection_mode] = hsv_range

        # 結果を表示
        self.display_results()

        ui.notify(f"✓ {self.selection_mode} のHSV範囲を計算しました", type='positive')

    def display_results(self):
        """計算結果を表示"""
        self.result_panel.clear()

        with self.result_panel:
            ui.label("計算済みHSV範囲").classes('text-subtitle1 mb-2')

            for target, hsv_range in self.selections.items():
                with ui.card().classes('w-full p-2 mb-2'):
                    target_names = {
                        'white_line': '白線',
                        'yellow_line': '黄線',
                        'floor': '床',
                        'vehicle_ref': '車体基準'
                    }
                    ui.label(target_names.get(target, target)).classes('font-bold')

                    h_range = hsv_range['h_range']
                    s_range = hsv_range['s_range']
                    v_range = hsv_range['v_range']

                    ui.label(f"H: {h_range[0]} - {h_range[1]}").classes('text-sm')
                    ui.label(f"S: {s_range[0]} - {s_range[1]}").classes('text-sm')
                    ui.label(f"V: {v_range[0]} - {v_range[1]}").classes('text-sm')

    def update_preview(self):
        """プレビューを更新"""
        if self.current_image is None:
            ui.notify("エラー: 画像が読み込まれていません", type='warning')
            return

        if not self.selections:
            ui.notify("エラー: HSV範囲が計算されていません", type='warning')
            return

        # 全てのマスクを統合してプレビュー
        preview = self.current_image.copy()

        for target, hsv_range in self.selections.items():
            mask_preview = self.calibrator.preview_detection(self.current_image, hsv_range, alpha=0.3)
            preview = cv2.addWeighted(preview, 0.5, mask_preview, 0.5, 0)

        self._display_image(self.preview_display, preview)

    def save_environment(self):
        """環境設定を保存"""
        if not self.selections:
            ui.notify("エラー: HSV範囲が計算されていません", type='warning')
            return

        env_name = self.env_name_input.value

        if not env_name:
            ui.notify("エラー: 環境名を入力してください", type='warning')
            return

        # 設定を構築
        config = {
            'environment_name': env_name,
            'created_at': datetime.now().isoformat(),
            'calibration': {}
        }

        # 色キャリブレーション
        if 'white_line' in self.selections:
            config['calibration']['white_line'] = self.selections['white_line']

        if 'yellow_line' in self.selections:
            config['calibration']['yellow_line'] = self.selections['yellow_line']

        if 'floor' in self.selections:
            config['calibration']['floor'] = self.selections['floor']

        # 深度キャリブレーション（車体基準）
        if 'vehicle_ref' in self.selections:
            # 車体基準は深度推定で使用
            roi = (
                int(self.roi_inputs['x1'].value),
                int(self.roi_inputs['y1'].value),
                int(self.roi_inputs['x2'].value),
                int(self.roi_inputs['y2'].value)
            )
            config['calibration']['depth'] = {
                'reference_roi': roi,
                'reference_distance_cm': 5.0  # デフォルト値
            }

        # 保存
        saved_path = self.env_manager.save_environment(env_name, config)

        # 環境一覧を更新
        self.update_env_list()

        ui.notify(f"✓ 環境設定を保存しました: {env_name}", type='positive')

    def update_env_list(self):
        """環境一覧を更新"""
        self.env_list.clear()

        env_names = self.env_manager.list_environments()

        if not env_names:
            with self.env_list:
                ui.label("保存済み環境はありません").classes('text-sm text-grey')
            return

        with self.env_list:
            for env_name in env_names:
                with ui.row().classes('w-full items-center gap-2'):
                    ui.label(env_name).classes('flex-1 text-sm')
                    ui.button(
                        "現在に設定",
                        on_click=lambda n=env_name: self.set_current_env(n),
                        icon='check_circle',
                        color='positive'
                    ).props('dense flat')
                    ui.button(
                        "削除",
                        on_click=lambda n=env_name: self.delete_env(n),
                        icon='delete',
                        color='negative'
                    ).props('dense flat')

    def set_current_env(self, env_name: str):
        """環境を現在に設定"""
        success = self.env_manager.set_current(env_name)
        if success:
            ui.notify(f"✓ 現在の環境を {env_name} に設定しました", type='positive')
        else:
            ui.notify(f"エラー: 環境の設定に失敗しました", type='negative')

    def delete_env(self, env_name: str):
        """環境を削除"""
        success = self.env_manager.delete_environment(env_name)
        if success:
            self.update_env_list()
            ui.notify(f"✓ 環境 {env_name} を削除しました", type='positive')
        else:
            ui.notify(f"エラー: 環境の削除に失敗しました", type='negative')

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


def main(port: int = 8083):
    """メインエントリポイント"""
    calibration_ui = CalibrationUI()
    calibration_ui.create_ui()
    ui.run(port=port, title="JetRacer Calibration")


if __name__ in {"__main__", "__mp_main__"}:
    main()

"""
ライン検出モジュール

白線・黄線を検出し、走行目標を算出する。
環境キャリブレーションにより、照明・床色の変動に対応。
"""

from typing import Tuple, Dict, Optional, List
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import yaml


class LineDetector:
    """
    白線・黄線検出

    使用例:
        detector = LineDetector(config)
        result = detector.detect(image)
        # result = {
        #     "white_mask": np.ndarray,
        #     "yellow_mask": np.ndarray,
        #     "combined_mask": np.ndarray,
        #     "lines": [...],
        #     "target_point": (x, y),
        #     "steering_offset": 0.1,
        #     "confidence": 0.85
        # }
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 検出設定
                - white_line: {h_range, s_range, v_range}
                - yellow_line: {h_range, s_range, v_range}
                - preprocessing: {blur_kernel, clahe_clip_limit, ...}
                - roi: {top_ratio, bottom_ratio}
                - hough: {rho, theta_degrees, threshold, ...}
        """
        self.config = config

        # 前処理パラメータ
        preproc = config.get('preprocessing', {})
        self.blur_kernel = preproc.get('blur_kernel', 5)
        self.clahe_clip_limit = preproc.get('clahe_clip_limit', 2.0)
        self.clahe_tile_size = tuple(preproc.get('clahe_tile_size', [8, 8]))

        # ROIパラメータ
        roi_config = config.get('roi', {})
        self.roi_top_ratio = roi_config.get('top_ratio', 0.4)
        self.roi_bottom_ratio = roi_config.get('bottom_ratio', 0.9)

        # Houghパラメータ
        hough = config.get('hough', {})
        self.hough_rho = hough.get('rho', 1)
        self.hough_theta = np.pi / 180 * hough.get('theta_degrees', 1)
        self.hough_threshold = hough.get('threshold', 50)
        self.hough_min_line_length = hough.get('min_line_length', 50)
        self.hough_max_line_gap = hough.get('max_line_gap', 10)

    def detect(self, image: np.ndarray) -> Dict:
        """
        ライン検出を実行

        Args:
            image: 入力画像 (BGR)

        Returns:
            Dict with keys:
                - white_mask: 白線マスク (H, W) uint8
                - yellow_mask: 黄線マスク (H, W) uint8
                - combined_mask: 統合マスク (H, W) uint8
                - lines: List of line segments [(x1,y1,x2,y2), ...]
                - target_point: (x, y) or None
                - steering_offset: float (-1.0 to 1.0)
                - confidence: float (0.0 to 1.0)
        """
        h, w = image.shape[:2]

        # 1. 前処理
        processed = self.preprocess(image)

        # 2. HSVに変換
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        # 3. 白線・黄線検出
        white_mask = self.detect_white_line(hsv)
        yellow_mask = self.detect_yellow_line(hsv)

        # 4. 統合マスク
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 5. ROI適用
        combined_mask_roi = self.apply_roi(combined_mask)

        # 6. Hough変換で直線検出
        lines = self.detect_lines_hough(combined_mask_roi)

        # 7. 走行目標点を算出
        target_point, steering_offset, confidence = self.calculate_target_point(
            combined_mask_roi, lines, w, h
        )

        return {
            "white_mask": white_mask,
            "yellow_mask": yellow_mask,
            "combined_mask": combined_mask,
            "lines": lines,
            "target_point": target_point,
            "steering_offset": steering_offset,
            "confidence": confidence
        }

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        前処理: ブラー、CLAHE

        Args:
            image: 入力画像 (BGR)

        Returns:
            processed: 前処理済み画像 (BGR)
        """
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # LAB色空間で明度チャンネルに適用
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge([l_clahe, a, b])
        processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return processed

    def detect_white_line(self, hsv: np.ndarray) -> np.ndarray:
        """
        白線を検出

        Args:
            hsv: HSV画像

        Returns:
            mask: 白線マスク (H, W) uint8
        """
        white_config = self.config.get('white_line', {})
        h_range = white_config.get('h_range', [0, 180])
        s_range = white_config.get('s_range', [0, 30])
        v_range = white_config.get('v_range', [200, 255])

        lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
        upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        return mask

    def detect_yellow_line(self, hsv: np.ndarray) -> np.ndarray:
        """
        黄線を検出

        Args:
            hsv: HSV画像

        Returns:
            mask: 黄線マスク (H, W) uint8
        """
        yellow_config = self.config.get('yellow_line', {})
        h_range = yellow_config.get('h_range', [20, 40])
        s_range = yellow_config.get('s_range', [100, 255])
        v_range = yellow_config.get('v_range', [100, 255])

        lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
        upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        return mask

    def apply_roi(self, mask: np.ndarray) -> np.ndarray:
        """
        ROIを適用（ROI外を黒にする）

        Args:
            mask: 入力マスク

        Returns:
            roi_mask: ROI適用後のマスク
        """
        h, w = mask.shape[:2]

        # ROI領域を作成
        roi_mask = np.zeros_like(mask)

        y_top = int(h * self.roi_top_ratio)
        y_bottom = int(h * self.roi_bottom_ratio)

        roi_mask[y_top:y_bottom, :] = mask[y_top:y_bottom, :]

        return roi_mask

    def detect_lines_hough(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Hough変換で直線検出

        Args:
            mask: 入力マスク

        Returns:
            lines: List of (x1, y1, x2, y2)
        """
        lines_hough = cv2.HoughLinesP(
            mask,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        if lines_hough is None:
            return []

        # (x1, y1, x2, y2) のリストに変換
        lines = []
        for line in lines_hough:
            x1, y1, x2, y2 = line[0]
            lines.append((x1, y1, x2, y2))

        return lines

    def calculate_target_point(
        self,
        mask: np.ndarray,
        lines: List[Tuple[int, int, int, int]],
        image_width: int,
        image_height: int
    ) -> Tuple[Optional[Tuple[int, int]], float, float]:
        """
        走行目標点を算出

        Args:
            mask: ラインマスク
            lines: 検出された直線のリスト
            image_width: 画像幅
            image_height: 画像高さ

        Returns:
            Tuple:
                - target_point: (x, y) or None
                - steering_offset: -1.0 (左) to 1.0 (右)
                - confidence: 0.0 to 1.0
        """
        # 方法1: マスクの重心を使用（シンプル）
        moments = cv2.moments(mask)

        if moments['m00'] == 0:
            # ラインが検出されなかった
            return None, 0.0, 0.0

        # 重心を計算
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # ステアリングオフセットを計算
        # 画像中央からの偏差を -1.0 〜 1.0 に正規化
        center_x = image_width / 2
        steering_offset = (cx - center_x) / (image_width / 2)
        steering_offset = np.clip(steering_offset, -1.0, 1.0)

        # 信頼度を計算（ライン面積比）
        line_area = np.count_nonzero(mask)
        total_area = mask.shape[0] * mask.shape[1]
        confidence = min(line_area / total_area * 10, 1.0)  # 10%で信頼度1.0

        return (cx, cy), steering_offset, confidence


class ColorCalibrator:
    """
    環境別色キャリブレーション

    ユーザーが画像上で対象物（白線、黄線、床）を選択し、
    その領域からHSV閾値を自動算出する。

    使用例:
        calibrator = ColorCalibrator()

        # ユーザーが選択した領域から閾値を算出
        white_range = calibrator.calculate_hsv_range(image, roi, margin_sigma=2.0)

        # 設定を保存
        calibrator.save_environment("室内_蛍光灯", {...})
    """

    def __init__(self):
        pass

    def calculate_hsv_range(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int],
        margin_sigma: float = 2.0
    ) -> Dict[str, Tuple[int, int]]:
        """
        選択領域からHSV範囲を算出

        Args:
            image: 入力画像 (BGR)
            roi: 選択領域 (x1, y1, x2, y2)
            margin_sigma: 標準偏差の何倍をマージンとするか

        Returns:
            Dict with keys:
                - h_range: (min, max)
                - s_range: (min, max)
                - v_range: (min, max)
        """
        x1, y1, x2, y2 = roi

        # ROI内を切り出し
        roi_bgr = image[y1:y2, x1:x2]

        # HSVに変換
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # 各チャンネルの平均・標準偏差を計算
        h, s, v = cv2.split(roi_hsv)

        h_mean, h_std = np.mean(h), np.std(h)
        s_mean, s_std = np.mean(s), np.std(s)
        v_mean, v_std = np.mean(v), np.std(v)

        # mean ± margin_sigma * std で範囲を決定
        h_min = max(0, int(h_mean - margin_sigma * h_std))
        h_max = min(179, int(h_mean + margin_sigma * h_std))

        s_min = max(0, int(s_mean - margin_sigma * s_std))
        s_max = min(255, int(s_mean + margin_sigma * s_std))

        v_min = max(0, int(v_mean - margin_sigma * v_std))
        v_max = min(255, int(v_mean + margin_sigma * v_std))

        return {
            'h_range': [h_min, h_max],
            's_range': [s_min, s_max],
            'v_range': [v_min, v_max]
        }

    def preview_detection(
        self,
        image: np.ndarray,
        hsv_range: Dict[str, Tuple[int, int]],
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        検出結果をプレビュー

        Args:
            image: 入力画像 (BGR)
            hsv_range: HSV範囲
            alpha: オーバーレイ透明度

        Returns:
            preview: マスクをオーバーレイした画像
        """
        # HSVに変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV範囲でマスク生成
        h_range = hsv_range['h_range']
        s_range = hsv_range['s_range']
        v_range = hsv_range['v_range']

        lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
        upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # カラーオーバーレイを作成（緑）
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]

        # ブレンド
        preview = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        return preview

    def save_environment(
        self,
        name: str,
        config: Dict,
        output_dir: str = "config/environments"
    ) -> str:
        """
        環境設定を保存

        Args:
            name: 環境名（ファイル名に使用）
            config: 設定内容
            output_dir: 出力ディレクトリ

        Returns:
            saved_path: 保存したファイルパス
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ファイル名を安全な形式に変換
        safe_name = name.replace(' ', '_').replace('/', '_')
        filename = f"{safe_name}.yaml"
        filepath = output_dir / filename

        # タイムスタンプを追加
        config['environment_name'] = name
        config['created_at'] = datetime.now().isoformat()

        # YAML保存
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ Environment saved to: {filepath}")
        return str(filepath)

    def load_environment(self, path: str) -> Dict:
        """
        環境設定を読み込み

        Args:
            path: 設定ファイルパス

        Returns:
            config: 設定内容
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Environment file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✓ Environment loaded from: {path}")
        return config

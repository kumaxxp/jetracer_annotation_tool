"""
深度推定モジュール

Depth Anything V2 Small を使用した単眼深度推定。
ONNX形式のモデルを使用し、ONNX Runtime または OpenCV DNNバックエンドで推論。
"""

from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import time
import cv2
import numpy as np

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


class DepthEstimator:
    """
    Depth Anything V2 Small による深度推定

    使用例:
        estimator = DepthEstimator(
            model_path="output/models/depth_anything_v2_small.onnx",
            input_size=(518, 518),
            use_cuda=True
        )

        depth_map, inference_time = estimator.inference(image)
        # depth_map: (H, W) float32, 値が大きいほど近い
    """

    # ImageNet正規化パラメータ
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (518, 518),
        use_cuda: bool = True
    ):
        """
        Args:
            model_path: ONNXモデルのパス
            input_size: 入力サイズ (width, height)
            use_cuda: CUDA使用フラグ
        """
        self.model_path = Path(model_path)
        self.input_width, self.input_height = input_size
        self.use_cuda = use_cuda
        self.session: Optional[Any] = None
        self.input_name: Optional[str] = None
        self.backend: str = "none"

    def load_model(self) -> bool:
        """
        モデルを読み込む

        Returns:
            bool: 読み込み成功した場合True
        """
        if not self.model_path.exists():
            print(f"Error: Model file not found: {self.model_path}")
            return False

        # Try ONNX Runtime first (better compatibility)
        if HAS_ONNXRUNTIME:
            try:
                return self._load_with_onnxruntime()
            except Exception as e:
                print(f"ONNX Runtime failed: {e}")
                print("Trying OpenCV DNN backend...")

        # Fallback to OpenCV DNN
        try:
            return self._load_with_opencv()
        except Exception as e:
            print(f"OpenCV DNN failed: {e}")
            return False

    def _load_with_onnxruntime(self) -> bool:
        """ONNX Runtimeでモデルを読み込む"""
        print(f"Loading ONNX model with ONNX Runtime from {self.model_path}...")

        # Configure providers
        providers = []
        if self.use_cuda:
            # Try CUDA provider first
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                print("✓ Using CUDA Execution Provider")
            elif 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
                print("✓ Using TensorRT Execution Provider")

        providers.append('CPUExecutionProvider')

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )

        # Get input name
        self.input_name = self.session.get_inputs()[0].name
        self.backend = "onnxruntime"

        print(f"✓ Model loaded successfully (ONNX Runtime)")
        print(f"  Input name: {self.input_name}")
        print(f"  Providers: {self.session.get_providers()}")
        return True

    def _load_with_opencv(self) -> bool:
        """OpenCV DNNでモデルを読み込む"""
        print(f"Loading ONNX model with OpenCV DNN from {self.model_path}...")

        self.session = cv2.dnn.readNetFromONNX(str(self.model_path))

        # CUDA設定
        if self.use_cuda:
            try:
                self.session.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.session.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                print("✓ Using CUDA backend (FP16)")
            except Exception as e:
                print(f"CUDA not available: {e}")
                print("Using CPU backend")
                self.use_cuda = False
        else:
            print("Using CPU backend")

        self.backend = "opencv"
        print(f"✓ Model loaded successfully (OpenCV DNN)")
        return True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        前処理: リサイズ、正規化、blob変換

        Args:
            image: 入力画像 (BGR, HWC)

        Returns:
            blob: (1, 3, H, W) float32
        """
        if image is None or image.ndim != 3:
            raise ValueError("image must be a BGR numpy array with shape (H, W, 3)")

        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # リサイズ
        resized = cv2.resize(
            rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR
        )

        # 0-1 正規化
        normalized = resized.astype(np.float32) / 255.0

        # ImageNet正規化 (mean, std)
        normalized = (normalized - self.MEAN) / self.STD

        # Transpose to (3, H, W) and add batch dimension
        blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        return blob.astype(np.float32)

    def inference(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        深度推定を実行

        Args:
            image: 入力画像 (BGR, HWC)

        Returns:
            Tuple:
                - depth_map: (H, W) float32, 相対深度（大きいほど近い）
                - inference_time_ms: 推論時間（ミリ秒）
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        original_h, original_w = image.shape[:2]

        # 前処理
        blob = self.preprocess(image)

        # 推論
        start = time.perf_counter()

        if self.backend == "onnxruntime":
            outputs = self.session.run(None, {self.input_name: blob})
            output = outputs[0]
        else:  # opencv
            self.session.setInput(blob)
            output = self.session.forward()

        inference_time_ms = (time.perf_counter() - start) * 1000

        # 後処理
        depth_map = self.postprocess(output, (original_w, original_h))

        return depth_map, inference_time_ms

    def postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        後処理: 元サイズにリサイズ、正規化

        Args:
            output: モデル出力
            original_size: 元画像サイズ (width, height)

        Returns:
            depth_map: (H, W) float32, 0-1正規化済み
        """
        # モデル出力の形状を確認: (1, 1, H, W) または (1, H, W) を想定
        if output.ndim == 4:
            depth = output[0, 0]  # (H, W)
        elif output.ndim == 3:
            depth = output[0]  # (H, W)
        else:
            depth = output

        # 元サイズにリサイズ
        width, height = original_size
        depth_resized = cv2.resize(
            depth,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # 0-1正規化（大きいほど近いに統一）
        depth_min = depth_resized.min()
        depth_max = depth_resized.max()

        if depth_max > depth_min:
            # 正規化して反転（Depth Anythingは小さいほど近い場合があるため）
            # モデルによって調整が必要
            depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_resized)

        return depth_normalized.astype(np.float32)


class DepthCalibrator:
    """
    車体基準による深度キャリブレーション

    使用例:
        calibrator = DepthCalibrator(
            reference_roi=(240, 420, 400, 480),
            reference_distance_cm=5.0
        )

        # キャリブレーション
        calibrator.calibrate(depth_map)

        # 実距離変換
        distance = calibrator.to_real_distance(depth_value)
    """

    def __init__(
        self,
        reference_roi: Tuple[int, int, int, int],
        reference_distance_cm: float
    ):
        """
        Args:
            reference_roi: 車体基準領域 (x1, y1, x2, y2)
            reference_distance_cm: 車体までの実距離 (cm)
        """
        self.reference_roi = reference_roi
        self.reference_distance_cm = reference_distance_cm
        self.reference_depth: Optional[float] = None
        self.is_calibrated: bool = False

    def calibrate(self, depth_map: np.ndarray) -> float:
        """
        キャリブレーションを実行

        Args:
            depth_map: 深度マップ (H, W)

        Returns:
            reference_depth: 基準深度値
        """
        x1, y1, x2, y2 = self.reference_roi

        # ROI内の深度値を取得
        roi_depth = depth_map[y1:y2, x1:x2]

        # 中央値を基準深度として記録
        self.reference_depth = float(np.median(roi_depth))
        self.is_calibrated = True

        print(f"✓ Calibration completed:")
        print(f"  Reference ROI: {self.reference_roi}")
        print(f"  Reference depth value: {self.reference_depth:.4f}")
        print(f"  Reference distance: {self.reference_distance_cm} cm")

        return self.reference_depth

    def to_real_distance(self, depth_value: float) -> float:
        """
        深度値を実距離に変換

        Args:
            depth_value: 深度値

        Returns:
            distance_cm: 推定距離 (cm)

        Raises:
            RuntimeError: キャリブレーション未実施の場合
        """
        if not self.is_calibrated:
            raise RuntimeError("Calibration not performed. Call calibrate() first.")

        if self.reference_depth == 0:
            return float('inf')

        # 深度値は大きいほど近い想定
        # distance ∝ 1 / depth
        distance_cm = self.reference_distance_cm * (self.reference_depth / depth_value)

        return float(distance_cm)


class ObstacleAnalyzer:
    """
    深度マップから障害物を分析

    使用例:
        analyzer = ObstacleAnalyzer(calibrator, config)
        result = analyzer.analyze(depth_map)
        # result = {
        #     "regions": {...},
        #     "min_distance_cm": 25.0,
        #     "warning_level": "SLOW",
        #     "recommended_direction": "GO"
        # }
    """

    def __init__(
        self,
        calibrator: DepthCalibrator,
        config: Dict
    ):
        """
        Args:
            calibrator: 深度キャリブレーター
            config: 設定（閾値など）
        """
        self.calibrator = calibrator
        self.config = config

        # 閾値を取得
        self.emergency_stop_cm = config.get('emergency_stop_cm', 10.0)
        self.stop_cm = config.get('stop_cm', 15.0)
        self.slow_cm = config.get('slow_cm', 30.0)

    def analyze(self, depth_map: np.ndarray) -> Dict:
        """
        障害物分析を実行

        Args:
            depth_map: 深度マップ (H, W)

        Returns:
            Dict with keys:
                - regions: Dict[str, float] - 各領域の距離
                - min_distance_cm: float - 最小距離
                - warning_level: str - "SAFE", "SLOW", "STOP", "EMERGENCY"
                - recommended_direction: str - "GO", "SLOW", "STOP", "TURN_LEFT", "TURN_RIGHT"
        """
        # 画面を3x3に分割
        regions_depth = self._divide_into_regions(depth_map)

        # 各領域の実距離に変換
        regions_distance = {}
        for region_name, region_depth_value in regions_depth.items():
            try:
                distance = self.calibrator.to_real_distance(region_depth_value)
                regions_distance[region_name] = distance
            except Exception as e:
                print(f"Warning: Failed to convert depth for {region_name}: {e}")
                regions_distance[region_name] = float('inf')

        # 最小距離を取得
        min_distance_cm = min(regions_distance.values())

        # 警告レベルを判定
        if min_distance_cm < self.emergency_stop_cm:
            warning_level = "EMERGENCY"
        elif min_distance_cm < self.stop_cm:
            warning_level = "STOP"
        elif min_distance_cm < self.slow_cm:
            warning_level = "SLOW"
        else:
            warning_level = "SAFE"

        # 推奨方向を決定
        recommended_direction = self._recommend_direction(regions_distance, warning_level)

        return {
            "regions": regions_distance,
            "min_distance_cm": min_distance_cm,
            "warning_level": warning_level,
            "recommended_direction": recommended_direction
        }

    def _divide_into_regions(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        深度マップを領域に分割

        Returns:
            Dict with keys: "top_left", "top_center", "top_right",
                           "mid_left", "mid_center", "mid_right",
                           "bot_left", "bot_center", "bot_right"
        """
        h, w = depth_map.shape

        # 下部1/8を車体基準として除外
        exclude_ratio = 0.125
        effective_h = int(h * (1 - exclude_ratio))

        # 3x3に分割
        row_step = effective_h // 3
        col_step = w // 3

        regions = {}
        region_names = [
            ["top_left", "top_center", "top_right"],
            ["mid_left", "mid_center", "mid_right"],
            ["bot_left", "bot_center", "bot_right"]
        ]

        for row in range(3):
            for col in range(3):
                y1 = row * row_step
                y2 = (row + 1) * row_step if row < 2 else effective_h
                x1 = col * col_step
                x2 = (col + 1) * col_step if col < 2 else w

                region_data = depth_map[y1:y2, x1:x2]
                # 中央値を使用
                region_value = float(np.median(region_data))

                regions[region_names[row][col]] = region_value

        return regions

    def _recommend_direction(
        self,
        regions: Dict[str, float],
        warning_level: str
    ) -> str:
        """
        推奨方向を決定

        Args:
            regions: 各領域の距離
            warning_level: 警告レベル

        Returns:
            推奨方向: "GO", "SLOW", "STOP", "TURN_LEFT", "TURN_RIGHT"
        """
        if warning_level == "EMERGENCY":
            return "STOP"

        if warning_level == "STOP":
            # 中央が塞がれている場合、左右を確認
            center_distance = regions.get("mid_center", float('inf'))
            left_distance = regions.get("mid_left", float('inf'))
            right_distance = regions.get("mid_right", float('inf'))

            if left_distance > right_distance and left_distance > self.slow_cm:
                return "TURN_LEFT"
            elif right_distance > left_distance and right_distance > self.slow_cm:
                return "TURN_RIGHT"
            else:
                return "STOP"

        if warning_level == "SLOW":
            return "SLOW"

        # SAFE
        return "GO"

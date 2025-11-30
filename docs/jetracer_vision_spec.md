# JetRacer Vision System 仕様書

## 1. プロジェクト概要

### 1.1 目的

JetRacerに2カメラシステムを導入し、以下の機能を実現する：

1. **足元カメラ**: 走行可能領域のセグメンテーション（既存機能の活用）
2. **正面カメラ**: 
   - Mode A: 深度推定による障害物回避
   - Mode B: ライン検出によるライントレース

### 1.2 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                        JetRacer 車両                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐              ┌──────────────┐                │
│  │ 足元カメラ    │              │ 正面カメラ    │                │
│  │ (CSI)        │              │ (CSI)        │                │
│  └──────┬───────┘              └──────┬───────┘                │
│         │                             │                        │
│         ▼                             ▼                        │
│  ┌──────────────┐              ┌──────────────────────┐        │
│  │セグメンテーション│              │ Mode A: 深度推定      │        │
│  │ (既存ONNX)   │              │ Mode B: ライン検出    │        │
│  └──────┬───────┘              └──────┬───────────────┘        │
│         │                             │                        │
│         ▼                             ▼                        │
│  ┌──────────────┐              ┌──────────────┐                │
│  │ 走行可能領域  │              │ 障害物距離 /  │                │
│  │   マスク     │              │ ライン位置    │                │
│  └──────┬───────┘              └──────┬───────┘                │
│         │                             │                        │
│         └──────────┬──────────────────┘                        │
│                    ▼                                           │
│            ┌──────────────┐                                    │
│            │  走行判定     │                                    │
│            │ (統合ロジック) │                                    │
│            └──────┬───────┘                                    │
│                   │                                            │
│                   ▼                                            │
│            ┌──────────────┐                                    │
│            │  走行制御     │                                    │
│            │ (JetRacer)   │                                    │
│            └──────────────┘                                    │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 開発フェーズ

| Phase | 内容 | 成果物 |
|-------|------|--------|
| Phase 1 | オフライン検証環境構築 | JPEG読み込み、可視化UI |
| Phase 2 | 深度推定 (Mode A) | DA2 Small導入、キャリブレーション |
| Phase 3 | ライン検出 (Mode B) | 白線・黄線検出、環境キャリブレーション |
| Phase 4 | 統合判定 | 2カメラ統合、走行コマンド生成 |
| Phase 5 | リアルタイム化 | カメラ入力、JetRacer制御接続 |

**重要**: Phase 1-4 は録画済みJPEGファイルを使用したオフライン検証。Phase 5 でリアルタイム化。

---

## 2. ディレクトリ構成

既存の `jetracer_annotation_tool/` プロジェクトに新規ファイルを追加する。

```
jetracer_annotation_tool/
├── core/                              # コア機能
│   ├── ade20k_segmentation.py         # 既存
│   ├── mapping.py                     # 既存
│   ├── model_trainer.py               # 既存
│   ├── onnx_inference.py              # 既存（活用）
│   ├── pytorch_inference.py           # 既存
│   ├── segmentation.py                # 既存
│   ├── training_export.py             # 既存
│   ├── vehicle_mask_generator.py      # 既存
│   │
│   ├── camera_manager.py              # NEW: 2カメラ管理
│   ├── depth_estimation.py            # NEW: 深度推定
│   ├── line_detection.py              # NEW: ライン検出
│   ├── obstacle_analyzer.py           # NEW: 障害物分析
│   ├── driving_decision.py            # NEW: 走行判定統合
│   ├── calibration.py                 # NEW: 環境キャリブレーション
│   └── image_loader.py                # NEW: 画像読み込みユーティリティ
│
├── ui/                                # NiceGUI UI
│   ├── annotation_ui.py               # 既存
│   ├── training_ui.py                 # 既存
│   │
│   ├── calibration_ui.py              # NEW: 環境キャリブレーション
│   ├── offline_viewer_ui.py           # NEW: オフライン検証ビューア
│   ├── depth_test_ui.py               # NEW: 深度推定テスト
│   ├── line_test_ui.py                # NEW: ライン検出テスト
│   └── realtime_ui.py                 # NEW: リアルタイム (Phase 5)
│
├── config/                            # NEW: 設定ファイル
│   ├── default.yaml                   # デフォルト設定
│   └── environments/                  # 環境別設定
│       └── sample.yaml                # サンプル環境設定
│
├── data/                              # 既存
│   └── ade20k_labels.py               # 既存
│
├── output/                            # 既存 + 拡張
│   ├── road_mapping.json              # 既存
│   ├── vehicle_mask.png               # 既存
│   ├── models/                        # モデル格納
│   │   ├── best_model.pth             # 既存
│   │   ├── road_segmentation.onnx     # 既存
│   │   └── depth_anything_v2_small.onnx  # NEW
│   └── recordings/                    # NEW: 録画データ
│       └── YYYYMMDD_HHMMSS/
│           ├── metadata.yaml
│           ├── front/
│           │   ├── 000001.jpg
│           │   └── ...
│           └── ground/
│               ├── 000001.jpg
│               └── ...
│
├── demo_images/                       # 既存（オフライン検証に使用）
│   ├── front/                         # NEW: 正面カメラ画像
│   └── ground/                        # NEW: 足元カメラ画像（または既存画像）
│
├── scripts/                           # NEW: ユーティリティスクリプト
│   └── convert_depth_model.py         # DA2 ONNX変換
│
├── main.py                            # 既存
├── main_training.py                   # 既存
├── main_calibration.py                # NEW: キャリブレーション起動
├── main_offline_test.py               # NEW: オフライン検証起動
└── main_realtime.py                   # NEW: リアルタイム起動 (Phase 5)
```

---

## 3. 設定ファイル仕様

### 3.1 default.yaml

```yaml
# config/default.yaml
# JetRacer Vision System デフォルト設定

# =============================================================================
# カメラ設定 (Phase 5 で使用)
# =============================================================================
cameras:
  ground:
    device_id: 0
    resolution: [640, 480]
    fps: 15
    rotation: 0  # 0, 90, 180, 270
  front:
    device_id: 1
    resolution: [640, 480]
    fps: 15
    rotation: 0

# =============================================================================
# 処理設定
# =============================================================================
processing:
  # 足元カメラ: セグメンテーション (既存モデル使用)
  segmentation:
    model_path: "output/models/road_segmentation.onnx"
    input_size: [320, 240]  # width, height
    use_cuda: true
    classes:
      OTHER: 0
      ROAD: 1
      MYCAR: 2

  # 正面カメラ Mode A: 深度推定
  depth:
    enabled: true
    model_path: "output/models/depth_anything_v2_small.onnx"
    input_size: [320, 240]
    use_cuda: true
    
    # 車体基準キャリブレーション
    vehicle_reference:
      roi: [240, 420, 400, 480]  # x1, y1, x2, y2 (640x480想定)
      distance_cm: 5.0           # 車体までの実距離
    
    # ROI分割 (3x3 + 車体基準)
    roi_grid:
      rows: 3
      cols: 3
      exclude_bottom_ratio: 0.125  # 下部1/8を車体基準として除外

  # 正面カメラ Mode B: ライン検出
  line_detection:
    enabled: true
    
    # 前処理
    preprocessing:
      blur_kernel: 5
      clahe_clip_limit: 2.0
      clahe_tile_size: [8, 8]
    
    # 白線検出 (HSV)
    white_line:
      h_range: [0, 180]
      s_range: [0, 30]
      v_range: [200, 255]
    
    # 黄線検出 (HSV)
    yellow_line:
      h_range: [20, 40]
      s_range: [100, 255]
      v_range: [100, 255]
    
    # ROI (画面の40%〜90%の高さを使用)
    roi:
      top_ratio: 0.4
      bottom_ratio: 0.9
    
    # 直線検出パラメータ
    hough:
      rho: 1
      theta_degrees: 1
      threshold: 50
      min_line_length: 50
      max_line_gap: 10

# =============================================================================
# 判定設定
# =============================================================================
decision:
  # モード選択: "OBSTACLE_AVOIDANCE" or "LINE_FOLLOWING"
  mode: "OBSTACLE_AVOIDANCE"
  
  # 障害物回避 (Mode A) 閾値
  obstacle_avoidance:
    emergency_stop_cm: 10.0
    stop_cm: 15.0
    slow_cm: 30.0
  
  # 足元セグメンテーション閾値
  ground:
    min_drivable_ratio: 0.3  # 30%以上で走行可能
  
  # ライントレース (Mode B) 設定
  line_following:
    steering_gain: 0.5
    target_point_ratio: 0.7  # 画面上部70%の位置を目標

  # 走行速度
  speed:
    base: 0.2
    slow: 0.1

# =============================================================================
# UI設定
# =============================================================================
ui:
  theme: "dark"
  refresh_interval_ms: 100

# =============================================================================
# ログ設定
# =============================================================================
logging:
  level: "INFO"
  save_to_file: false
  file_path: "output/logs/vision.log"
```

### 3.2 環境別設定 (environments/sample.yaml)

```yaml
# config/environments/sample.yaml
# 特定の環境（場所・照明条件）用の設定

environment_name: "室内_蛍光灯"
created_at: "2025-11-30T12:00:00"

# 環境キャリブレーションで取得した値
calibration:
  # 白線の色範囲
  white_line:
    h_range: [0, 180]
    s_range: [0, 35]
    v_range: [190, 255]
  
  # 黄線の色範囲
  yellow_line:
    h_range: [18, 42]
    s_range: [110, 255]
    v_range: [140, 255]
  
  # 床の色範囲 (除外用)
  floor:
    h_range: [10, 30]
    s_range: [20, 80]
    v_range: [80, 160]
  
  # 深度キャリブレーション
  depth:
    reference_depth_value: 0.85  # 車体基準の深度値
    scale_factor: 1.0            # 距離スケール補正
```

---

## 4. モジュール詳細仕様

### 4.1 core/image_loader.py

JPEGファイルの読み込みとセッション管理。

```python
"""
画像読み込みユーティリティ

録画済みJPEGファイルを読み込み、2カメラの同期データを提供する。
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np
import yaml


class ImageLoader:
    """
    録画データからの画像読み込み
    
    使用例:
        loader = ImageLoader("output/recordings/20251130_120000")
        loader.load_session()
        
        # フレーム取得
        frame = loader.get_frame(0)
        # frame = {"front": np.ndarray, "ground": np.ndarray, "timestamp": ...}
        
        # イテレーション
        for frame in loader:
            process(frame)
    """
    
    def __init__(self, session_path: str):
        """
        Args:
            session_path: 録画セッションのディレクトリパス
                         例: "output/recordings/20251130_120000"
        """
        self.session_path = Path(session_path)
        self.metadata: Optional[Dict] = None
        self.front_images: List[Path] = []
        self.ground_images: List[Path] = []
        self.current_index: int = 0
    
    def load_session(self) -> bool:
        """
        セッションを読み込む
        
        Returns:
            bool: 読み込み成功した場合True
        """
        # TODO: 実装
        # 1. metadata.yaml を読み込み
        # 2. front/, ground/ ディレクトリの画像一覧を取得
        # 3. ファイル名でソート
        pass
    
    def get_frame(self, index: int) -> Optional[Dict[str, any]]:
        """
        指定インデックスのフレームを取得
        
        Args:
            index: フレームインデックス
        
        Returns:
            Dict with keys:
                - "front": np.ndarray (BGR) or None
                - "ground": np.ndarray (BGR) or None
                - "index": int
                - "front_path": str
                - "ground_path": str
        """
        # TODO: 実装
        pass
    
    def get_frame_count(self) -> int:
        """フレーム総数を取得"""
        return max(len(self.front_images), len(self.ground_images))
    
    def __len__(self) -> int:
        return self.get_frame_count()
    
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self) -> Dict[str, any]:
        if self.current_index >= len(self):
            raise StopIteration
        frame = self.get_frame(self.current_index)
        self.current_index += 1
        return frame


class SingleFolderLoader:
    """
    単一フォルダからの画像読み込み（デモ用）
    
    使用例:
        loader = SingleFolderLoader("demo_images/front")
        images = loader.load_all()
    """
    
    def __init__(self, folder_path: str, extensions: List[str] = None):
        """
        Args:
            folder_path: 画像フォルダパス
            extensions: 対象拡張子リスト（デフォルト: [".jpg", ".jpeg", ".png"]）
        """
        self.folder_path = Path(folder_path)
        self.extensions = extensions or [".jpg", ".jpeg", ".png"]
        self.image_paths: List[Path] = []
    
    def load_paths(self) -> List[Path]:
        """画像パスのリストを取得"""
        # TODO: 実装
        pass
    
    def load_image(self, path: Path) -> np.ndarray:
        """画像を読み込み (BGR形式)"""
        return cv2.imread(str(path))
    
    def load_all(self) -> List[Tuple[Path, np.ndarray]]:
        """全画像を読み込み"""
        # TODO: 実装
        pass
```

### 4.2 core/depth_estimation.py

Depth Anything V2 Smallによる深度推定。

```python
"""
深度推定モジュール

Depth Anything V2 Small を使用した単眼深度推定。
ONNX形式のモデルを使用し、OpenCV DNNバックエンドで推論。
"""

from pathlib import Path
from typing import Tuple, Dict, Optional
import cv2
import numpy as np


class DepthEstimator:
    """
    Depth Anything V2 Small による深度推定
    
    使用例:
        estimator = DepthEstimator(
            model_path="output/models/depth_anything_v2_small.onnx",
            input_size=(320, 240),
            use_cuda=True
        )
        
        depth_map = estimator.inference(image)
        # depth_map: (H, W) float32, 値が大きいほど近い
    """
    
    # ImageNet正規化パラメータ
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (320, 240),
        use_cuda: bool = True
    ):
        """
        Args:
            model_path: ONNXモデルのパス
            input_size: 入力サイズ (width, height)
            use_cuda: CUDA使用フラグ
        """
        self.model_path = Path(model_path)
        self.input_size = input_size  # (width, height)
        self.use_cuda = use_cuda
        self.net: Optional[cv2.dnn.Net] = None
        
    def load_model(self) -> bool:
        """
        モデルを読み込む
        
        Returns:
            bool: 読み込み成功した場合True
        """
        # TODO: 実装
        # 1. cv2.dnn.readNetFromONNX() でモデル読み込み
        # 2. use_cuda の場合は setPreferableBackend/Target を設定
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        前処理: リサイズ、正規化、blob変換
        
        Args:
            image: 入力画像 (BGR, HWC)
        
        Returns:
            blob: (1, 3, H, W) float32
        """
        # TODO: 実装
        # 1. BGR → RGB
        # 2. リサイズ
        # 3. 0-1 正規化
        # 4. ImageNet正規化 (mean, std)
        # 5. HWC → CHW
        # 6. バッチ次元追加
        pass
    
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
        # TODO: 実装
        # 1. preprocess
        # 2. net.setInput
        # 3. net.forward
        # 4. 後処理（元サイズにリサイズ）
        # 5. 時間計測
        pass
    
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
        # TODO: 実装
        pass


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
        # TODO: 実装
        # 1. ROI内の深度値を取得
        # 2. 中央値を基準深度として記録
        pass
    
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
        # TODO: 実装
        # 深度値は近いほど大きい想定
        # distance ∝ 1 / depth
        pass


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
        # TODO: 実装
        # 1. 画面を3x3に分割
        # 2. 各領域の深度統計を計算
        # 3. 実距離に変換
        # 4. 閾値判定
        # 5. 推奨方向を決定
        pass
    
    def _divide_into_regions(
        self, 
        depth_map: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        深度マップを領域に分割
        
        Returns:
            Dict with keys: "top_left", "top_center", "top_right",
                           "mid_left", "mid_center", "mid_right",
                           "bot_left", "bot_center", "bot_right"
        """
        # TODO: 実装
        pass
```

### 4.3 core/line_detection.py

白線・黄線検出によるライントレース。

```python
"""
ライン検出モジュール

白線・黄線を検出し、走行目標を算出する。
環境キャリブレーションにより、照明・床色の変動に対応。
"""

from typing import Tuple, Dict, Optional, List
import cv2
import numpy as np


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
        # TODO: 実装
        # 1. 前処理
        # 2. HSVフィルタリング
        # 3. ROI適用
        # 4. Hough変換
        # 5. 走行目標算出
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        前処理: ブラー、CLAHE
        
        Args:
            image: 入力画像 (BGR)
        
        Returns:
            processed: 前処理済み画像 (BGR)
        """
        # TODO: 実装
        # 1. ガウシアンブラー
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        pass
    
    def detect_white_line(self, hsv: np.ndarray) -> np.ndarray:
        """白線を検出"""
        # TODO: 実装
        # HSV閾値でマスク生成
        pass
    
    def detect_yellow_line(self, hsv: np.ndarray) -> np.ndarray:
        """黄線を検出"""
        # TODO: 実装
        pass
    
    def apply_roi(self, mask: np.ndarray) -> np.ndarray:
        """ROIを適用（ROI外を黒にする）"""
        # TODO: 実装
        pass
    
    def detect_lines_hough(
        self, 
        mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Hough変換で直線検出
        
        Returns:
            lines: List of (x1, y1, x2, y2)
        """
        # TODO: 実装
        pass
    
    def calculate_target_point(
        self,
        mask: np.ndarray,
        lines: List[Tuple[int, int, int, int]],
        image_width: int,
        image_height: int
    ) -> Tuple[Optional[Tuple[int, int]], float, float]:
        """
        走行目標点を算出
        
        Returns:
            Tuple:
                - target_point: (x, y) or None
                - steering_offset: -1.0 (左) to 1.0 (右)
                - confidence: 0.0 to 1.0
        """
        # TODO: 実装
        # 方法1: マスクの重心
        # 方法2: 左右の線の中間
        # 方法3: 線の上端（遠方）を追跡
        pass


class ColorCalibrator:
    """
    環境別色キャリブレーション
    
    ユーザーが画像上で対象物（白線、黄線、床）を選択し、
    その領域からHSV閾値を自動算出する。
    
    使用例:
        calibrator = ColorCalibrator()
        
        # ユーザーが選択した領域から閾値を算出
        white_range = calibrator.calculate_hsv_range(image, roi, margin=1.5)
        
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
        # TODO: 実装
        # 1. ROI内をHSVに変換
        # 2. 各チャンネルの平均・標準偏差を計算
        # 3. mean ± margin_sigma * std で範囲を決定
        # 4. 0-255 (S,V) / 0-179 (H) にクリップ
        pass
    
    def preview_detection(
        self,
        image: np.ndarray,
        hsv_range: Dict[str, Tuple[int, int]]
    ) -> np.ndarray:
        """
        検出結果をプレビュー
        
        Args:
            image: 入力画像 (BGR)
            hsv_range: HSV範囲
        
        Returns:
            preview: マスクをオーバーレイした画像
        """
        # TODO: 実装
        pass
    
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
        # TODO: 実装
        pass
    
    def load_environment(self, path: str) -> Dict:
        """環境設定を読み込み"""
        # TODO: 実装
        pass
```

### 4.4 core/driving_decision.py

2カメラ情報の統合判定。

```python
"""
走行判定統合モジュール

足元カメラ（セグメンテーション）と正面カメラ（深度/ライン）の
結果を統合し、走行コマンドを生成する。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


class DrivingCommand(Enum):
    """走行コマンド"""
    GO = "GO"
    SLOW = "SLOW"
    STOP = "STOP"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class ProcessingMode(Enum):
    """正面カメラの処理モード"""
    OBSTACLE_AVOIDANCE = "OBSTACLE_AVOIDANCE"
    LINE_FOLLOWING = "LINE_FOLLOWING"


@dataclass
class DrivingDecision:
    """走行判定結果"""
    command: DrivingCommand
    speed: float           # 0.0 - 1.0
    steering: float        # -1.0 (左) - 1.0 (右)
    reason: str
    confidence: float      # 0.0 - 1.0
    
    # 詳細情報
    ground_safe: bool
    ground_drivable_ratio: float
    front_warning_level: str
    front_min_distance_cm: Optional[float]


class IntegratedDecisionMaker:
    """
    統合走行判定
    
    使用例:
        maker = IntegratedDecisionMaker(config)
        maker.set_mode(ProcessingMode.OBSTACLE_AVOIDANCE)
        
        decision = maker.decide(
            ground_result=segmentation_result,
            front_result=depth_analysis_result
        )
        
        print(decision.command)  # GO, SLOW, STOP, etc.
        print(decision.steering)  # -0.3
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: decision設定セクション
        """
        self.config = config
        self.mode = ProcessingMode.OBSTACLE_AVOIDANCE
    
    def set_mode(self, mode: ProcessingMode):
        """処理モードを設定"""
        self.mode = mode
    
    def decide(
        self,
        ground_result: Optional[Dict],
        front_result: Optional[Dict]
    ) -> DrivingDecision:
        """
        統合判定を実行
        
        Args:
            ground_result: 足元セグメンテーション結果
                - mask: np.ndarray
                - drivable_ratio: float
                - balance: float (-1 to 1, 正なら左に余裕)
            front_result: 正面カメラ結果
                Mode A (障害物回避):
                    - min_distance_cm: float
                    - warning_level: str
                    - recommended_direction: str
                Mode B (ライントレース):
                    - target_point: (x, y)
                    - steering_offset: float
                    - confidence: float
        
        Returns:
            DrivingDecision
        """
        # TODO: 実装
        # 優先度順:
        # 1. 正面の緊急停止判定
        # 2. 足元の走行可能判定
        # 3. 正面の障害物回避 or ライン追従
        # 4. 通常走行
        pass
    
    def _decide_obstacle_avoidance(
        self,
        ground_result: Dict,
        front_result: Dict
    ) -> DrivingDecision:
        """Mode A: 障害物回避の判定"""
        # TODO: 実装
        pass
    
    def _decide_line_following(
        self,
        ground_result: Dict,
        front_result: Dict
    ) -> DrivingDecision:
        """Mode B: ライントレースの判定"""
        # TODO: 実装
        pass
    
    def _analyze_ground(self, ground_result: Dict) -> Dict:
        """足元セグメンテーション結果を分析"""
        # TODO: 実装
        # - drivable_ratio の計算
        # - 左右バランスの計算
        pass
```

### 4.5 core/calibration.py

環境キャリブレーションの統合管理。

```python
"""
環境キャリブレーション管理

深度キャリブレーションと色キャリブレーションを統合管理する。
"""

from pathlib import Path
from typing import Dict, Optional
import yaml


class EnvironmentManager:
    """
    環境設定の管理
    
    使用例:
        manager = EnvironmentManager("config")
        
        # 現在の環境を読み込み
        env = manager.load_current()
        
        # 環境一覧
        envs = manager.list_environments()
        
        # 環境を切り替え
        manager.set_current("室内_蛍光灯")
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Args:
            config_dir: 設定ディレクトリパス
        """
        self.config_dir = Path(config_dir)
        self.environments_dir = self.config_dir / "environments"
        self.current_file = self.config_dir / "current_environment.yaml"
    
    def load_default(self) -> Dict:
        """デフォルト設定を読み込み"""
        # TODO: 実装
        pass
    
    def load_current(self) -> Dict:
        """現在の環境設定を読み込み（デフォルトとマージ）"""
        # TODO: 実装
        pass
    
    def list_environments(self) -> list[str]:
        """利用可能な環境一覧を取得"""
        # TODO: 実装
        pass
    
    def set_current(self, name: str) -> bool:
        """現在の環境を設定"""
        # TODO: 実装
        pass
    
    def save_environment(self, name: str, config: Dict) -> str:
        """環境設定を保存"""
        # TODO: 実装
        pass
    
    def delete_environment(self, name: str) -> bool:
        """環境設定を削除"""
        # TODO: 実装
        pass
```

---

## 5. UI仕様

### 5.1 ui/calibration_ui.py

環境キャリブレーションUI。

```python
"""
環境キャリブレーションUI

NiceGUIを使用した環境キャリブレーションツール。
ユーザーが画像上で対象物を選択し、色閾値を自動算出する。
"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Tuple
import cv2
import numpy as np
import base64


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
        
        # 選択中の対象
        self.selection_mode: str = "white_line"  # white_line, yellow_line, floor, vehicle_ref
        self.selections: Dict[str, Dict] = {}
        
        # UI要素の参照
        self.image_display = None
        self.preview_display = None
        self.result_panel = None
    
    def create_ui(self):
        """UIを構築"""
        
        with ui.header():
            ui.label("JetRacer 環境キャリブレーション").classes("text-h5")
        
        with ui.row().classes("w-full"):
            # 左パネル: 画像選択・表示
            with ui.card().classes("w-1/2"):
                ui.label("1. 画像を選択").classes("text-h6")
                
                # 画像選択
                # TODO: ファイル選択UI
                
                # 画像表示（クリック可能）
                # TODO: ui.interactive_image() で範囲選択可能に
                pass
            
            # 右パネル: 対象選択・設定
            with ui.card().classes("w-1/2"):
                ui.label("2. 対象を選択").classes("text-h6")
                
                # 対象選択ラジオボタン
                # TODO: white_line, yellow_line, floor, vehicle_ref
                
                # 選択済み一覧
                # TODO: 各対象のHSV範囲を表示
                
                # プレビューボタン
                # TODO: 現在の設定で検出した結果を表示
                pass
        
        with ui.row().classes("w-full"):
            # 下パネル: プレビュー・保存
            with ui.card().classes("w-full"):
                ui.label("3. プレビュー・保存").classes("text-h6")
                
                # 検出結果プレビュー
                # TODO: マスクをオーバーレイ表示
                
                # 環境名入力
                # TODO: テキスト入力
                
                # 保存ボタン
                # TODO: YAML保存
                pass
    
    def on_image_click(self, x: int, y: int):
        """画像クリック時のハンドラ（範囲選択開始）"""
        # TODO: 実装
        pass
    
    def on_selection_complete(self, roi: Tuple[int, int, int, int]):
        """範囲選択完了時のハンドラ"""
        # TODO: 実装
        # 1. HSV範囲を計算
        # 2. 結果を表示
        # 3. selections に保存
        pass
    
    def update_preview(self):
        """検出結果プレビューを更新"""
        # TODO: 実装
        pass
    
    def save_environment(self, name: str):
        """環境設定を保存"""
        # TODO: 実装
        pass


def main():
    """メインエントリポイント"""
    calibration_ui = CalibrationUI()
    calibration_ui.create_ui()
    ui.run(port=8083, title="JetRacer Calibration")


if __name__ in {"__main__", "__mp_main__"}:
    main()
```

### 5.2 ui/offline_viewer_ui.py

オフライン検証ビューア。

```python
"""
オフライン検証ビューアUI

録画済みJPEGファイルに対して処理を実行し、結果を確認する。
"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np


class OfflineViewerUI:
    """
    オフライン検証ビューア
    
    機能:
        1. 録画セッション選択
        2. フレーム単位での処理・表示
        3. 足元/正面カメラの同時表示
        4. 処理結果（セグメンテーション、深度、ライン）のオーバーレイ
        5. モード切り替え（障害物回避/ライントレース）
        6. パラメータ調整
    """
    
    def __init__(self):
        self.session_path: Optional[Path] = None
        self.current_frame_index: int = 0
        self.mode: str = "OBSTACLE_AVOIDANCE"
        
        # 処理モジュール
        self.segmenter = None  # 既存のONNXSegmenter
        self.depth_estimator = None
        self.line_detector = None
        self.decision_maker = None
        
        # 表示用
        self.ground_display = None
        self.front_display = None
        self.result_panel = None
    
    def create_ui(self):
        """UIを構築"""
        
        with ui.header():
            ui.label("JetRacer オフライン検証").classes("text-h5")
            
            # モード切り替え
            with ui.row():
                ui.label("Mode:")
                ui.toggle(
                    ["障害物回避", "ライントレース"],
                    value="障害物回避",
                    on_change=self.on_mode_change
                )
        
        with ui.row().classes("w-full"):
            # 左: 足元カメラ
            with ui.card().classes("w-1/2"):
                ui.label("足元カメラ (セグメンテーション)").classes("text-h6")
                # TODO: 画像表示
                # TODO: セグメンテーション結果オーバーレイ
                pass
            
            # 右: 正面カメラ
            with ui.card().classes("w-1/2"):
                ui.label("正面カメラ").classes("text-h6")
                # TODO: 画像表示
                # TODO: 深度/ライン結果オーバーレイ
                pass
        
        with ui.row().classes("w-full"):
            # フレーム操作
            with ui.card().classes("w-1/2"):
                ui.label("フレーム操作").classes("text-h6")
                with ui.row():
                    ui.button("◀ 前", on_click=self.prev_frame)
                    ui.button("次 ▶", on_click=self.next_frame)
                    # TODO: スライダー
                    # TODO: 再生/停止
                pass
            
            # 判定結果
            with ui.card().classes("w-1/2"):
                ui.label("走行判定").classes("text-h6")
                # TODO: DrivingDecision の表示
                # - コマンド
                # - 速度
                # - ステアリング
                # - 理由
                pass
    
    def load_session(self, path: str):
        """セッションを読み込み"""
        # TODO: 実装
        pass
    
    def process_frame(self, index: int) -> Dict:
        """指定フレームを処理"""
        # TODO: 実装
        # 1. 画像読み込み
        # 2. 足元: セグメンテーション
        # 3. 正面: 深度 or ライン検出
        # 4. 統合判定
        # 5. 結果を返す
        pass
    
    def update_display(self):
        """表示を更新"""
        # TODO: 実装
        pass
    
    def prev_frame(self):
        """前のフレームへ"""
        # TODO: 実装
        pass
    
    def next_frame(self):
        """次のフレームへ"""
        # TODO: 実装
        pass
    
    def on_mode_change(self, e):
        """モード変更時"""
        # TODO: 実装
        pass


def main():
    """メインエントリポイント"""
    viewer = OfflineViewerUI()
    viewer.create_ui()
    ui.run(port=8084, title="JetRacer Offline Viewer")


if __name__ in {"__main__", "__mp_main__"}:
    main()
```

### 5.3 ui/depth_test_ui.py

深度推定テストUI。

```python
"""
深度推定テストUI

深度推定の動作確認とキャリブレーション調整。
"""

from nicegui import ui


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
    
    def __init__(self):
        pass
    
    def create_ui(self):
        """UIを構築"""
        
        with ui.header():
            ui.label("JetRacer 深度推定テスト").classes("text-h5")
        
        with ui.row().classes("w-full"):
            # 左: 元画像
            with ui.card().classes("w-1/3"):
                ui.label("元画像").classes("text-h6")
                # TODO: 画像表示
                pass
            
            # 中央: 深度マップ
            with ui.card().classes("w-1/3"):
                ui.label("深度マップ").classes("text-h6")
                # TODO: カラーマップ表示
                # TODO: ROI表示（3x3グリッド + 車体基準）
                pass
            
            # 右: 設定・結果
            with ui.card().classes("w-1/3"):
                ui.label("設定・結果").classes("text-h6")
                
                # 車体基準ROI設定
                # TODO: x1, y1, x2, y2 入力
                # TODO: 基準距離入力
                
                # キャリブレーションボタン
                # TODO: 実行ボタン
                
                # 結果表示
                # TODO: 各領域の距離
                # TODO: 警告レベル
                pass
        
        with ui.row().classes("w-full"):
            # 閾値調整
            with ui.card().classes("w-full"):
                ui.label("距離閾値").classes("text-h6")
                # TODO: EMERGENCY_STOP, STOP, SLOW のスライダー
                # TODO: プレビュー（閾値を超えた領域をハイライト）
                pass


def main():
    """メインエントリポイント"""
    depth_ui = DepthTestUI()
    depth_ui.create_ui()
    ui.run(port=8085, title="JetRacer Depth Test")


if __name__ in {"__main__", "__mp_main__"}:
    main()
```

### 5.4 ui/line_test_ui.py

ライン検出テストUI。

```python
"""
ライン検出テストUI

ライン検出の動作確認とパラメータ調整。
"""

from nicegui import ui


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
    
    def __init__(self):
        pass
    
    def create_ui(self):
        """UIを構築"""
        
        with ui.header():
            ui.label("JetRacer ライン検出テスト").classes("text-h5")
        
        with ui.row().classes("w-full"):
            # 左: 元画像 + 検出結果オーバーレイ
            with ui.card().classes("w-1/2"):
                ui.label("検出結果").classes("text-h6")
                # TODO: 画像表示
                # TODO: 検出ラインを描画
                # TODO: 走行目標点を描画
                pass
            
            # 右: マスク表示
            with ui.card().classes("w-1/2"):
                with ui.tabs() as tabs:
                    ui.tab("白線", name="white")
                    ui.tab("黄線", name="yellow")
                    ui.tab("統合", name="combined")
                
                with ui.tab_panels(tabs):
                    with ui.tab_panel("white"):
                        # TODO: 白線マスク表示
                        pass
                    with ui.tab_panel("yellow"):
                        # TODO: 黄線マスク表示
                        pass
                    with ui.tab_panel("combined"):
                        # TODO: 統合マスク表示
                        pass
        
        with ui.row().classes("w-full"):
            # パラメータ調整パネル
            with ui.card().classes("w-1/2"):
                ui.label("白線パラメータ").classes("text-h6")
                # TODO: H, S, V の範囲スライダー
                pass
            
            with ui.card().classes("w-1/2"):
                ui.label("黄線パラメータ").classes("text-h6")
                # TODO: H, S, V の範囲スライダー
                pass
        
        with ui.row().classes("w-full"):
            # ステアリング結果
            with ui.card().classes("w-full"):
                ui.label("ステアリング").classes("text-h6")
                # TODO: steering_offset を視覚化
                # TODO: 信頼度表示
                pass


def main():
    """メインエントリポイント"""
    line_ui = LineTestUI()
    line_ui.create_ui()
    ui.run(port=8086, title="JetRacer Line Test")


if __name__ in {"__main__", "__mp_main__"}:
    main()
```

---

## 6. 起動スクリプト

### 6.1 main_calibration.py

```python
#!/usr/bin/env python3
"""
環境キャリブレーションツール起動スクリプト
"""

import argparse
from ui.calibration_ui import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JetRacer 環境キャリブレーション")
    parser.add_argument("--port", type=int, default=8083, help="ポート番号")
    parser.add_argument("--image-dir", type=str, default="demo_images", help="画像ディレクトリ")
    args = parser.parse_args()
    
    main()
```

### 6.2 main_offline_test.py

```python
#!/usr/bin/env python3
"""
オフライン検証ツール起動スクリプト
"""

import argparse
from ui.offline_viewer_ui import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JetRacer オフライン検証")
    parser.add_argument("--port", type=int, default=8084, help="ポート番号")
    parser.add_argument("--session", type=str, help="録画セッションパス")
    args = parser.parse_args()
    
    main()
```

---

## 7. Depth Anything V2 Small 導入手順

### 7.1 モデルのダウンロード

```bash
# Hugging Face からダウンロード
pip install huggingface_hub

python -c "
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Small',
    filename='depth_anything_v2_vits.pth'
)
print(f'Downloaded to: {model_path}')
"
```

### 7.2 ONNX変換スクリプト

`scripts/convert_depth_model.py`:

```python
#!/usr/bin/env python3
"""
Depth Anything V2 Small を ONNX に変換するスクリプト

使用法:
    python scripts/convert_depth_model.py \
        --input depth_anything_v2_vits.pth \
        --output output/models/depth_anything_v2_small.onnx \
        --input-size 320 240
"""

import argparse
import torch
import torch.onnx

# TODO: 実装
# 1. モデルをロード
# 2. ダミー入力を作成
# 3. torch.onnx.export() で変換
# 4. onnx.checker.check_model() で検証

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="入力 .pth ファイル")
    parser.add_argument("--output", required=True, help="出力 .onnx ファイル")
    parser.add_argument("--input-size", nargs=2, type=int, default=[320, 240],
                        help="入力サイズ (width height)")
    args = parser.parse_args()
    
    # TODO: 変換処理
    pass


if __name__ == "__main__":
    main()
```

### 7.3 TensorRT変換（Jetson用）

```bash
# Jetson上で実行
/usr/src/tensorrt/bin/trtexec \
    --onnx=output/models/depth_anything_v2_small.onnx \
    --saveEngine=output/models/depth_anything_v2_small.trt \
    --fp16
```

---

## 8. 実装順序

### Phase 1: オフライン検証環境構築

1. `core/image_loader.py` - 画像読み込み
2. `config/default.yaml` - 設定ファイル
3. `ui/offline_viewer_ui.py` - 基本表示機能のみ

### Phase 2: 深度推定 (Mode A)

1. `scripts/convert_depth_model.py` - モデル変換
2. `core/depth_estimation.py` - DepthEstimator, DepthCalibrator
3. `core/obstacle_analyzer.py` - ObstacleAnalyzer（depth_estimation.pyに含めてもよい）
4. `ui/depth_test_ui.py` - 動作確認UI
5. `ui/calibration_ui.py` - 車体基準キャリブレーション部分

### Phase 3: ライン検出 (Mode B)

1. `core/line_detection.py` - LineDetector, ColorCalibrator
2. `ui/line_test_ui.py` - 動作確認UI
3. `ui/calibration_ui.py` - 色キャリブレーション部分
4. `core/calibration.py` - EnvironmentManager

### Phase 4: 統合判定

1. `core/driving_decision.py` - IntegratedDecisionMaker
2. `ui/offline_viewer_ui.py` - 統合表示・判定結果表示
3. 統合テスト

### Phase 5: リアルタイム化（この仕様書のスコープ外）

1. `core/camera_manager.py` - カメラ入力
2. `ui/realtime_ui.py` - リアルタイム表示
3. JetRacer制御との接続

---

## 9. テスト手順

### 9.1 単体テスト

```bash
# 深度推定テスト
python -m pytest tests/test_depth_estimation.py -v

# ライン検出テスト
python -m pytest tests/test_line_detection.py -v

# 統合判定テスト
python -m pytest tests/test_driving_decision.py -v
```

### 9.2 オフライン検証

```bash
# 1. キャリブレーションツールで環境設定
python main_calibration.py --port 8083
# → ブラウザで http://localhost:8083 を開く
# → 画像を選択し、白線・黄線・床・車体基準を設定
# → 環境名を入力して保存

# 2. オフライン検証ツールで処理確認
python main_offline_test.py --port 8084
# → ブラウザで http://localhost:8084 を開く
# → 録画セッションを選択
# → フレームを進めながら処理結果を確認
```

### 9.3 期待される出力

**深度推定:**
- 入力: 640x480 BGR画像
- 出力: 
  - depth_map: 640x480 float32 (0.0-1.0)
  - 推論時間: < 50ms (目標)
  - 各領域の距離推定値

**ライン検出:**
- 入力: 640x480 BGR画像
- 出力:
  - white_mask, yellow_mask: 640x480 uint8
  - lines: 検出された直線のリスト
  - target_point: 走行目標点 (x, y)
  - steering_offset: -1.0 〜 1.0

**統合判定:**
- 入力: 足元セグメンテーション結果、正面カメラ結果
- 出力:
  - command: GO / SLOW / STOP / TURN_LEFT / TURN_RIGHT
  - speed: 0.0-1.0
  - steering: -1.0 〜 1.0
  - reason: 判定理由

---

## 10. 既存コードとの連携

### 10.1 セグメンテーションの使用

既存の `core/onnx_inference.py` の `ONNXSegmenter` をそのまま使用:

```python
from core.onnx_inference import ONNXSegmenter

# 初期化
segmenter = ONNXSegmenter(
    model_path="output/models/road_segmentation.onnx",
    input_size=(320, 240),
    use_cuda=True
)

# 推論
mask, inference_time = segmenter.inference(image)
# mask: 0=Other, 1=ROAD, 2=MYCAR
```

### 10.2 設定の読み込み

新規の YAML 設定と既存の JSON 設定を両方サポート:

```python
import yaml
import json
from pathlib import Path

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.suffix == ".yaml":
        with open(path) as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
```

---

## 11. 注意事項

### 11.1 パフォーマンス

- 目標フレームレート: 15fps（処理込み）
- セグメンテーション: 約30fps (既存)
- 深度推定: 目標15fps以上
- 両方合わせて15fps達成が目標

### 11.2 メモリ管理

- Jetson Orin Nano のメモリ制約を考慮
- 大きな画像は処理前にリサイズ
- モデルは起動時に一度だけロード

### 11.3 エラーハンドリング

- モデルファイルが見つからない場合の適切なエラーメッセージ
- カメラ画像が取得できない場合のフォールバック
- 設定ファイルの読み込みエラー時のデフォルト値使用

---

## 12. 付録

### 12.1 requirements.txt への追加

```
# 既存の requirements.txt に追加
pyyaml>=6.0              # YAML設定ファイル用
```

### 12.2 画面レイアウト（正面カメラ）

```
正面カメラ画像 (640x480)
┌─────────────────────────────────────────┐  y=0
│                                         │
│    ┌─────────┬─────────┬─────────┐      │  y=60 (H*0.125)
│    │  左上   │  中上   │  右上   │      │
│    │         │         │         │      │
│    ├─────────┼─────────┼─────────┤      │  y=200 (H*0.417)
│    │  左中   │  中央   │  右中   │      │
│    │    ★主要判定領域★           │      │
│    ├─────────┼─────────┼─────────┤      │  y=340 (H*0.708)
│    │  左下   │  中下   │  右下   │      │
│    │         │         │         │      │
│    └─────────┴─────────┴─────────┘      │  y=420 (H*0.875)
│                                         │
├────────────┬───────────┬────────────────┤  y=420
│            │ 車体基準   │                │
│            │ (160x60)  │                │  面積: 約1/16
│            │ x:240-400 │                │
└────────────┴───────────┴────────────────┘  y=480

3x3 ROI:
  各セル: 約160x120ピクセル
  x分割: 0-213, 213-427, 427-640
  y分割: 60-200, 200-340, 340-420 (車体基準を除く)
```

---

以上

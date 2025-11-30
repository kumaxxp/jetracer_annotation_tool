# JetRacer 軽量セグメンテーションシステム 仕様書

## 1. システム概要

### 1.1 目的

JetRacer（Jetson Orin Nano Super）で10-15 FPSのリアルタイム自律走行を実現するための軽量セグメンテーションシステムの開発。

### 1.2 開発フロー

```
[アノテーションツール] → [ROADデータセット作成] → [軽量モデル訓練] → [モデルテスト] → [自律走行]
```

1. **Phase 0 (完了)**: ADE20K アノテーション GUI の実装
2. **Phase 1**: トレーニングデータエクスポート機能
3. **Phase 2**: モデルテスト・推論機能
4. **Phase 3**: JetRacer への統合

---

## 2. 現在のモデル (GUI用)

### 2.1 OneFormer モデル

アノテーションツールで使用している現在のモデル：

| 項目 | 値 |
|------|------|
| **モデル名** | OneFormer (ADE20K Swin Tiny) |
| **HuggingFace ID** | `shi-labs/oneformer_ade20k_swin_tiny` |
| **フレームワーク** | PyTorch + Transformers |
| **出力クラス数** | 150 (ADE20K) |
| **入力サイズ** | 可変 (640x480で測定) |

### 2.2 パフォーマンス測定結果

**測定環境**: Jetson Orin Nano Super (実機測定)

| 項目 | 値 |
|------|------|
| **処理時間** | ~6秒/画像 (640x480) |
| **バッチ処理** | 6画像で36秒 |
| **FPS換算** | ~0.17 FPS |
| **推論デバイス** | CPU (CUDA未使用) |

**測定ログ抜粋**:
```
16:34:53 [INFO] ============================================================
16:34:53 [INFO] ADE20K Image Segmentation
16:34:53 [INFO] ============================================================
16:34:53 [INFO] Input dir:  /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358
16:34:53 [INFO] Output dir: /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358_segmented
16:34:53 [INFO] Model:      shi-labs/oneformer_ade20k_swin_tiny
...
16:35:29 [INFO] ✓ Segmentation complete!
16:35:29 [INFO]   Processed: 6 images
```

### 2.3 結論

- **リアルタイム性**: ❌ 不適合
- **自律走行用途**: ❌ 使用不可 (目標15 FPS vs 実測0.17 FPS)
- **アノテーション用途**: ✅ 適合 (高精度・150クラス)

OneFormerは高精度なアノテーション生成には適しているが、リアルタイム推論には重すぎる。

---

## 3. ターゲットモデル (自律走行用)

### 3.1 MobileNetV2-DeepLabV3

`jetracer_minimal/segmentation.py` で使用されている軽量モデル：

| 項目 | 値 |
|------|------|
| **アーキテクチャ** | MobileNetV2 + DeepLabV3 |
| **フォーマット** | ONNX |
| **モデルサイズ** | 43 MB |
| **入力サイズ** | 320×240 (固定) |
| **出力クラス数** | カスタム (0: 非走行可能, 1: 走行可能) |
| **推論バックエンド** | OpenCV DNN (CUDA FP16対応) |

### 3.2 実装詳細

```python
# jetracer_minimal/segmentation.py より
class SegmentationModel:
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (320, 240),
        road_classes: Iterable[int] | None = None,
    ):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
```

### 3.3 パフォーマンス目標

**ターゲット FPS**: 15 FPS (66.7ms/フレーム)

| コンポーネント | 処理時間予算 | 備考 |
|---------------|-------------|------|
| カメラキャプチャ | ~10ms | CSI カメラ (640×480 @ 21fps) |
| 前処理 (リサイズ等) | ~5ms | 640×480 → 320×240 |
| **セグメンテーション** | **< 40ms** | **CUDA FP16推論** |
| 後処理 (マスク生成) | ~5ms | argmax + binary mask |
| 制御ロジック | ~6.7ms | ステアリング計算 |
| **合計** | **< 66.7ms** | **15 FPS達成** |

**重要**: セグメンテーション処理は40ms以下である必要がある。

---

## 4. Phase 1: トレーニングデータエクスポート機能

### 4.1 目的

アノテーション結果（ROAD mapping）を基に、軽量モデル訓練用のデータセットを生成する。

### 4.2 機能要件

#### 入力

1. **元画像**: `session_YYYYMMDD_HHMMSS/img_XXXX.jpg` (640×480)
2. **ADE20Kセグメンテーション**: `img_XXXX_seg.png` (PNG, 各ピクセル = ADE20K class ID)
3. **ROADマッピング**: `output/road_mapping.json`

#### 出力

トレーニング用データセット（Train/Val分割）:

```
output/training_data/
├── train/
│   ├── images/
│   │   ├── img_0001.jpg (640×480 オリジナル)
│   │   ├── img_0003.jpg
│   │   └── ...
│   └── labels/
│       ├── img_0001.png (640×480 バイナリマスク: 0=非ROAD, 255=ROAD)
│       ├── img_0003.png
│       └── ...
└── val/
    ├── images/
    │   ├── img_0002.jpg
    │   └── ...
    └── labels/
        ├── img_0002.png
        └── ...
```

#### データセット分割

- **Train**: 80%
- **Val**: 20%
- ランダムシード固定でシャッフル

### 4.3 ラベル生成ロジック

```python
def generate_binary_mask(seg_image: np.ndarray, road_mapping: dict) -> np.ndarray:
    """
    ADE20K セグメンテーション画像からバイナリROADマスクを生成

    Args:
        seg_image: ADE20K segmentation (H, W) - 各ピクセルはクラスID (0-150)
        road_mapping: {label_name: is_road} 辞書

    Returns:
        Binary mask (H, W) - 0: 非ROAD, 255: ROAD
    """
    # ADE20K_LABELS を使ってクラスID → ラベル名の逆引き
    mask = np.zeros_like(seg_image, dtype=np.uint8)

    for class_id, label_name in ADE20K_LABELS.items():
        if road_mapping.get(label_name, False):
            mask[seg_image == class_id] = 255

    return mask
```

### 4.4 UI実装

**追加ボタン**:
```
[Export Training Data]
```

**機能**:
1. 現在のROADマッピングを使用
2. 全画像を処理してバイナリマスク生成
3. Train/Val分割
4. `output/training_data/` に保存
5. 進捗表示 (例: "Processing 1/6...")
6. 完了通知 (例: "✓ Exported 4 train, 2 val images")

---

## 5. Phase 2: モデルテスト・推論機能

### 5.1 目的

訓練済みONNXモデルを使って推論結果を可視化し、性能を評価する。

### 5.2 機能要件

#### 入力

1. **ONNXモデル**: `models/road_segmentation.onnx`
2. **テスト画像**: 任意の640×480画像

#### 出力

1. **推論結果**: バイナリマスク (320×240 → 640×480にリサイズ)
2. **推論時間**: ミリ秒単位で測定
3. **可視化**: 元画像 + マスクオーバーレイ

### 5.3 推論パイプライン

```python
def inference_pipeline(image: np.ndarray, model_path: str) -> Tuple[np.ndarray, float]:
    """
    OpenCV DNN を使った推論

    Args:
        image: BGR image (H, W, 3)
        model_path: ONNX model path

    Returns:
        mask: Binary mask (H, W) - 0 or 255
        inference_time: ミリ秒
    """
    # 1. モデルロード
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    # 2. 前処理
    resized = cv2.resize(image, (320, 240))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(rgb, scalefactor=1.0/255.0, size=(320, 240))

    # 3. 推論
    net.setInput(blob)
    start = time.perf_counter()
    output = net.forward()
    inference_time = (time.perf_counter() - start) * 1000

    # 4. 後処理
    logits = output[0]  # (num_classes, 240, 320)
    class_map = np.argmax(logits, axis=0)  # (240, 320)
    mask = (class_map == 1).astype(np.uint8) * 255

    # 5. 元のサイズにリサイズ
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask, inference_time
```

### 5.4 UI実装

**新規タブ**: "Model Testing"

**コンポーネント**:
1. **モデル選択**: ファイルピッカーでONNXモデルを選択
2. **画像選択**: 現在のセッション画像から選択
3. **推論実行**: [Run Inference] ボタン
4. **結果表示**:
   - 元画像
   - 推論マスク
   - オーバーレイ表示
   - **推論時間**: 例 "Inference time: 35.2ms (28.4 FPS)"
5. **バッチテスト**: 全画像で推論し、平均時間を計算

**パフォーマンス判定**:
```
✅ < 40ms:  "Real-time capable (15+ FPS)"
⚠️ 40-66ms: "Near real-time (10-15 FPS)"
❌ > 66ms:  "Too slow for real-time (<10 FPS)"
```

---

## 6. データフォーマット仕様

### 6.1 ROAD Mapping JSON

**ファイル**: `output/road_mapping.json`

**フォーマット**:
```json
{
  "road": true,
  "floor": true,
  "path": true,
  "sidewalk": true,
  "wall": false,
  "sky": false,
  "tree": false,
  "grass": false,
  ...
}
```

- **キー**: ADE20K ラベル名 (150種類)
- **値**: `true` (ROAD) または `false` (非ROAD)

### 6.2 セグメンテーション画像

**ファイル**: `img_XXXX_seg.png`

**フォーマット**:
- PNG 8-bit grayscale
- 各ピクセル値 = ADE20K クラスID (0-150)
- サイズ: 元画像と同じ (640×480)

### 6.3 バイナリラベル画像

**ファイル**: `output/training_data/{train,val}/labels/img_XXXX.png`

**フォーマット**:
- PNG 8-bit grayscale
- ピクセル値: `0` (非ROAD) または `255` (ROAD)
- サイズ: 640×480 (訓練時に320×240にリサイズ)

---

## 7. モデル訓練仕様（参考）

### 7.1 推奨フレームワーク

**PyTorch + segmentation_models.pytorch**

```python
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,  # 背景 vs ROAD
)
```

### 7.2 訓練パラメータ（推奨）

| パラメータ | 値 |
|-----------|-----|
| Input Size | 320×240 |
| Batch Size | 16 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | BCEWithLogitsLoss |
| Epochs | 50-100 |
| Data Augmentation | RandomFlip, RandomBrightness, RandomContrast |

### 7.3 ONNX エクスポート

```python
# PyTorch → ONNX
dummy_input = torch.randn(1, 3, 240, 320)
torch.onnx.export(
    model,
    dummy_input,
    "road_segmentation.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)
```

---

## 8. パフォーマンスベンチマーク

### 8.1 モデル比較

| モデル | FPS (推定) | 用途 | サイズ | 精度 |
|--------|-----------|------|--------|------|
| **OneFormer (Swin Tiny)** | 0.17 | アノテーション | ~1-2 GB | 非常に高 (150クラス) |
| **MobileNetV2-DeepLabV3** | 15-25 | リアルタイム推論 | 43 MB | 中程度 (2クラス) |

### 8.2 JetRacer 処理フロー (目標)

```
[CSI Camera]
    ↓ ~10ms
[Image Capture] (640×480)
    ↓ ~5ms
[Preprocessing] (→ 320×240)
    ↓ ~40ms (CRITICAL PATH)
[Segmentation] (ONNX + CUDA FP16)
    ↓ ~5ms
[Postprocessing] (Binary Mask)
    ↓ ~6.7ms
[Control Logic] (Steering Calculation)
    ↓
[Motor Control]

Total: ~66.7ms (15 FPS)
```

---

## 9. 実装優先度

### High Priority (Phase 1)

- [x] ADE20K アノテーション GUI (完了)
- [ ] Training data export 機能
- [ ] Train/Val split 実装
- [ ] Binary mask 生成ロジック

### Medium Priority (Phase 2)

- [ ] ONNX モデルローダー (OpenCV DNN)
- [ ] 推論パイプライン実装
- [ ] パフォーマンス測定機能
- [ ] Model Testing タブ UI

### Low Priority (Phase 3)

- [ ] JetRacer メインコードへの統合
- [ ] リアルタイムパフォーマンステスト
- [ ] モデル最適化（量子化など）

---

## 10. 参考リンク

- **jetracer_minimal**: `/home/jetson/jetracer_minimal/segmentation.py`
- **OpenCV DNN CUDA**: [OpenCV CUDA Modules](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- **ADE20K Dataset**: [MIT ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
- **segmentation_models.pytorch**: [GitHub](https://github.com/qubvel/segmentation_models.pytorch)

---

## 11. 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-11-30 | 1.0 | 初版作成 - Phase 0完了、Phase 1-2仕様策定 |

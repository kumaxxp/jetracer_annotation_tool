# モデル訓練・テストガイド

## 概要

JetRacerの軽量セグメンテーションモデルを訓練・テストするための完全なワークフローガイドです。

## ワークフロー

```
[1. アノテーション] → [2. データエクスポート] → [3. モデル訓練] → [4. モデルテスト] → [5. ONNX推論]
```

---

## 1. アノテーション作業

### ステップ1: アノテーションGUIを起動

```bash
# アノテーションGUIを起動（ポート8081）
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358 --port 8081
```

### ステップ2: ROAD領域をマッピング

1. 画像上のセグメンテーション領域をクリック
2. ラベル情報を確認
3. "Toggle ROAD"ボタンで走行可能領域として設定
4. 複数の画像で繰り返し
5. "Save Mapping"で保存

**推奨ROADラベル**:
- road (道路)
- floor (床)
- path (小道)
- sidewalk (歩道)

---

## 2. トレーニングデータのエクスポート

### ステップ1: アノテーション完了後、データをエクスポート

アノテーションGUI（ポート8081）で:
1. "Export Training Data"ボタンをクリック
2. データが自動的に以下の形式で出力されます：

```
output/training_data/
├── train/           # 訓練データ（80%）
│   ├── images/      # 元画像
│   │   ├── img_0001.jpg
│   │   ├── img_0003.jpg
│   │   └── ...
│   └── labels/      # バイナリマスク（0=非ROAD, 255=ROAD）
│       ├── img_0001.png
│       ├── img_0003.png
│       └── ...
└── val/             # 検証データ（20%）
    ├── images/
    └── labels/
```

### エクスポート結果の確認

```bash
# ファイル数を確認
ls output/training_data/train/images/ | wc -l
ls output/training_data/val/images/ | wc -l
```

---

## 3. モデル訓練

### ステップ1: トレーニングGUIを起動

```bash
# トレーニング/テストGUIを起動（ポート8082）
./run_training.sh --port 8082
```

ブラウザでアクセス:
- http://localhost:8082
- http://192.168.1.65:8082

### ステップ2: Trainingタブで訓練実行

1. **Training Configuration**を確認:
   - Training Data: `output/training_data`
   - Epochs: 50（推奨）
   - Batch Size: 8（メモリに応じて調整）
   - Learning Rate: 0.0001

2. **Start Training**ボタンをクリック

3. **Training Log**で進捗を確認:
   ```
   Epoch 1/50 - Train Loss: 0.4523, Val Loss: 0.4012
   Epoch 2/50 - Train Loss: 0.3891, Val Loss: 0.3654
   ...
   ```

4. 訓練完了後、モデルが保存されます:
   ```
   output/models/
   ├── best_model.pth          # 最良のモデル（Val Lossが最小）
   ├── final_model.pth         # 最終エポックのモデル
   ├── model_epoch_10.pth      # チェックポイント
   ├── model_epoch_20.pth
   └── road_segmentation.onnx  # ONNX形式（推論用）
   ```

### 訓練パラメータの調整

| パラメータ | デフォルト | 推奨範囲 | 説明 |
|-----------|----------|---------|------|
| Epochs | 50 | 30-100 | 訓練回数 |
| Batch Size | 8 | 4-16 | バッチサイズ（メモリに応じて） |
| Learning Rate | 0.0001 | 0.00001-0.001 | 学習率 |

**メモリ不足の場合**: Batch Sizeを4に減らす

---

## 4. モデルテスト

### ステップ1: Testingタブに移動

トレーニングGUI（ポート8082）のTestingタブを開く

### ステップ2: モデルとテスト画像をロード

1. **Model Selection**:
   - ONNX Model: `output/models/road_segmentation.onnx`
   - "Load Model"ボタンをクリック

2. **Test Images**:
   - Test Images: `output/training_data/val/images`
   - "Load Images"ボタンをクリック

### ステップ3: 推論実行

1. 画像を選択（Previous/Nextボタン）
2. "Run Inference"ボタンをクリック
3. 3つのセグメンテーションを比較:

| 列 | 内容 | 説明 |
|----|------|------|
| **左** | ADE20K Segmentation | OneFormerの高精度セグメンテーション（150クラス） |
| **中央** | Ground Truth | アノテーション結果から生成したバイナリマスク |
| **右** | Prediction | 訓練したモデルの推論結果 |

### ステップ4: パフォーマンス確認

**Performance Metrics**で推論時間を確認:

```
✅ Inference time: 35.2ms (28.4 FPS)  # 目標達成！
⚠️ Inference time: 52.3ms (19.1 FPS)  # ほぼ目標
❌ Inference time: 78.5ms (12.7 FPS)  # 遅い
```

**目標値**:
- **< 40ms**: リアルタイム対応（15+ FPS）
- **40-66ms**: ニアリアルタイム（10-15 FPS）
- **> 66ms**: リアルタイムには遅い（<10 FPS）

---

## 5. ONNX推論（Python）

訓練したモデルをPythonコードで使用する例：

```python
from core.onnx_inference import ONNXSegmenter
import cv2

# モデルロード
segmenter = ONNXSegmenter(
    model_path="output/models/road_segmentation.onnx",
    input_size=(320, 240),
    use_cuda=True  # CUDA利用
)

# 画像ロード
image = cv2.imread("test.jpg")

# 推論
mask, inference_time = segmenter.inference(image)

print(f"Inference time: {inference_time:.1f}ms")
print(f"Mask shape: {mask.shape}")  # (H, W) - 0 or 255

# マスクを可視化
cv2.imshow("ROAD Mask", mask)
cv2.waitKey(0)
```

---

## トラブルシューティング

### 訓練がメモリ不足で失敗

```bash
# Batch Sizeを減らす
# トレーニングGUIで: Batch Size = 4
```

### CUDA out of memory

```bash
# CPUで訓練（遅いが安定）
export CUDA_VISIBLE_DEVICES=""
./run_training.sh
```

### 推論が遅い（>40ms）

考えられる原因：
1. **CUDAが無効**: モデルロード時に`use_cuda=True`を確認
2. **入力サイズが大きい**: 320x240を推奨
3. **モデルが複雑**: エンコーダを軽量化（mobilenet_v2推奨）

### Ground Truthが表示されない

テスト画像ディレクトリの構造を確認：
```
output/training_data/val/
├── images/
│   ├── img_0002.jpg
│   └── ...
└── labels/
    ├── img_0002.png  # 同じファイル名
    └── ...
```

---

## パフォーマンス最適化

### 1. モデルの軽量化

```python
# より軽量なエンコーダを使用
from core.model_trainer import ROADTrainer

trainer = ROADTrainer(
    encoder_name="mobilenet_v2",  # 最軽量
    # encoder_name="resnet18",    # やや重い
    # encoder_name="resnet50",    # 重い
)
```

### 2. 入力サイズの調整

```python
# 小さいサイズ = 高速だが精度低下
trainer = ROADTrainer(input_size=(256, 192))  # 超高速

# デフォルトサイズ = バランス
trainer = ROADTrainer(input_size=(320, 240))  # 推奨

# 大きいサイズ = 高精度だが低速
trainer = ROADTrainer(input_size=(640, 480))  # 遅い
```

### 3. CUDA FP16の確認

```python
# ONNXSegmenterがCUDA FP16を使用しているか確認
segmenter = ONNXSegmenter(model_path="...", use_cuda=True)
# 出力: "✓ Using CUDA backend (FP16)"
```

---

## 次のステップ

1. ✅ **アノテーション**: ROAD領域をマッピング
2. ✅ **エクスポート**: トレーニングデータを生成
3. ✅ **訓練**: 軽量モデルを訓練
4. ✅ **テスト**: 3つのセグメンテーションを比較
5. 🚀 **自律走行**: JetRacerに統合

---

## 参考資料

- [SPECIFICATION.md](SPECIFICATION.md) - システム仕様
- [README.md](README.md) - メインドキュメント
- [REAL_DATA_USAGE.md](REAL_DATA_USAGE.md) - 実写データ使用方法

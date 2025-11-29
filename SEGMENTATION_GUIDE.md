# ADE20K セグメンテーション ガイド

## 概要

このガイドでは、JetRacerの実写データに対してADE20Kセグメンテーションを適用する方法を説明します。

## セグメンテーションとは

セグメンテーションは、画像内の各ピクセルを意味のあるクラス（道路、壁、空など）に分類する技術です。ADE20Kは150種類のクラスを識別できる大規模なセグメンテーションデータセットです。

## 必要な依存関係のインストール

```bash
# 仮想環境を有効化
source venv/bin/activate

# 追加の依存関係をインストール
pip install torch transformers opencv-python
```

## セグメンテーションの実行

### 方法1: 専用スクリプトを使用

```bash
# 基本的な使い方
python segment_images.py --input-dir /path/to/images

# カスタム出力ディレクトリを指定
python segment_images.py \
    --input-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358 \
    --output-dir /home/jetson/jetracer_annotation_tool/segmented_data

# PNGファイルを処理
python segment_images.py --input-dir /path/to/images --pattern "*.png"
```

### 方法2: Pythonスクリプトで使用

```python
from core.ade20k_segmentation import segment_directory

# ディレクトリ内の全画像をセグメンテーション
output_paths = segment_directory(
    input_dir="/path/to/images",
    output_dir="/path/to/output",
    pattern="*.jpg"
)

print(f"Processed {len(output_paths)} images")
```

## 実写データでの使用例

### ステップ1: セグメンテーション処理

```bash
# 実写データをセグメンテーション
python segment_images.py \
    --input-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358

# 出力先: /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358_segmented
```

### ステップ2: セグメンテーション結果を元の画像ディレクトリにコピー

```bash
# セグメンテーション結果を元のディレクトリにコピー
cp /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358_segmented/*_seg.png \
   /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358/
```

### ステップ3: アノテーションツールで確認

```bash
# アノテーションツールを起動
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358
```

アノテーションツールは自動的に`*_seg.png`ファイルを検出し、実際のセグメンテーション結果を表示します。

## 出力ファイル形式

セグメンテーション結果は以下の形式で保存されます：

```
元の画像:        img_0001.jpg
セグメンテーション: img_0001_seg.png
```

セグメンテーション画像（PNG）の各ピクセル値は、ADE20KクラスIDを表します（0-149）。

## モデルについて

デフォルトで使用されるモデル：
- **OneFormer (ADE20K Swin Tiny)**
- HuggingFace: `shi-labs/oneformer_ade20k_swin_tiny`
- 高速で軽量なモデル
- Jetson Nanoでも動作可能

### 異なるモデルを使用

```bash
# より高精度なモデルを使用（重い）
python segment_images.py \
    --input-dir /path/to/images \
    --model shi-labs/oneformer_ade20k_swin_large
```

## パフォーマンス

**Jetson Nanoでの推定処理時間**:
- Swin Tiny: ~3-5秒/画像
- Swin Large: ~10-15秒/画像

**推奨**:
- バッチ処理: デスクトップPCやクラウドで処理
- リアルタイム: Jetson Xavier NX以上を推奨

## トラブルシューティング

### メモリ不足エラー

```bash
# より小さいモデルを使用
python segment_images.py \
    --input-dir /path/to/images \
    --model shi-labs/oneformer_ade20k_swin_tiny
```

### CUDA out of memory

```python
# CPUで実行
import torch
torch.cuda.empty_cache()  # メモリをクリア
```

または、環境変数でCPUモードを強制：
```bash
CUDA_VISIBLE_DEVICES="" python segment_images.py --input-dir /path/to/images
```

### モデルのダウンロードに時間がかかる

初回実行時、HuggingFaceから約1-2GBのモデルをダウンロードします。安定したネットワーク接続が必要です。

## ADE20Kクラス一覧（主要なもの）

| クラスID | クラス名 | 日本語 |
|---------|---------|--------|
| 4 | floor | 床 |
| 7 | road | 道路 |
| 12 | sidewalk | 歩道 |
| 53 | path | 小道 |
| 1 | wall | 壁 |
| 3 | sky | 空 |
| 5 | tree | 木 |
| 10 | grass | 草 |
| 13 | person | 人 |
| 21 | car | 車 |

完全なクラスリストは[data/ade20k_labels.py](data/ade20k_labels.py)を参照してください。

## ワンライナーでの完全な処理フロー

```bash
# 1. セグメンテーション実行
python segment_images.py \
    --input-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358 && \

# 2. セグメンテーション結果をコピー
cp /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358_segmented/*_seg.png \
   /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358/ && \

# 3. アノテーションツールを起動
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358
```

## 次のステップ

1. **セグメンテーション実行**: 実写データをセグメンテーション
2. **アノテーション**: ROADラベルをマッピング
3. **保存**: マッピングをJSONで保存
4. **自律走行**: マッピング結果をJetRacerで使用

---

**参考**:
- [README.md](README.md) - メインドキュメント
- [REAL_DATA_USAGE.md](REAL_DATA_USAGE.md) - 実写データ使用方法
- [QUICKSTART.md](QUICKSTART.md) - クイックスタート

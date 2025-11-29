# JetRacer ROAD Annotation Tool

ADE20Kセグメンテーション画像からJetRacerの走行可能領域（ROAD）をマッピングするアノテーションツール。

## 概要

このツールを使用すると：
- ADE20Kセグメンテーション画像を視覚化
- クリックでラベルを選択し、ROAD属性をトグル
- ROAD領域に縞々パターンを表示
- マッピングテーブルをJSONで保存

## 機能

### 主要機能
1. **画像表示**: セグメンテーションオーバーレイ付き画像表示
2. **インタラクティブ選択**: クリックでラベル選択
3. **ROADマッピング**: ラベルごとにROAD属性をトグル
4. **ビジュアルフィードバック**: ROAD領域に黄色の縞々パターン
5. **ナビゲーション**: 前へ/次へボタンで画像を切り替え
6. **永続化**: マッピングをJSONで保存

### UI構成

```
┌─────────────────────────────────────────────────┐
│  JetRacer ROAD Annotation Tool                  │
├────────────────────────┬────────────────────────┤
│                        │  Selected Label        │
│  セグメンテーション画像  │  - Label: road          │
│  (クリック可能)          │  - ID: 7                │
│                        │  - Status: ROAD        │
│                        │  [Toggle ROAD]         │
│                        ├────────────────────────┤
│                        │  ROAD Labels           │
│                        │  ✓ floor               │
│                        │  ✓ road                │
│                        │  ✓ path                │
│                        │  ✓ sidewalk            │
├────────────────────────┴────────────────────────┤
│  [Previous]  Image 1/3  [Next]  [Save Mapping] │
└─────────────────────────────────────────────────┘
```

## インストール

### 必要要件
- Python 3.10以上
- pip

### セットアップ

```bash
# リポジトリに移動
cd jetracer_annotation_tool

# 仮想環境を作成（推奨）
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

仮想環境を使用せずにグローバルにインストールする場合：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使い方（デモモード）

**仮想環境を使用する場合（推奨）：**
```bash
# 起動スクリプトを使用
./run.sh --generate-demo

# または、手動で仮想環境を有効化
source venv/bin/activate
python main.py --generate-demo
```

**仮想環境を使用しない場合：**
```bash
python main.py --generate-demo
```

### カスタム画像で使用

```bash
# 仮想環境使用の場合
./run.sh --image-dir /path/to/your/images --output-dir /path/to/output

# または
source venv/bin/activate
python main.py --image-dir /path/to/your/images --output-dir /path/to/output
```

### JetRacer実写データで使用

JetRacerで撮影した実写データを使用する場合：

```bash
# 実写データのディレクトリを指定
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358

# 別のポートで起動する場合
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358 --port 8081
```

**注意**: 実写データにセグメンテーション画像がない場合、自動的にダミーのセグメンテーションが生成されます。実際のADE20Kセグメンテーション画像を使用するには、以下のセグメンテーション機能を使用してください。

### ADE20Kセグメンテーション処理

実写データに対してADE20Kセグメンテーションを適用できます：

```bash
# 追加の依存関係をインストール（初回のみ）
source venv/bin/activate
pip install torch transformers opencv-python

# 実写データをセグメンテーション
python segment_images.py \
    --input-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358

# セグメンテーション結果を元のディレクトリにコピー
cp /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358_segmented/*_seg.png \
   /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358/

# アノテーションツールで確認
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358
```

詳細は[SEGMENTATION_GUIDE.md](SEGMENTATION_GUIDE.md)を参照してください。

### オプション

```bash
python main.py --help
```

主なオプション：
- `--image-dir`: 画像ディレクトリ (デフォルト: demo_images)
- `--output-dir`: 出力ディレクトリ (デフォルト: output)
- `--host`: バインドするホスト (デフォルト: 0.0.0.0)
- `--port`: ポート番号 (デフォルト: 8080)
- `--generate-demo`: デモ画像を生成
- `--num-demo-images`: デモ画像の数 (デフォルト: 3)

### Jetsonでの起動

```bash
# 仮想環境を有効化（推奨）
source venv/bin/activate

# Jetson上で起動
python main.py --host 0.0.0.0 --port 8080

# または起動スクリプトを使用
./run.sh --host 0.0.0.0 --port 8080

# JetsonのIPアドレスを確認
hostname -I

# ブラウザでアクセス (別のデバイスから)
# http://<JETSON_IP>:8080
```

## ワークフロー

1. **起動**: `./run.sh --generate-demo` (または `python main.py --generate-demo`)
2. **画像表示**: 最初の画像が自動的に読み込まれます
3. **ラベル選択**: 画像をクリックしてラベルを選択
4. **ROAD設定**: "Toggle ROAD"ボタンでROAD属性を切り替え
5. **ナビゲーション**: 次の画像に移動して繰り返し
6. **保存**: "Save Mapping"ボタンでマッピングを保存

## 出力形式

マッピングは `output/road_mapping.json` に保存されます：

```json
{
  "road_labels": [
    "floor",
    "path",
    "road",
    "sidewalk"
  ],
  "mapping": {
    "floor": true,
    "wall": false,
    "road": true,
    "sidewalk": true,
    "grass": false,
    ...
  }
}
```

## プロジェクト構成

```
jetracer_annotation_tool/
├── main.py                    # エントリーポイント
├── requirements.txt           # 依存関係
├── README.md                  # このファイル
├── ui/
│   ├── __init__.py
│   └── annotation_ui.py      # NiceGUI UI実装
├── core/
│   ├── __init__.py
│   ├── segmentation.py       # セグメンテーション処理
│   └── mapping.py            # マッピング管理
├── data/
│   ├── __init__.py
│   └── ade20k_labels.py      # ADE20Kラベル定義
├── demo_images/              # デモ画像 (自動生成)
└── output/                   # 出力ファイル
    └── road_mapping.json     # マッピング結果
```

## 開発

### デモ画像の生成

```python
from core.segmentation import generate_demo_images
from pathlib import Path

generate_demo_images(Path("demo_images"), num_images=5)
```

### マッピングの読み込み

```python
from core.mapping import ROADMapping

mapping = ROADMapping("output/road_mapping.json")
road_labels = mapping.get_road_labels()
print(f"ROAD labels: {road_labels}")
```

## トラブルシューティング

### ポートが使用中
```bash
# 別のポートを指定
python main.py --port 8081
```

### Jetsonでアクセスできない
- ファイアウォール設定を確認
- JetsonのIPアドレスが正しいか確認: `hostname -I`
- 同じネットワークに接続されているか確認

### 画像が表示されない
- 画像ディレクトリのパスが正しいか確認
- 画像形式がJPGまたはPNGか確認

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！

## 連絡先

問題や質問がある場合は、Issueを作成してください。

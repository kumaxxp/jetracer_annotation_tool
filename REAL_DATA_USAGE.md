# 実写データの使用方法

## 現在の状態

✅ **アプリケーションは実写データで起動中です！**

- **データソース**: `/home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358`
- **画像数**: 6枚（640×480 JPEG）
- **アクセスURL**:
  - http://localhost:8081
  - http://192.168.1.65:8081

## 実写データの詳細

```
セッション: session_20251127_153358
撮影日時: 2025年11月27日 15:33
解像度: 640×480
FPS: 15
形式: JPEG

画像一覧:
- img_0001.jpg ~ img_0006.jpg
- 各画像にメタデータJSON付き
```

## 使い方

### 1. ブラウザでアクセス

```
http://192.168.1.65:8081
```

### 2. アノテーション作業

1. **画像をクリック**: セグメンテーション領域を選択
2. **ラベルを確認**: 右パネルでラベル名とIDを確認
3. **ROADを設定**: "Toggle ROAD"ボタンで走行可能領域として設定
4. **次の画像へ**: "Next"ボタンで次の画像に移動
5. **保存**: 作業完了後、"Save Mapping"で保存

### 3. マッピングの保存

マッピングは以下に保存されます：
```
/home/jetson/jetracer_annotation_tool/output/road_mapping.json
```

## セグメンテーションについて

現在、実写データには実際のADE20Kセグメンテーション画像がないため、**ダミーのセグメンテーション**が自動生成されています。

### 実際のセグメンテーションを使用する（推奨）

**組み込みのセグメンテーション機能を使用**:

```bash
# ステップ1: 追加の依存関係をインストール（初回のみ）
source venv/bin/activate
pip install torch transformers opencv-python

# ステップ2: セグメンテーション実行
python segment_images.py \
    --input-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358

# ステップ3: セグメンテーション結果をコピー
cp /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358_segmented/*_seg.png \
   /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358/

# ステップ4: アノテーションツールを再起動
# 自動的に実際のセグメンテーションを使用します
```

詳細は[SEGMENTATION_GUIDE.md](SEGMENTATION_GUIDE.md)を参照してください。

### その他の方法

#### 方法1: 外部セグメンテーションモデルを使用

```bash
# セグメンテーションモデル（例：MMSegmentation、DeepLabなど）で処理
# 出力: PNG形式のラベル画像（各ピクセルがADE20KラベルIDを持つ）
# ファイル名: {元の画像名}_seg.png
```

#### 方法2: 手動配置

セグメンテーション画像を元の画像と同じディレクトリに配置：
```
img_0001.jpg       # 元の画像
img_0001_seg.png   # セグメンテーション画像（自動検出）
```

## 次のステップ

1. **アノテーション作業**: 実写データで走行可能領域をマッピング
2. **マッピング保存**: `road_mapping.json`を保存
3. **自律走行に活用**: マッピング結果をJetRacerの制御に使用

## 別のセッションデータを使用

他のセッションのデータを使用する場合：

```bash
# 別のセッションを指定
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/[セッション名]

# 例
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_YYYYMMDD_HHMMSS
```

## トラブルシューティング

### 画像が表示されない

- 画像ディレクトリのパスが正しいか確認
- JPEGファイルが存在するか確認

### ダミーセグメンテーションを無効化

実際のセグメンテーション画像を用意して、`annotation_ui.py`で`seg_path`を指定してください。

---

**現在のステータス**: ✅ 正常動作中（ポート8081）

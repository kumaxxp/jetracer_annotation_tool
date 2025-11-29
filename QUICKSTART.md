# クイックスタートガイド

## 初回セットアップ

### 1. 仮想環境の作成とセットアップ

```bash
# プロジェクトディレクトリに移動
cd /home/jetson/jetracer_annotation_tool

# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. アプリケーションの起動

```bash
# 起動スクリプトを使用（デモモード）
./run.sh --generate-demo

# または、手動で起動
source venv/bin/activate
python main.py --generate-demo
```

### 3. ブラウザでアクセス

アプリケーションが起動したら、以下のURLにアクセス：

- **ローカル**: http://localhost:8080
- **ネットワーク**: http://192.168.1.65:8080

JetsonのIPアドレスを確認するには：
```bash
hostname -I
```

## 使い方

1. **画像をクリック** → セグメンテーションラベルを選択
2. **Toggle ROADボタン** → ROAD（走行可能領域）として設定/解除
3. **前へ/次へボタン** → 画像を切り替え
4. **Save Mappingボタン** → マッピングを保存

## トラブルシューティング

### ポートが使用中のエラー

```bash
# 既存のプロセスを停止
pkill -f "python.*main.py"

# 別のポートで起動
./run.sh --port 8081
```

### 仮想環境が有効でない

```bash
# 仮想環境を有効化
source venv/bin/activate

# プロンプトに (venv) が表示されることを確認
```

### クリックイベントが機能しない

- ブラウザのコンソールでエラーを確認
- ページをリロード
- アプリケーションを再起動

## 実写データの使用

JetRacerで撮影した実写データを使用する場合：

```bash
# 仮想環境を有効化
source venv/bin/activate

# 実写データで起動
./run.sh --image-dir /home/jetson/jetracer_minimal/data/raw_images/session_20251127_153358
```

現在、実写データ（6枚の画像）で起動中です：
- http://localhost:8081
- http://192.168.1.65:8081

## 次のステップ

詳細なドキュメント: [README.md](README.md)

マッピング結果: [output/road_mapping.json](output/road_mapping.json)

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
        if not self.session_path.exists():
            print(f"Error: Session path does not exist: {self.session_path}")
            return False

        # 1. metadata.yaml を読み込み
        metadata_path = self.session_path / "metadata.yaml"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = yaml.safe_load(f)
                print(f"✓ Loaded metadata from {metadata_path}")
            except Exception as e:
                print(f"Warning: Failed to load metadata: {e}")
                self.metadata = {}
        else:
            print(f"Warning: metadata.yaml not found in {self.session_path}")
            self.metadata = {}

        # 2. front/, ground/ ディレクトリの画像一覧を取得
        front_dir = self.session_path / "front"
        ground_dir = self.session_path / "ground"

        if front_dir.exists():
            self.front_images = sorted(front_dir.glob("*.jpg")) + sorted(front_dir.glob("*.jpeg")) + sorted(front_dir.glob("*.png"))
            print(f"✓ Found {len(self.front_images)} front images")
        else:
            print(f"Warning: front directory not found: {front_dir}")

        if ground_dir.exists():
            self.ground_images = sorted(ground_dir.glob("*.jpg")) + sorted(ground_dir.glob("*.jpeg")) + sorted(ground_dir.glob("*.png"))
            print(f"✓ Found {len(self.ground_images)} ground images")
        else:
            print(f"Warning: ground directory not found: {ground_dir}")

        if len(self.front_images) == 0 and len(self.ground_images) == 0:
            print("Error: No images found in session")
            return False

        return True

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
        if index < 0 or index >= self.get_frame_count():
            return None

        result = {
            "index": index,
            "front": None,
            "ground": None,
            "front_path": None,
            "ground_path": None
        }

        # 正面カメラ画像を読み込み
        if index < len(self.front_images):
            front_path = self.front_images[index]
            front_image = cv2.imread(str(front_path))
            if front_image is not None:
                result["front"] = front_image
                result["front_path"] = str(front_path)
            else:
                print(f"Warning: Failed to load image: {front_path}")

        # 足元カメラ画像を読み込み
        if index < len(self.ground_images):
            ground_path = self.ground_images[index]
            ground_image = cv2.imread(str(ground_path))
            if ground_image is not None:
                result["ground"] = ground_image
                result["ground_path"] = str(ground_path)
            else:
                print(f"Warning: Failed to load image: {ground_path}")

        return result

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
        if not self.folder_path.exists():
            print(f"Error: Folder does not exist: {self.folder_path}")
            return []

        self.image_paths = []
        for ext in self.extensions:
            # 拡張子の大文字・小文字両方に対応
            self.image_paths.extend(sorted(self.folder_path.glob(f"*{ext}")))
            self.image_paths.extend(sorted(self.folder_path.glob(f"*{ext.upper()}")))

        # 重複を除いてソート
        self.image_paths = sorted(list(set(self.image_paths)))

        print(f"✓ Found {len(self.image_paths)} images in {self.folder_path}")
        return self.image_paths

    def load_image(self, path: Path) -> np.ndarray:
        """画像を読み込み (BGR形式)"""
        return cv2.imread(str(path))

    def load_all(self) -> List[Tuple[Path, np.ndarray]]:
        """全画像を読み込み"""
        if not self.image_paths:
            self.load_paths()

        results = []
        for path in self.image_paths:
            image = self.load_image(path)
            if image is not None:
                results.append((path, image))
            else:
                print(f"Warning: Failed to load image: {path}")

        return results

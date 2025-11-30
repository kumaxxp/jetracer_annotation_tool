"""
環境キャリブレーション管理

深度キャリブレーションと色キャリブレーションを統合管理する。
"""

from pathlib import Path
from typing import Dict, Optional, List
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
        self.default_file = self.config_dir / "default.yaml"

        # ディレクトリを作成
        self.environments_dir.mkdir(parents=True, exist_ok=True)

    def load_default(self) -> Dict:
        """
        デフォルト設定を読み込み

        Returns:
            config: デフォルト設定
        """
        if not self.default_file.exists():
            print(f"Warning: Default config not found: {self.default_file}")
            return {}

        with open(self.default_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✓ Loaded default config from {self.default_file}")
        return config

    def load_current(self) -> Dict:
        """
        現在の環境設定を読み込み（デフォルトとマージ）

        Returns:
            config: 現在の設定（デフォルト + 環境設定）
        """
        # デフォルト設定を読み込み
        config = self.load_default()

        # 現在の環境ファイルが存在する場合はマージ
        if self.current_file.exists():
            with open(self.current_file, 'r', encoding='utf-8') as f:
                current_env_name = yaml.safe_load(f).get('current_environment')

            if current_env_name:
                env_config = self.load_environment(current_env_name)
                config = self._merge_configs(config, env_config)
                print(f"✓ Current environment: {current_env_name}")
        else:
            print("No current environment set. Using default config.")

        return config

    def list_environments(self) -> List[str]:
        """
        利用可能な環境一覧を取得

        Returns:
            env_names: 環境名のリスト
        """
        if not self.environments_dir.exists():
            return []

        env_files = list(self.environments_dir.glob("*.yaml"))
        env_names = [f.stem for f in env_files]

        return sorted(env_names)

    def set_current(self, name: str) -> bool:
        """
        現在の環境を設定

        Args:
            name: 環境名

        Returns:
            success: 設定成功した場合True
        """
        env_file = self.environments_dir / f"{name}.yaml"

        if not env_file.exists():
            print(f"Error: Environment not found: {name}")
            return False

        # current_environment.yamlに保存
        current_data = {
            'current_environment': name
        }

        with open(self.current_file, 'w', encoding='utf-8') as f:
            yaml.dump(current_data, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ Current environment set to: {name}")
        return True

    def load_environment(self, name: str) -> Dict:
        """
        環境設定を読み込み

        Args:
            name: 環境名

        Returns:
            config: 環境設定
        """
        env_file = self.environments_dir / f"{name}.yaml"

        if not env_file.exists():
            raise FileNotFoundError(f"Environment not found: {name}")

        with open(env_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def save_environment(self, name: str, config: Dict) -> str:
        """
        環境設定を保存

        Args:
            name: 環境名
            config: 設定内容

        Returns:
            saved_path: 保存したファイルパス
        """
        # ファイル名を安全な形式に変換
        safe_name = name.replace(' ', '_').replace('/', '_')
        env_file = self.environments_dir / f"{safe_name}.yaml"

        # YAML保存
        with open(env_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ Environment saved to: {env_file}")
        return str(env_file)

    def delete_environment(self, name: str) -> bool:
        """
        環境設定を削除

        Args:
            name: 環境名

        Returns:
            success: 削除成功した場合True
        """
        env_file = self.environments_dir / f"{name}.yaml"

        if not env_file.exists():
            print(f"Warning: Environment not found: {name}")
            return False

        # 現在の環境として設定されている場合は解除
        if self.current_file.exists():
            with open(self.current_file, 'r', encoding='utf-8') as f:
                current_data = yaml.safe_load(f)
                if current_data.get('current_environment') == name:
                    self.current_file.unlink()
                    print(f"Cleared current environment: {name}")

        # ファイルを削除
        env_file.unlink()
        print(f"✓ Environment deleted: {name}")
        return True

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """
        設定をマージ（オーバーライド優先）

        Args:
            base: ベース設定
            override: オーバーライド設定

        Returns:
            merged: マージされた設定
        """
        merged = base.copy()

        # calibrationセクションをマージ
        if 'calibration' in override:
            if 'calibration' not in merged:
                merged['calibration'] = {}

            calibration = override['calibration']

            # 白線設定
            if 'white_line' in calibration:
                if 'processing' not in merged:
                    merged['processing'] = {}
                if 'line_detection' not in merged['processing']:
                    merged['processing']['line_detection'] = {}

                merged['processing']['line_detection']['white_line'] = calibration['white_line']

            # 黄線設定
            if 'yellow_line' in calibration:
                if 'processing' not in merged:
                    merged['processing'] = {}
                if 'line_detection' not in merged['processing']:
                    merged['processing']['line_detection'] = {}

                merged['processing']['line_detection']['yellow_line'] = calibration['yellow_line']

            # 深度キャリブレーション
            if 'depth' in calibration:
                if 'processing' not in merged:
                    merged['processing'] = {}
                if 'depth' not in merged['processing']:
                    merged['processing']['depth'] = {}

                merged['processing']['depth'].update(calibration['depth'])

        return merged

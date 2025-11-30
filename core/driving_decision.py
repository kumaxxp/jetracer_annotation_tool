"""
統合判定モジュール

2カメラの結果を統合し、運転判断を行う。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


class DrivingCommand(Enum):
    """運転コマンド"""
    GO = "go"  # 直進
    SLOW = "slow"  # 減速
    STOP = "stop"  # 停止
    TURN_LEFT = "turn_left"  # 左折
    TURN_RIGHT = "turn_right"  # 右折
    EMERGENCY_STOP = "emergency_stop"  # 緊急停止


class ProcessingMode(Enum):
    """処理モード"""
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"  # Mode A: 障害物回避
    LINE_FOLLOWING = "line_following"  # Mode B: ライントレース


@dataclass
class DrivingDecision:
    """
    運転判断結果

    Attributes:
        command: 運転コマンド
        speed: 速度 (0.0 ~ 1.0)
        steering: ステアリング (-1.0 ~ 1.0, 左が負)
        reason: 判断理由
        confidence: 信頼度 (0.0 ~ 1.0)
        ground_safe: 地面の安全性 (True/False)
        front_clear: 前方の障害物なし (True/False)
        mode: 処理モード
    """
    command: DrivingCommand
    speed: float
    steering: float
    reason: str
    confidence: float
    ground_safe: bool
    front_clear: bool
    mode: ProcessingMode


class IntegratedDecisionMaker:
    """
    統合判定クラス

    地面カメラ（セグメンテーション）と前方カメラ（深度 or ライン検出）の
    結果を統合し、運転判断を行う。

    判断優先度:
        1. 緊急停止（地面が危険 or 前方に障害物）
        2. 地面の安全性チェック
        3. 前方チェック（障害物 or ライン）
        4. 通常走行

    使用例:
        decision_maker = IntegratedDecisionMaker(config)

        # Mode A: 障害物回避
        decision = decision_maker.decide_obstacle_avoidance(
            segmentation_result,
            depth_result
        )

        # Mode B: ライントレース
        decision = decision_maker.decide_line_following(
            segmentation_result,
            line_result
        )
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定（decision セクション）
        """
        self.config = config.get('decision', {})

        # 閾値
        self.ground_safe_threshold = self.config.get('ground_safe_threshold', 0.6)
        self.obstacle_distance_threshold = self.config.get('obstacle_distance_threshold', 30.0)  # cm
        self.line_confidence_threshold = self.config.get('line_confidence_threshold', 0.3)

        # 速度設定
        self.normal_speed = self.config.get('normal_speed', 0.3)
        self.slow_speed = self.config.get('slow_speed', 0.15)

        # ステアリングゲイン
        self.steering_gain = self.config.get('steering_gain', 0.5)

    def decide_obstacle_avoidance(
        self,
        segmentation_result: Dict,
        depth_result: Dict
    ) -> DrivingDecision:
        """
        Mode A: 障害物回避モードの判断

        Args:
            segmentation_result: セグメンテーション結果
                {
                    'mask': np.ndarray,  # (H, W) uint8
                    'class_areas': {0: area, 1: area, 2: area},
                    'inference_time_ms': float
                }
            depth_result: 深度推定結果
                {
                    'depth_map': np.ndarray,  # (H, W) float32
                    'obstacle_analysis': {
                        'grid': np.ndarray,  # (3, 3) float32, 各グリッドの平均距離
                        'warning_level': int,  # 0-3
                        'recommended_direction': str,  # 'left', 'right', 'stop'
                    },
                    'inference_time_ms': float
                }

        Returns:
            decision: 運転判断結果
        """
        # 1. 地面の安全性チェック
        ground_safe, ground_ratio = self._check_ground_safety(segmentation_result)

        # 2. 前方障害物チェック
        obstacle_analysis = depth_result.get('obstacle_analysis', {})
        warning_level = obstacle_analysis.get('warning_level', 0)
        recommended_direction = obstacle_analysis.get('recommended_direction', 'center')
        grid = obstacle_analysis.get('grid', np.zeros((3, 3)))

        # 3. 判断ロジック

        # 緊急停止条件
        if not ground_safe:
            return DrivingDecision(
                command=DrivingCommand.EMERGENCY_STOP,
                speed=0.0,
                steering=0.0,
                reason=f"地面が危険（通行可能領域: {ground_ratio*100:.1f}%）",
                confidence=1.0 - ground_ratio,
                ground_safe=False,
                front_clear=False,
                mode=ProcessingMode.OBSTACLE_AVOIDANCE
            )

        if warning_level >= 3:
            return DrivingDecision(
                command=DrivingCommand.EMERGENCY_STOP,
                speed=0.0,
                steering=0.0,
                reason=f"前方に障害物（警告レベル: {warning_level}）",
                confidence=1.0,
                ground_safe=True,
                front_clear=False,
                mode=ProcessingMode.OBSTACLE_AVOIDANCE
            )

        # 減速・回避
        if warning_level == 2:
            # 中レベル警告: 減速して回避
            steering = self._calculate_avoidance_steering(recommended_direction)

            return DrivingDecision(
                command=DrivingCommand.SLOW,
                speed=self.slow_speed,
                steering=steering,
                reason=f"障害物を回避中（方向: {recommended_direction}）",
                confidence=0.7,
                ground_safe=True,
                front_clear=False,
                mode=ProcessingMode.OBSTACLE_AVOIDANCE
            )

        if warning_level == 1:
            # 低レベル警告: 軽微な回避
            steering = self._calculate_avoidance_steering(recommended_direction) * 0.5

            return DrivingDecision(
                command=DrivingCommand.GO,
                speed=self.normal_speed * 0.8,
                steering=steering,
                reason=f"軽微な障害物（方向: {recommended_direction}）",
                confidence=0.5,
                ground_safe=True,
                front_clear=True,
                mode=ProcessingMode.OBSTACLE_AVOIDANCE
            )

        # 通常走行
        return DrivingDecision(
            command=DrivingCommand.GO,
            speed=self.normal_speed,
            steering=0.0,
            reason="前方クリア",
            confidence=0.9,
            ground_safe=True,
            front_clear=True,
            mode=ProcessingMode.OBSTACLE_AVOIDANCE
        )

    def decide_line_following(
        self,
        segmentation_result: Dict,
        line_result: Dict
    ) -> DrivingDecision:
        """
        Mode B: ライントレースモードの判断

        Args:
            segmentation_result: セグメンテーション結果
            line_result: ライン検出結果
                {
                    'white_mask': np.ndarray,
                    'yellow_mask': np.ndarray,
                    'combined_mask': np.ndarray,
                    'lines': List[Tuple[int, int, int, int]],
                    'target_point': Optional[Tuple[int, int]],
                    'steering_offset': float,  # -1.0 ~ 1.0
                    'confidence': float,  # 0.0 ~ 1.0
                }

        Returns:
            decision: 運転判断結果
        """
        # 1. 地面の安全性チェック
        ground_safe, ground_ratio = self._check_ground_safety(segmentation_result)

        # 2. ライン検出結果
        steering_offset = line_result.get('steering_offset', 0.0)
        confidence = line_result.get('confidence', 0.0)
        target_point = line_result.get('target_point')

        # 3. 判断ロジック

        # 緊急停止条件
        if not ground_safe:
            return DrivingDecision(
                command=DrivingCommand.EMERGENCY_STOP,
                speed=0.0,
                steering=0.0,
                reason=f"地面が危険（通行可能領域: {ground_ratio*100:.1f}%）",
                confidence=1.0 - ground_ratio,
                ground_safe=False,
                front_clear=False,
                mode=ProcessingMode.LINE_FOLLOWING
            )

        # ライン未検出
        if confidence < self.line_confidence_threshold or target_point is None:
            return DrivingDecision(
                command=DrivingCommand.STOP,
                speed=0.0,
                steering=0.0,
                reason=f"ライン未検出（信頼度: {confidence:.2f}）",
                confidence=0.0,
                ground_safe=True,
                front_clear=True,
                mode=ProcessingMode.LINE_FOLLOWING
            )

        # ライントレース
        steering = steering_offset * self.steering_gain
        steering = np.clip(steering, -1.0, 1.0)

        # ステアリング量に応じて速度調整
        if abs(steering) > 0.5:
            # 急カーブ: 減速
            speed = self.slow_speed
            command = DrivingCommand.SLOW
            reason = f"急カーブ（ステアリング: {steering:.2f}）"
        else:
            # 通常走行
            speed = self.normal_speed
            command = DrivingCommand.GO
            reason = f"ライン追従中（オフセット: {steering_offset:.2f}）"

        return DrivingDecision(
            command=command,
            speed=speed,
            steering=steering,
            reason=reason,
            confidence=confidence,
            ground_safe=True,
            front_clear=True,
            mode=ProcessingMode.LINE_FOLLOWING
        )

    def _check_ground_safety(self, segmentation_result: Dict) -> Tuple[bool, float]:
        """
        地面の安全性をチェック

        Args:
            segmentation_result: セグメンテーション結果

        Returns:
            (safe, ratio): 安全性と通行可能領域の割合
        """
        mask = segmentation_result.get('mask')

        if mask is None:
            return False, 0.0

        # クラス2（通行可能領域）のピクセル数
        class_areas = segmentation_result.get('class_areas', {})
        passable_area = class_areas.get(2, 0)

        # 全体に対する割合
        total_area = mask.shape[0] * mask.shape[1]
        passable_ratio = passable_area / total_area if total_area > 0 else 0.0

        # 閾値判定
        safe = passable_ratio >= self.ground_safe_threshold

        return safe, passable_ratio

    def _calculate_avoidance_steering(self, recommended_direction: str) -> float:
        """
        回避方向からステアリング値を計算

        Args:
            recommended_direction: 推奨方向 ('left', 'right', 'center', 'stop')

        Returns:
            steering: ステアリング値 (-1.0 ~ 1.0)
        """
        if recommended_direction == 'left':
            return -0.5  # 左に回避
        elif recommended_direction == 'right':
            return 0.5  # 右に回避
        elif recommended_direction == 'center':
            return 0.0  # 直進
        else:  # 'stop'
            return 0.0

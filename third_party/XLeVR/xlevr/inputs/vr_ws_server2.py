"""
VR WebSocket server for receiving controller data from web browsers.
Refactored for Stability: "One Frame = One Goal" principle.
"""

import asyncio
import json
import ssl
import websockets
import numpy as np
import math
import logging
from typing import Dict, Optional, Set, List
from scipy.spatial.transform import Rotation as R

from .base import BaseInputProvider, ControlGoal, ControlMode
from ..config import XLeVRConfig

logger = logging.getLogger(__name__)

class VRControllerState:
    """State tracking for a VR controller."""
    def __init__(self, hand: str):
        self.hand = hand
        # 记录上一帧的按键状态用于去抖动或边缘检测（如果需要）
        self.last_buttons = {}
        # 记录原点用于重置/校准
        self.origin_position = None
        self.origin_quaternion = None

    def reset_origin(self):
        self.origin_position = None
        self.origin_quaternion = None

class VRWebSocketServer(BaseInputProvider):
    """WebSocket server for VR controller input."""
    
    def __init__(self, command_queue: asyncio.Queue, config: XLeVRConfig, print_only: bool = False):
        super().__init__(command_queue)
        self.config = config
        self.clients: Set = set()
        self.server = None
        self.print_only = print_only
        
        # Controller states
        self.left_controller = VRControllerState("left")
        self.right_controller = VRControllerState("right")
        
    async def start(self):
        if not self.config.enable_vr:
            logger.info("VR WebSocket server disabled in configuration")
            return

        host = '0.0.0.0'
        port = self.config.websocket_port

        # 禁用 SSL (根据你的需求)
        self.server = await websockets.serve(
            self.websocket_handler,
            host,
            port,
            max_size=None, # 允许大数据包
            ping_interval=None # 防止意外断开
        )
        self.is_running = True
        logger.info(f"VR WebSocket server running on ws://{host}:{port}")

    async def stop(self):
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("VR WebSocket server stopped")
    
    async def websocket_handler(self, websocket, path=None):
        """Handle WebSocket connections."""
        client_address = websocket.remote_address
        logger.info(f"VR client connected: {client_address}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_frame(data)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error processing VR frame: {e}")
        except Exception as e:
            logger.warning(f"VR client disconnected: {e}")
        finally:
            self.clients.discard(websocket)

    async def process_frame(self, data: Dict):
        """
        处理一帧完整的 VR 数据。
        原则：一帧数据生成并发送一次 ControlGoal，包含所有状态。
        """
        
        # 1. 处理头显 (Headset) - 通常用于调整视角，优先级较低
        if 'headset' in data:
            # 这里可以根据需要实现，暂时省略以保持主逻辑清晰
            pass

        # 2. 处理手柄
        if 'leftController' in data:
            await self.process_single_controller('left', data['leftController'])
        
        if 'rightController' in data:
            await self.process_single_controller('right', data['rightController'])

    async def process_single_controller(self, hand: str, data: Dict):
        """
        解析单个手柄的所有数据，打包成唯一的 ControlGoal
        """
        #controller = self.left_controller if hand == 'left' else self.right_controller
        
        # --- 1. 提取基础输入数据 ---
        position = data.get('position', {})
        rotation = data.get('rotation', {})
        quaternion = data.get('quaternion', {}) 
        trigger = data.get('trigger', 0.0)
        #grip = data.get('grip', 0.0) # 有些手柄区分 grip 和 trigger
        grip_active = data.get('gripActive', False)
        thumbstick = data.get('thumbstick', {'x': 0, 'y': 0})
        buttons_raw = data.get('buttons', {})

        # --- 2. 处理按钮信息 (Metadata) ---
        # 提取所有被按下的按钮名称
        active_buttons = [k for k, v in buttons_raw.items() if v]
        
        # 检测特定功能键 (例如 "squeeze" 或 "grip" 用于离合/使能)
        # 假设: trigger > 0.5 视为闭合夹爪
        # 假设: squeeze/grip 键视为 "允许移动" (Clutch)
        is_trigger_pressed = trigger > 0.5
        #is_grip_pressed = buttons_raw.get('squeeze', False) or buttons_raw.get('grip', False) or (grip > 0.5)
        is_grip_pressed = grip_active

        # 打印调试信息 (仅当有动作时)
        if self.print_only and (active_buttons or abs(thumbstick['x']) > 0.1):
            print(f"[{hand.upper()}] Stick:({thumbstick['x']:.2f}, {thumbstick['y']:.2f}) Btn:{active_buttons}")

        # --- 3. 姿态计算 (Position & Rotation) ---
        target_pos_array = None
        wrist_roll = 0.0
        wrist_flex = 0.0
        
        # 只有当位置数据有效时才计算
        if position and all(k in position for k in ['x', 'y', 'z']):
            # 原始位置 (VR 坐标系)
            raw_pos = np.array([position['x'], position['y'], position['z']])
            
            # 转换位置: 缩放
            # 注意: 这里直接使用绝对位置映射。如果需要"相对原点"逻辑，
            # 可以在这里减去 controller.origin_position
            target_pos_array = raw_pos * self.config.vr_to_robot_scale

            # 姿态计算 (优先使用四元数)
            current_quat = None
            if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                current_quat = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
            elif rotation:
                current_quat = self.euler_to_quaternion(rotation)

            if current_quat is not None:
                # 这里我们需要一个基准姿态来计算手腕角度
                # 简单起见，这里演示直接从四元数提取欧拉角
                # 实际应用中，你可能需要根据 robot 的末端执行器定义来调整轴向
                
                # 将 VR 手柄的姿态 (通常是 Grip Pose) 映射到 机器人 TCP 姿态
                # 这是一个简化的映射，具体轴向可能需要根据你的 Simulation 环境调整
                r = R.from_quat(current_quat)
                euler = r.as_euler('xyz', degrees=True)
                
                # 假设 VR 的 Pitch 对应 Wrist Flex, Roll 对应 Wrist Roll
                # 注意正负号可能需要反转
                wrist_flex = euler[0] 
                wrist_roll = euler[1] # 或者 euler[2] 取决于手柄握持方式

        # --- 4. 统一构建 ControlGoal ---
        # 无论数据如何，只要手柄在线，就发送状态，保持数据流连续
        
        # 如果位置丢失，保持 Mode 为 IDLE 或者保持上一次状态
        mode = ControlMode.POSITION_CONTROL if target_pos_array is not None else ControlMode.IDLE
        
        # 构造 Metadata，包含所有按钮状态
        metadata = {
            "source": "vr_unified",
            "buttons": active_buttons,        # 列表: ['trigger', 'a', 'b']
            "buttons_raw": buttons_raw,       # 字典: {'trigger': True, ...}
            "thumbstick": thumbstick,         # 字典: {'x': 0.5, 'y': 0.1}
            "trigger_value": trigger,         # 浮点数值
            "raw_position": [position.get('x'), position.get('y'), position.get('z')],
            "squeeze": is_grip_pressed,        # 方便上层直接读取
            "quaternion": quaternion
        }

        goal = ControlGoal(
            arm=hand,
            mode=mode,
            
            # 位置信息 (如果是 None，上层应忽略或保持原地)
            target_position=target_pos_array if target_pos_array is not None else None,
            
            # 旋转信息
            wrist_roll_deg=wrist_roll,
            wrist_flex_deg=wrist_flex,
            
            # 夹爪状态 (基于 Trigger)
            # 逻辑: Trigger 按下(>0.5) = 闭合(True)，松开 = 打开(False)
            # 或者根据你的习惯反过来
            gripper_closed=is_trigger_pressed, 
            
            # 附加信息
            metadata=metadata
        )

        await self.send_goal(goal)

    # --- 辅助函数 ---
    def euler_to_quaternion(self, euler_deg: Dict[str, float]) -> np.ndarray:
        try:
            euler_rad = [
                math.radians(euler_deg.get('x', 0)), 
                math.radians(euler_deg.get('y', 0)), 
                math.radians(euler_deg.get('z', 0))
            ]
            return R.from_euler('xyz', euler_rad).as_quat()
        except Exception:
            return np.array([0, 0, 0, 1])

    async def send_goal(self, goal: ControlGoal):
        if self.print_only:
            # 简化打印，避免刷屏，只打印关键变化
            pass 
        else:
            await super().send_goal(goal)
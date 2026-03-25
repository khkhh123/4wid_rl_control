import sys
import socket
import struct
import asyncio
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# CarMaker API 경로
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi

class CarMaker4WIDEnv(gym.Env):
    def __init__(self, app, variation):
        super().__init__()
        self.app = app
        self.variation = variation
        self.simcontrol = None
        self.client_sock = None
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('127.0.0.1', 5555))
        self.server_sock.listen(1)

    # [수정] async def로 변경하고 loop 호출 제거
    async def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.simcontrol:
            await self.simcontrol.stop_sim()

        if self.simcontrol is None:
            print("Initial CimControl Setup...")
            self.simcontrol = cmapi.SimControlInteractive()
            await self.simcontrol.set_master(self.app)
            await self.simcontrol.start_and_connect()
        
        if self.client_sock:
            # Non-blocking 모드로 잠시 전환하여 버퍼가 빌 때까지 읽음
            self.client_sock.setblocking(False)
            try:
                while True:
                    # 1024바이트씩 계속 읽어서 버림 (더 이상 읽을 게 없을 때까지)
                    data = self.client_sock.recv(1024)
                    if not data: break 
            except (BlockingIOError, OSError):
                # 버퍼가 텅 비면 여기로 넘어옵니다.
                pass
            
            # 깨끗해진 소켓을 닫음
            self.client_sock.close()
            self.client_sock = None
            print("Socket buffer flushed and connection closed.")

        self.simcontrol.set_variation(self.variation.clone())

        print("befor start_sim")        
        await self.simcontrol.start_sim()
        print("after start_sim")        

        if self.client_sock is None:
            print("Waiting for User.c connection...")
            print(self.simcontrol.get_status())
            self.client_sock, _ = self.server_sock.accept()
            
        raw_obs = self.client_sock.recv(24)
        data = struct.unpack('ddd', raw_obs) # (curr_v, v_diff, sim_state)
        init_action = struct.pack('dddd', 0.0, 0.0, 0.0, 0.0)
        self.client_sock.sendall(init_action)

        # [수정] AI에게는 앞의 2개(v, v_diff)만 전달
        obs = np.array(data[:2], dtype=np.float32)
        
        return np.array(obs, dtype=np.float32), {}

    async def step(self, action):
        # 1. 상태 수신 (User.c가 send한 24바이트 수신: double 3개)
        # 이제 여기서 엔진 상태(sim_state)까지 한꺼번에 받습니다.
        raw_data = self.client_sock.recv(24) 
        if not raw_data:
            return np.zeros(2), 0, True, False, {}

        curr_v, v_diff, sim_state = struct.unpack('ddd', raw_data)

        # 보상 및 종료 조건 판단
        reward = -abs(v_diff*v_diff) -10 * np.sum(action**2)
        terminated = False
        
        if sim_state != 8.0:
            terminated = True
            print("Terminatingggggg")
        else:
            # 액션 전송
            data = struct.pack('dddd', *action)
            self.client_sock.sendall(data)  
            # 종료 시에만 비동기로 제어 명령 전송 (루프 정체 방지)
            # await self.simcontrol.stop_sim()
        #     # disconnect는 reset 시작 시점에서 처리하는 것이 가장 안전함

        obs = np.array([curr_v, v_diff], dtype=np.float32)
        return obs, reward, terminated, False, {}

    async def close(self):
        if self.simcontrol:
            await self.simcontrol.stop_and_disconnect()
        if self.client_sock:
            self.client_sock.close()
        self.server_sock.close()
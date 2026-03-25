import sys, socket, struct, asyncio, numpy as np
import gymnasium as gym
from gymnasium import spaces

# CarMaker API 경로 (사용자 환경에 맞게 유지)
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi

class CarMaker4WIDEnv(gym.Env):
    def __init__(self, app, variation, loop):
        super().__init__()
        self.app = app
        self.variation = variation
        self.loop = loop  # 주입받은 루프 저장
        self.simcontrol = None
        self.client_sock = None
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # 서버 소켓 초기 설정
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('127.0.0.1', 5555))
        self.server_sock.listen(1)
        # accept 시 너무 오래 대기하지 않도록 타임아웃 설정 (선택 사항)
        self.server_sock.settimeout(10.0) 

    async def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. 기존 시뮬레이션 및 소켓 정리
        if self.simcontrol:
            await self.simcontrol.stop_sim()
        
        if self.client_sock:
            try:
                self.client_sock.close()
            except: pass
            self.client_sock = None

        # 2. SimControl 초기화 (최초 1회)
        if self.simcontrol is None:
            self.simcontrol = cmapi.SimControlInteractive()
            await self.simcontrol.set_master(self.app)
            await self.simcontrol.start_and_connect()
        
        # 3. 테스트런 설정 및 시작
        self.simcontrol.set_variation(self.variation.clone())
        await self.simcontrol.start_sim()
        await asyncio.sleep(0.5)
        # 4. User.c 연결 수락 (Blocking 구간)
        # run_in_executor를 사용하면 accept 대기 중에도 루프가 멈추지 않습니다.
        print("Waiting for User.c connection...")
        self.client_sock, _ = await self.loop.run_in_executor(None, self.server_sock.accept)
        
        # 5. 초기 데이터 교환 (Handshake)
        # User.c가 보낸 초기 24바이트(v, v_diff, state)를 즉시 읽어 병목 해소
        raw_obs = await self.loop.run_in_executor(None, self.client_sock.recv, 24)
        curr_v, v_diff, _ = struct.unpack('ddd', raw_obs)
        
        # 초기 액션 전송
        init_action = struct.pack('dddd', 0.0, 0.0, 0.0, 0.0)
        self.client_sock.sendall(init_action)

        return np.array([curr_v, v_diff], dtype=np.float32), {}

    async def step(self, action):
        # 1. 데이터 수신 (Blocking 방지를 위해 executor 사용 권장)
        raw_data = await self.loop.run_in_executor(None, self.client_sock.recv, 24)
        if not raw_data or len(raw_data) < 24:
            return np.zeros(2), 0, True, False, {}

        curr_v, v_diff, sim_state = struct.unpack('ddd', raw_data)

        # 2. 보상 계산
        reward = -float(v_diff**2) - 0.1 * np.sum(action**2)

        # 3. 종료 조건 처리 (State 8이 아니면 종료)
        terminated = (sim_state != 8.0)
        
        if not terminated:
            # 액션 전송
            data = struct.pack('dddd', *map(float, action))
            self.client_sock.sendall(data)
        else:
            print("Simulation finished or interrupted.")

        obs = np.array([curr_v, v_diff], dtype=np.float32)
        return obs, reward, terminated, False, {}

    async def close(self):
        if self.simcontrol:
            await self.simcontrol.stop_and_disconnect()
        if self.client_sock:
            self.client_sock.close()
        self.server_sock.close()
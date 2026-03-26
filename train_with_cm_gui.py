import sys, asyncio, threading, struct, socket, psutil
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi


class SaveBestEpisodeRewardCallback(BaseCallback):
    def __init__(self, best_model_path, verbose=1):
        super().__init__(verbose)
        self.best_model_path = best_model_path
        self.current_episode_reward = 0.0
        self.best_episode_reward = -np.inf

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        self.current_episode_reward += float(rewards[0])
        if bool(dones[0]):
            if self.current_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.current_episode_reward
                self.model.save(self.best_model_path)
                if self.verbose:
                    print(f"[BEST] episode_reward={self.best_episode_reward:.3f} -> {self.best_model_path}")
            self.current_episode_reward = 0.0
        return True

def get_carmaker_pid():
    target = "CarMaker.linux64"
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if target in proc.info['name'] or any(target in arg for arg in (proc.info['cmdline'] or [])):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    return None

# --- 1. 독립적인 강화학습 환경 (데이터 전달 역할) ---
class CarMaker4WIDEnv(gym.Env):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.client_sock = None
        self.last_obs = None
        
        # 지휘자와 소통하기 위한 이벤트
        self.reset_req = asyncio.Event()
        self.ready_evt = asyncio.Event()
        self.stop_req = asyncio.Event() # 종료 신호
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    async def reset(self, seed=None, options=None):
        # 지휘자에게 리셋 요청 후 준비될 때까지 대기
        self.reset_req.set()
        await self.ready_evt.wait()
        self.ready_evt.clear()
        return np.array(self.last_obs, dtype=np.float32), {}

    async def step(self, action):
        # 기존 step 로직: 소켓을 통해 데이터 주고받기
        try:
            # 액션 전송
            data = struct.pack('dddd', *map(float, action))
            await self.loop.run_in_executor(None, self.client_sock.sendall, data)
            
            # 다음 상태 수신
            raw_data = await self.loop.run_in_executor(None, self.client_sock.recv, 24)
            if not raw_data or len(raw_data) < 24:
                return np.zeros(2), 0, True, False, {}

            curr_v, v_diff, sim_state = struct.unpack('ddd', raw_data)
            self.last_obs = [curr_v, v_diff]
            
            reward = -float(v_diff**2) - 100.0 * np.sum(action**2)
            print("v_diff penalty: ", v_diff**2)
            print("action penalty: ", np.sum(action**2))

            terminated = (sim_state != 8.0)
            
            return np.array(self.last_obs, dtype=np.float32), reward, terminated, False, {}
        except Exception as e:
            return np.zeros(2), 0, True, False, {}

# --- 2. 카메이커 지휘자 (예제 형태 유지 및 생명주기 관리) ---
async def carmaker_orchestrator(env, simcontrol, variation, server_sock):
    loop = asyncio.get_running_loop()
    
    while True:
        # RL의 리셋 신호 대기
        await env.reset_req.wait()
        env.reset_req.clear()

        await simcontrol.connect()
        
        print("\n[Orchestrator] Starting New Episode...")
        
        # 카메이커 표준 예제 흐름
        if simcontrol.get_status() != cmapi.SimControlState.configure:
            await simcontrol.stop_sim()
            await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

        simcontrol.set_variation(variation.clone())
        await simcontrol.start_sim()
        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.running).wait()
        
        # 소켓 연결 수락 및 초기화
        env.client_sock, _ = await loop.run_in_executor(None, server_sock.accept)
        raw_obs = await loop.run_in_executor(None, env.client_sock.recv, 24)
        
        # 첫 관측값 설정 후 Env 깨우기
        env.last_obs = struct.unpack('ddd', raw_obs)[:2]
        env.ready_evt.set()
        
        # 시뮬레이션이 종료(Idle)될 때까지 배경에서 대기 및 감시
        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()
        
        if env.client_sock:
            env.client_sock.close()
            env.client_sock = None
        print("[Orchestrator] Episode Finished & Cleaned up.")
        await simcontrol.disconnect()

# --- 3. 동기식 브릿지 및 메인 함수 ---
class SyncBridge(gym.Env):
    def __init__(self, a_env, loop):
        self.a_env, self.loop = a_env, loop
        self.action_space, self.observation_space = a_env.action_space, a_env.observation_space
    def reset(self, **kwargs): return asyncio.run_coroutine_threadsafe(self.a_env.reset(**kwargs), self.loop).result()
    def step(self, action):    return asyncio.run_coroutine_threadsafe(self.a_env.step(action), self.loop).result()
    def close(self):           pass

def run_learning(env):
    log_dir = "./ppo_carmaker_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    model_path = "carmaker_ppo_4wid2.zip"
    best_model_path = "carmaker_ppo_4wid2_best.zip"
    interrupt_model_path = "carmaker_ppo_4wid2_interrupt.zip"
    callback = SaveBestEpisodeRewardCallback(best_model_path)

    # 1. 기존 모델이 있는지 확인
    if os.path.exists(model_path):
        print(f"--- 기존 모델 '{model_path}'을(를) 불러와서 학습을 재개합니다. ---")
        # 저장된 모델 로드
        model = PPO.load(model_path, env=env, device="cpu", verbose=1)
    else:
        print("--- 기존 모델이 없습니다. 새로운 모델을 생성합니다. ---")
        # 새로 시작할 경우
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    # 모델 업데이트 빈도를 줄여 시뮬레이션 버벅거림을 완화하기 위해 n_steps를 늘립니다.
    # n_steps는 에이전트가 환경에서 데이터를 수집하는 스텝 수입니다. 이 값을 늘리면 업데이트가 덜 자주 발생합니다.
    model.n_steps = 4096*2 # 기본값 2048에서 4096으로 증가 (필요에 따라 더 늘릴 수 있습니다)
    model.batch_size = 256 # batch_size도 함께 조정하여 효율적인 학습을 유도할 수 있습니다.
    model.tensorboard_log=log_dir
    model.learning_rate = 1e-4
    print("clip_range: ", model.clip_range)
    try:
        model.learn(total_timesteps=100000, callback=callback)
    except KeyboardInterrupt:
        print("학습 중단 요청됨. 모델 저장 중...")
        model.save(interrupt_model_path)
        print(f"중단 시점 모델이 '{interrupt_model_path}'에 저장되었습니다.")
    finally:
        model.save(model_path)
        print(f"모델이 '{model_path}'에 저장되었습니다.")

async def main():
    # CarMaker 초기화 로직 (기존 main 내용)
    proj_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(proj_path)
    
    # 서버 소켓 준비
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('127.0.0.1', 5556))
    server_sock.listen(1)

    project_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(project_path)
    
    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=get_carmaker_pid(), description="Idle"))
    master.set_host("localhost")
    # master = cmapi.CarMaker()
    # master.set_executable_path(project_path / "src/CarMaker.linux64")

    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(master)
    # ApoServer 설정 및 master 연결 (기존 코드와 동일)
    # master = ... / await simcontrol.set_master(master) / await simcontrol.start_and_connect()
    
    testrun = cmapi.Project.instance().load_testrun_parametrization(proj_path / "Data/TestRun/testrun_test1")
    variation = cmapi.Variation.create_from_testrun(testrun)

    loop = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop)
    
    # 지휘자 가동 (백그라운드)
    cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))
    
    # 학습 시작
    env_sync = SyncBridge(env_async, loop)
    await loop.run_in_executor(None, run_learning, env_sync)

    print("Learning finished. Sending stop signal to orchestrator...")
    env_async.stop_req.set()
    loop.stop()

if __name__ == "__main__":
    cmapi.Task.run_main_task(main())
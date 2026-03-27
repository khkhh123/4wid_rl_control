import sys, asyncio, threading, struct, socket, psutil
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi

# 액션 크기 및 바퀴 간 불균형 패널티 가중치
ACTION_VELOCITY_WEIGHT = float(os.getenv("ACTION_VELOCITY_WEIGHT", "10.0"))
ACTION_EFFORT_WEIGHT = float(os.getenv("ACTION_EFFORT_WEIGHT", "1"))
ACTION_BALANCE_WEIGHT = 0.0  # balance penalty disabled


SAC_PRESETS = {
    # 보수적: 안정성 우선
    "A": {
        "learning_rate": 1e-4,
        "buffer_size": 200000,
        "learning_starts": 10000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (64, "step"),
        "gradient_steps": 64,
        "ent_coef": "auto_0.1",
    },
    # 기본: 균형형
    "B": {
        "learning_rate": 3e-4,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (64, "step"),
        "gradient_steps": 64,
        "ent_coef": "auto",
    },
    # 공격적: 환경 샘플당 업데이트 비율 증가
    "C": {
        "learning_rate": 1.0e-3,
        "buffer_size": 100000000,
        "learning_starts": 5000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (128*32, "step"),
        "gradient_steps": 128,
        "ent_coef": "auto",
    }
}

class SaveBestEpisodeRewardCallback(BaseCallback):
    def __init__(self, best_model_path, verbose=1):
        super().__init__(verbose)
        self.best_model_path = best_model_path
        self.best_reward_path = best_model_path + ".reward"
        self.current_episode_reward = 0.0
        self.best_episode_reward = -np.inf
        # 파일에서 best reward 불러오기
        if os.path.exists(self.best_reward_path):
            try:
                with open(self.best_reward_path, "r") as f:
                    self.best_episode_reward = float(f.read().strip())
            except Exception:
                self.best_episode_reward = -np.inf

    def _save_best_reward(self):
        try:
            with open(self.best_reward_path, "w") as f:
                f.write(str(self.best_episode_reward))
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to save best reward: {e}")

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
                self._save_best_reward()
                if self.verbose:
                    print(f"[BEST] episode_reward={self.best_episode_reward:.3f} -> {self.best_model_path}")
            self.current_episode_reward = 0.0
        return True


def get_sac_profile():
    # 우선순위: 환경변수 > CLI 인자 > 기본값(B)
    profile = os.getenv("SAC_PROFILE")
    if not profile and len(sys.argv) > 1:
        profile = sys.argv[1]

    profile = (profile or "B").upper()
    if profile not in SAC_PRESETS:
        print(f"알 수 없는 프로필 '{profile}' 입니다. B 프로필로 진행합니다. (사용 가능: A/B/C)")
        profile = "B"
    return profile


def get_carmaker_pid():
    target = "CarMaker_5555.linux64"
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if target in proc.info['name'] or any(target in arg for arg in (proc.info['cmdline'] or [])):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


# --- 1. 독립적인 강화학습 환경 (데이터 전달 역할) ---

class CarMaker4WIDEnv(gym.Env):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.client_sock = None
        self.last_obs = None
        self.episode_count = 0
        self.step_count = 0
        self.max_steps = int(os.getenv("EPISODE_MAX_STEPS", "500"))
        self.early_done_penalty = float(os.getenv("EARLY_DONE_PENALTY", "-1000000.0"))

        # 지휘자와 소통하기 위한 이벤트
        self.reset_req = asyncio.Event()
        self.ready_evt = asyncio.Event()
        self.stop_req = asyncio.Event()  # 종료 신호

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    async def reset(self, seed=None, options=None):
        self.episode_count += 1
        self.step_count = 0
        self.reset_req.set()
        await self.ready_evt.wait()
        self.ready_evt.clear()
        print(f"[EPISODE] {self.episode_count} started")
        return np.array(self.last_obs, dtype=np.float32), {}

    def action_to_torques(self, action):
        """Convert action [demand_ratio, left_ratio, right_ratio] to wheel torques [FL, FR, RL, RR].
        
        Args:
            action: shape (3,)
                action[0]: base torque demand (-1 ~ 1), sign determines L/R, abs determines magnitude
                action[1]: left wheel front/rear ratio (-1: front 100%, 1: rear 100%)
                action[2]: right wheel front/rear ratio (-1: front 100%, 1: rear 100%)
        
        Returns:
            torques: [FL, FR, RL, RR] shape (4,)
        """
        demand_ratio, left_ratio, right_ratio = action
        base_torque = abs(demand_ratio) * 0.1  # Scale to [0, 0.1]
        lr_ratio = (demand_ratio + 1) / 2  # Convert [-1, 1] to [0, 1]: 0=left, 1=right
        
        left_torque = base_torque * (1 - lr_ratio)
        right_torque = base_torque * lr_ratio
        
        # Front/rear distribution: -1 = front 100%, 1 = rear 100%
        fl = left_torque * (1 - left_ratio) / 2
        rl = left_torque * (1 + left_ratio) / 2
        fr = right_torque * (1 - right_ratio) / 2
        rr = right_torque * (1 + right_ratio) / 2
        
        return np.array([fl, fr, rl, rr], dtype=np.float32)

    async def step(self, action):
        try:
            self.step_count += 1
            torques = self.action_to_torques(action)
            data = struct.pack('dddd', *map(float, torques))
            await self.loop.run_in_executor(None, self.client_sock.sendall, data)

            raw_data = await self.loop.run_in_executor(None, self.client_sock.recv, 24)
            if not raw_data or len(raw_data) < 24:
                return np.zeros(2), 0, True, False, {}

            curr_v, v_diff, sim_state = struct.unpack('ddd', raw_data)
            self.last_obs = [curr_v, v_diff]

            action = np.asarray(action, dtype=np.float32)
            velocity_penalty = ACTION_VELOCITY_WEIGHT * np.exp(-float(v_diff**2) / 1600)

            effort_penalty = ACTION_EFFORT_WEIGHT * float(np.sum(action**2))
            # weight_dist = 1.0 / (1.0 + np.exp(v_diff**2))
            weight_dist = 1.0

            reward = velocity_penalty - weight_dist * effort_penalty
            # 지수형 보상 함수
            terminated = (sim_state != 8.0)

            # 조기 종료 패널티 적용
            if terminated and self.step_count < self.max_steps:
                print(f"[EARLY DONE] step={self.step_count} < max_steps={self.max_steps} → penalty {self.early_done_penalty}")
                reward += self.early_done_penalty

            print(f"[STEP] episode={self.episode_count} step={self.step_count}")
            print(f"[ACTION_RAW] demand={action[0]:7.4f} | left_ratio={action[1]:7.4f} | right_ratio={action[2]:7.4f}")
            print(f"[TORQUES] FL={torques[0]:7.4f} | FR={torques[1]:7.4f} | RL={torques[2]:7.4f} | RR={torques[3]:7.4f}")
            print(
                f"[REWARD] V_diff: {velocity_penalty:9.4f}"
                f"| Effort: {effort_penalty:9.4f} | Total: {reward:9.4f}"
            )

            return np.array(self.last_obs, dtype=np.float32), reward, terminated, False, {}
        except Exception:
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
        simcontrol.set_realtimefactor(100.0)
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

    def reset(self, **kwargs):
        return asyncio.run_coroutine_threadsafe(self.a_env.reset(**kwargs), self.loop).result()

    def step(self, action):
        return asyncio.run_coroutine_threadsafe(self.a_env.step(action), self.loop).result()

    def close(self):
        pass


def preload_replay_buffer_from_custom_data(model):
    """Load custom controller transition files and push them into SAC replay buffer.

    Expected transition format per item:
    (obs, action, reward, next_obs, done)
    """
    data_glob = os.getenv("CUSTOM_DATA_GLOB", "custom_episode_*.npy")
    data_files = sorted(Path(".").glob(data_glob))

    if not data_files:
        print(f"[REPLAY PRELOAD] No files found for pattern: {data_glob}")
        return 0

    loaded_count = 0
    for file_path in data_files:
        try:
            arr = np.load(file_path, allow_pickle=True)
            for item in arr:
                obs, action, reward, next_obs, done = item

                obs = np.asarray(obs, dtype=np.float32)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                action = np.asarray(action, dtype=np.float32)
                reward = float(reward)
                done = bool(done)

                model.replay_buffer.add(
                    obs=np.array([obs], dtype=np.float32),
                    next_obs=np.array([next_obs], dtype=np.float32),
                    action=np.array([action], dtype=np.float32),
                    reward=np.array([reward], dtype=np.float32),
                    done=np.array([done], dtype=np.float32),
                    infos=[{}],
                )
                loaded_count += 1
        except Exception as e:
            print(f"[REPLAY PRELOAD][WARN] Failed to load {file_path}: {e}")

    print(f"[REPLAY PRELOAD] Loaded {loaded_count} transitions from {len(data_files)} file(s).")
    return loaded_count


def run_learning(env):
    profile = get_sac_profile()
    sac_kwargs = SAC_PRESETS[profile]
    model_path = f"carmaker_sac_4wid_gui_{profile}.zip"
    best_model_path = f"carmaker_sac_4wid_gui_{profile}_best2.zip"
    interrupt_model_path = f"carmaker_sac_4wid_gui_{profile}_interrupt.zip"
    callback = SaveBestEpisodeRewardCallback(best_model_path)

    print(f"--- SAC 프로필: {profile} ---")
    print(f"--- SAC 설정: {sac_kwargs} ---")

    # if os.path.exists(best_model_path):
    #     print(f"--- 기존 모델 '{best_model_path}'을(를) 불러와서 학습을 재개합니다. ---")
    #     model = SAC.load(best_model_path, env=env, device="cpu", verbose=1)
    if os.path.exists(model_path):
        print(f"--- 기존 모델 '{model_path}'을(를) 불러와서 학습을 재개합니다. ---")
        model = SAC.load(model_path, env=env, device="cpu", verbose=1)
    else:
        print("--- 기존 모델이 없습니다. 새로운 SAC 모델을 생성합니다. ---")
        model = SAC("MlpPolicy", env, verbose=1, device="cpu", **sac_kwargs)

    preload_count = preload_replay_buffer_from_custom_data(model)
    if preload_count > 0:
        print(f"--- replay buffer 사전 주입 완료: {preload_count} transitions ---")

    try:
        model.learn(total_timesteps=1000000, callback=callback)
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
    server_sock.bind(('127.0.0.1', 5555))
    server_sock.listen(1)

    project_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(project_path)

    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=get_carmaker_pid(), description="Idle"))
    master.set_host("localhost")
    
    simcontrol = cmapi.SimControlInteractive()
    
    await simcontrol.set_master(master)
    

    testrun = cmapi.Project.instance().load_testrun_parametrization(project_path / "Data/TestRun/testrun_test1")
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

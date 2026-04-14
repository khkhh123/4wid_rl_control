import sys, asyncio, threading, struct, socket, psutil
import os
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi


# 액션 크기 및 보상 가중치
ACTION_VELOCITY_WEIGHT = float(os.getenv("ACTION_VELOCITY_WEIGHT", "10"))
ACTION_EFFORT_WEIGHT = float(os.getenv("ACTION_EFFORT_WEIGHT", "1.0"))
WHEEL_TORQUE_LIMIT = float(os.getenv("WHEEL_TORQUE_LIMIT", "630.0"))
TOTAL_TORQUE_LIMIT = float(os.getenv("TOTAL_TORQUE_LIMIT", str(WHEEL_TORQUE_LIMIT * 4.0)))
TARGET_SPEED_KPH = float(os.getenv("TARGET_SPEED_KPH", "40.0"))
CONTROL_DT = float(os.getenv("CONTROL_DT", "0.05"))
TARGET_SPEED_MODE = os.getenv("TARGET_SPEED_MODE", "scenario").strip().lower()
SCENARIO_DIR = os.getenv("SCENARIO_DIR", "scenarios").strip()
SCENARIO_CSV_PATH = os.getenv("SCENARIO_CSV_PATH", "").strip()

PROTOCOL_LEGACY = "legacy3"
PROTOCOL_ASSESSMENT = "assessment23"
MODEL_BASENAME = os.getenv("MODEL_BASENAME", "carmaker_ppo_4wid_dyc").strip()


PPO_PRESETS = {
    # 보수적: 안정성 우선
    "A": {
        "learning_rate": 1e-5,
        "batch_size": 256,
        "gamma": 0.99,
        "n_steps": 8192,
        "ent_coef": 0.1,
    },
    # 기본: 균형형
    "B": {
        "learning_rate": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "n_steps": 4096,
        "ent_coef": 0.0,
    },
    # 공격적: 빠른 학습률
    "C": {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "gamma": 0.99,
        "n_steps": 8192,
        "ent_coef": 0.0,
    },
}


class SaveBestEpisodeRewardCallback(BaseCallback):
    def __init__(self, best_model_path, verbose=1):
        super().__init__(verbose)
        self.best_model_path = best_model_path
        self.best_reward_path = best_model_path + ".reward"
        self.current_episode_reward = 0.0
        self.best_episode_reward = -np.inf
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


def get_ppo_profile():
    # 우선순위: 환경변수 > CLI 인자 > 기본값(B)
    profile = os.getenv("PPO_PROFILE")
    if not profile and len(sys.argv) > 1:
        profile = sys.argv[1]

    profile = (profile or "B").upper()
    if profile not in PPO_PRESETS:
        print(f"알 수 없는 프로필 '{profile}' 입니다. B 프로필로 진행합니다. (사용 가능: A/B/C)")
        profile = "B"
    return profile

def get_carmaker_pid():
    target = "CarMaker.linux64"
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if target in proc.info['name'] or any(target in arg for arg in (proc.info['cmdline'] or [])):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    return None


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def recv_exact(sock_obj: socket.socket, nbytes: int) -> bytes:
    chunks = []
    received = 0
    while received < nbytes:
        chunk = sock_obj.recv(nbytes - received)
        if not chunk:
            return b""
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def find_scenario_csv_files(base_dir: Path):
    if SCENARIO_CSV_PATH:
        candidate = Path(SCENARIO_CSV_PATH)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        return [candidate] if candidate.exists() else []

    scenario_dir = Path(SCENARIO_DIR)
    if not scenario_dir.is_absolute():
        scenario_dir = base_dir / scenario_dir
    if not scenario_dir.exists():
        return []
    return sorted(p for p in scenario_dir.glob("*.csv") if p.is_file())


def load_speed_profile(csv_path: Path):
    time_vals = []
    speed_vals = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None

        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        time_col = field_map.get("time")
        speed_col = field_map.get("speed")
        if not time_col or not speed_col:
            return None

        for row in reader:
            try:
                t = float(row[time_col])
                v = float(row[speed_col])
            except (ValueError, TypeError, KeyError):
                continue
            time_vals.append(t)
            speed_vals.append(v)

    if len(time_vals) < 2:
        return None
    return np.asarray(time_vals, dtype=np.float64), np.asarray(speed_vals, dtype=np.float64)

# --- 1. 독립적인 강화학습 환경 (데이터 전달 역할) ---
class CarMaker4WIDEnv(gym.Env):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.client_sock = None
        self.last_obs = None
        self.step_count = 0
        self.max_steps = int(os.getenv("EPISODE_MAX_STEPS", "500"))
        self.early_done_penalty = float(os.getenv("EARLY_DONE_PENALTY", "-50.0"))
        self.protocol_mode = os.getenv("CM_PROTOCOL_MODE", PROTOCOL_LEGACY).strip().lower()
        if self.protocol_mode not in {PROTOCOL_LEGACY, PROTOCOL_ASSESSMENT}:
            print(f"[WARN] Unknown CM_PROTOCOL_MODE='{self.protocol_mode}', fallback to '{PROTOCOL_LEGACY}'")
            self.protocol_mode = PROTOCOL_LEGACY
        self.control_dt = max(CONTROL_DT, 1e-3)
        self.prev_yaw = None

        self.target_speed_fixed_kph = TARGET_SPEED_KPH
        self.target_speed_mode = TARGET_SPEED_MODE if TARGET_SPEED_MODE in {"scenario", "fixed"} else "scenario"
        self.base_dir = Path(__file__).resolve().parent
        self.scenario_files = find_scenario_csv_files(self.base_dir)
        self.scenario_idx = -1
        self.current_scenario_path = None
        self.scenario_time = None
        self.scenario_speed = None

        # 지휘자와 소통하기 위한 이벤트
        self.reset_req = asyncio.Event()
        self.ready_evt = asyncio.Event()
        self.stop_req = asyncio.Event()  # 종료 신호

        if self.protocol_mode == PROTOCOL_ASSESSMENT:
            # RL action: [left_total_norm, left_ratio, right_ratio]
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            # Minimal observation subset: v + wheel_rotv(4) + yaw_rate + ax + ay
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
            self.state_recv_bytes = 23 * 8
            self.action_send_fmt = "ddddd"
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
            self.state_recv_bytes = 24
            self.action_send_fmt = "dddd"

    def _advance_scenario_profile(self):
        if not self.scenario_files:
            self.current_scenario_path = None
            self.scenario_time = None
            self.scenario_speed = None
            return

        self.scenario_idx = (self.scenario_idx + 1) % len(self.scenario_files)
        path = self.scenario_files[self.scenario_idx]
        profile = load_speed_profile(path)
        if profile is None:
            print(f"[WARN] Invalid scenario CSV format: {path}. Fallback to fixed TARGET_SPEED_KPH.")
            self.current_scenario_path = None
            self.scenario_time = None
            self.scenario_speed = None
            return

        self.current_scenario_path = path
        self.scenario_time, self.scenario_speed = profile
        print(f"[SCENARIO] loaded: {path.name} ({len(self.scenario_time)} points)")

    def _target_speed_from_profile(self):
        if self.scenario_time is None or self.scenario_speed is None:
            return self.target_speed_fixed_kph
        t_now = float(self.step_count) * self.control_dt
        return float(np.interp(t_now, self.scenario_time, self.scenario_speed))

    def _target_speed_kph(self):
        if self.target_speed_mode == "fixed":
            return self.target_speed_fixed_kph
        return self._target_speed_from_profile()

    def _parse_assessment_state(self, raw_data: bytes):
        state = struct.unpack("d" * 23, raw_data)
        speed_mps = float(state[3])
        speed_kph = speed_mps * 3.6
        wheel_rotv = [float(state[i]) for i in range(4, 8)]
        yaw = float(state[2])
        if self.prev_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_rate = wrap_to_pi(yaw - self.prev_yaw) / self.control_dt
        self.prev_yaw = yaw
        ax = float(state[21])
        ay = float(state[22])
        obs = [speed_kph, wheel_rotv[0], wheel_rotv[1], wheel_rotv[2], wheel_rotv[3], yaw_rate, ax, ay]
        return state, obs

    def _build_assessment_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        left_total_norm = float(np.clip(action[0], -1.0, 1.0))
        left_ratio = float(np.clip(action[1], 0.0, 1.0))
        right_ratio = float(np.clip(action[2], 0.0, 1.0))

        speed_kph = float(self.last_obs[0]) if self.last_obs is not None else 0.0
        target_speed_kph = self._target_speed_kph()
        v_diff = target_speed_kph - speed_kph
        total_torque = np.clip(v_diff * 80.0, -TOTAL_TORQUE_LIMIT, TOTAL_TORQUE_LIMIT)

        left_total = left_total_norm * float(total_torque)
        right_total = float(total_torque) - left_total

        fl = left_total * left_ratio
        rl = left_total - fl
        fr = right_total * right_ratio
        rr = right_total - fr
        steering_cmd = 0.0

        wheel_torques = np.array([fl, fr, rl, rr], dtype=np.float32)
        return (fl, fr, rl, rr, steering_cmd), wheel_torques, v_diff, target_speed_kph

    async def reset(self, seed=None, options=None):
        # 지휘자에게 리셋 요청 후 준비될 때까지 대기
        self.step_count = 0
        self.prev_yaw = None
        if self.protocol_mode == PROTOCOL_ASSESSMENT and self.target_speed_mode == "scenario":
            self._advance_scenario_profile()
        self.reset_req.set()
        await self.ready_evt.wait()
        self.ready_evt.clear()
        return np.array(self.last_obs, dtype=np.float32), {}

    async def step(self, action):
        # 기존 step 로직: 소켓을 통해 데이터 주고받기
        try:
            self.step_count += 1
            if self.protocol_mode == PROTOCOL_ASSESSMENT:
                action_cmd, wheel_torques, v_diff, target_speed_kph = self._build_assessment_action(action)
            else:
                action = np.asarray(action, dtype=np.float32)
                action_cmd = tuple(map(float, action))

            # 액션 전송
            data = struct.pack(self.action_send_fmt, *action_cmd)
            await self.loop.run_in_executor(None, self.client_sock.sendall, data)

            # 다음 상태 수신
            raw_data = await self.loop.run_in_executor(None, recv_exact, self.client_sock, self.state_recv_bytes)
            if not raw_data or len(raw_data) < self.state_recv_bytes:
                return np.zeros(self.observation_space.shape[0]), 0, True, False, {}

            if self.protocol_mode == PROTOCOL_ASSESSMENT:
                _, obs = self._parse_assessment_state(raw_data)
                self.last_obs = obs
                sim_state = 8.0
                velocity_penalty = ACTION_VELOCITY_WEIGHT * float(v_diff**2) / 1600.0
                effort_penalty = ACTION_EFFORT_WEIGHT * float(np.sum((wheel_torques / WHEEL_TORQUE_LIMIT) ** 2))
            else:
                curr_v, v_diff, sim_state = struct.unpack("ddd", raw_data)
                self.last_obs = [curr_v, v_diff]
                velocity_penalty = ACTION_VELOCITY_WEIGHT * float(v_diff**2) / 1600
                effort_penalty = ACTION_EFFORT_WEIGHT * float(np.sum(action**2))

            reward = -velocity_penalty - effort_penalty
            terminated = sim_state != 8.0
            truncated = self.step_count >= self.max_steps

            # 조기 종료 패널티 적용
            if terminated and not truncated:
                print(f"[EARLY DONE] step={self.step_count} < max_steps={self.max_steps} -> penalty {self.early_done_penalty}")
                reward += self.early_done_penalty

            if self.protocol_mode == PROTOCOL_ASSESSMENT:
                print(
                    f"[ACTION-ASM] FL={action_cmd[0]:9.3f} | FR={action_cmd[1]:9.3f} "
                    f"| RL={action_cmd[2]:9.3f} | RR={action_cmd[3]:9.3f} | STR={action_cmd[4]:7.4f}"
                )
                print(f"[TARGET] mode={self.target_speed_mode} | target_speed_kph={target_speed_kph:7.3f}")
            else:
                print(f"[ACTION] FL={action[0]:7.4f} | FR={action[1]:7.4f} | RL={action[2]:7.4f} | RR={action[3]:7.4f}")
            print(
                f"[REWARD] V_diff: {velocity_penalty:9.4f}"
                f"| Effort: {effort_penalty:9.4f} | Total: {reward:9.4f}"
            )

            return np.array(self.last_obs, dtype=np.float32), reward, terminated, truncated, {}
        except Exception as e:
            print(f"[STEP ERROR] {type(e).__name__}: {e}")
            return np.zeros(self.observation_space.shape[0]), 0, True, False, {}

# --- 2. 카메이커 지휘자 (예제 형태 유지 및 생명주기 관리) ---
async def carmaker_orchestrator(env, simcontrol, variation, server_sock):
    loop = asyncio.get_running_loop()
    while True:
        # RL의 리셋 신호 대기
        await env.reset_req.wait()
        # while not env.stop_req.is_set():
        #     try:
        #         # 리셋 신호를 기다리되, 종료 신호가 오면 즉시 빠져나갈 수 있게 타임아웃을 짧게 줍니다.
        #         await asyncio.wait_for(env.reset_req.wait(), timeout=1.0)
        #         env.reset_req.clear()
        #     except asyncio.TimeoutError:
        #         if env.stop_req.is_set(): break
        #         continue

        env.reset_req.clear()
        
        await simcontrol.start_and_connect()
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
        raw_obs = await loop.run_in_executor(None, recv_exact, env.client_sock, env.state_recv_bytes)
        
        # 첫 관측값 설정 후 Env 깨우기
        if env.protocol_mode == PROTOCOL_ASSESSMENT:
            _, obs = env._parse_assessment_state(raw_obs)
            env.last_obs = obs
        else:
            env.last_obs = struct.unpack('ddd', raw_obs)[:2]
        env.ready_evt.set()
        
        # 시뮬레이션이 종료(Idle)될 때까지 배경에서 대기 및 감시
        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()
        
        if env.client_sock:
            env.client_sock.close()
            env.client_sock = None
        print("[Orchestrator] Episode Finished & Cleaned up.")
        await simcontrol.stop_and_disconnect()
        # print("[Orchestrator] Shutting down...")
        # if simcontrol.get_status() != cmapi.AppStatus.idle:
        #     await simcontrol.stop_sim()
        # server_sock.close()

# --- 3. 동기식 브릿지 및 메인 함수 ---
class SyncBridge(gym.Env):
    def __init__(self, a_env, loop):
        self.a_env, self.loop = a_env, loop
        self.action_space, self.observation_space = a_env.action_space, a_env.observation_space
    def reset(self, **kwargs): return asyncio.run_coroutine_threadsafe(self.a_env.reset(**kwargs), self.loop).result()
    def step(self, action):    return asyncio.run_coroutine_threadsafe(self.a_env.step(action), self.loop).result()
    def close(self):           pass

def run_learning(env):
    profile = get_ppo_profile()
    ppo_kwargs = PPO_PRESETS[profile]

    model_path = f"{MODEL_BASENAME}.zip"
    best_model_path = f"{MODEL_BASENAME}_best.zip"
    interrupt_model_path = f"{MODEL_BASENAME}_interrupt.zip"
    callback = SaveBestEpisodeRewardCallback(best_model_path)

    print(f"--- PPO 프로필: {profile} ---")
    print(f"--- PPO 설정: {ppo_kwargs} ---")

    # 1. 기존 모델이 있는지 확인
    if os.path.exists(model_path):
        print(f"--- 기존 모델 '{model_path}'을(를) 불러와서 학습을 재개합니다. ---")
        # 저장된 모델 로드
        model = PPO.load(model_path, env=env, device="cpu", verbose=1)
    else:
        print("--- 기존 모델이 없습니다. 새로운 모델을 생성합니다. ---")
        # 새로 시작할 경우
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", **ppo_kwargs)

    # 로드된 모델에도 프로필 설정을 동일하게 적용
    model.n_steps = ppo_kwargs["n_steps"]
    model.batch_size = ppo_kwargs["batch_size"]
    model.gamma = ppo_kwargs["gamma"]
    model.learning_rate = ppo_kwargs["learning_rate"]
    model.ent_coef = ppo_kwargs["ent_coef"]

    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "10000000"))
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
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
    cm_port = int(os.getenv("CM_PORT", "5555"))
    server_sock.bind(('127.0.0.1', cm_port))
    server_sock.listen(1)

    project_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(project_path)
    
    # master = cmapi.ApoServer()
    # master.set_sinfo(cmapi.ApoServerInfo(pid=get_carmaker_pid(), description="Idle"))
    # master.set_host("localhost")
    master = cmapi.CarMaker()
    # master.set_executable_path(project_path / "src/CarMaker_5555.linux64")
    master.set_executable_path(Path(os.getenv("CM_EXECUTABLE_PATH", str(project_path / "src/CarMaker.linux64"))))

    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(master)
    # ApoServer 설정 및 master 연결 (기존 코드와 동일)
    # master = ... / await simcontrol.set_master(master) / await simcontrol.start_and_connect()
    
    testrun_name = os.getenv("TESTRUN_NAME", "testrun_test1")
    testrun = cmapi.Project.instance().load_testrun_parametrization(proj_path / f"Data/TestRun/{testrun_name}")
    variation = cmapi.Variation.create_from_testrun(testrun)

    loop = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop)
    print(
        f"[CONFIG] CM_PROTOCOL_MODE={env_async.protocol_mode} | CONTROL_DT={env_async.control_dt:.3f}s "
        f"| CM_PORT={cm_port} | TESTRUN={testrun_name}"
    )
    if env_async.protocol_mode == PROTOCOL_ASSESSMENT:
        print(
            f"[CONFIG] TARGET_SPEED_MODE={env_async.target_speed_mode} | TARGET_SPEED_KPH={env_async.target_speed_fixed_kph:.3f} "
            f"| SCENARIO_DIR={SCENARIO_DIR} | SCENARIO_CSV_PATH={SCENARIO_CSV_PATH or '(auto)'}"
        )
    
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
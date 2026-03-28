import sys
import asyncio
import struct
import socket
import psutil
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from experiment_modes import (
    ACTION_MODE_RATIO3,
    ACTION_MODE_ACTION4,
    REWARD_MODE_EFFORT_ONLY,
    REWARD_MODE_VELOCITY_EFFORT,
    resolve_action_mode,
    resolve_reward_mode,
    resolve_model_basename,
    resolve_tensorboard_dir,
)

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi


BASE_DIR = Path(__file__).resolve().parent


ACTION_VELOCITY_WEIGHT = float(os.getenv("ACTION_VELOCITY_WEIGHT", "10"))
ACTION_EFFORT_WEIGHT = float(os.getenv("ACTION_EFFORT_WEIGHT", "1.0"))
DEFAULT_DEMAND_TORQUE = float(os.getenv("DEFAULT_DEMAND_TORQUE", "200.0"))
WHEEL_TORQUE_LIMIT = float(os.getenv("WHEEL_TORQUE_LIMIT", "875.0"))


PPO_PRESETS = {
    ACTION_MODE_RATIO3: {
        "A": {
            "learning_rate": 1e-5,
            "batch_size": 256,
            "gamma": 0.99,
            "n_steps": 50,
            "ent_coef": 0.0,
        },
        "B": {
            "learning_rate": 3e-4,
            "batch_size": 512,
            "gamma": 0.99,
            "n_steps": 2048,
            "ent_coef": 0.01,
        },
        "C": {
            "learning_rate": 1e-3,
            "batch_size": 512,
            "gamma": 0.99,
            "n_steps": 2048,
            "ent_coef": 0.0,
        },
    },
    ACTION_MODE_ACTION4: {
        "A": {
            "learning_rate": 1e-5,
            "batch_size": 256,
            "gamma": 0.99,
            "n_steps": 8192,
            "ent_coef": 0.1,
        },
        "B": {
            "learning_rate": 3e-4,
            "batch_size": 256,
            "gamma": 0.99,
            "n_steps": 4096,
            "ent_coef": 0.0,
        },
        "C": {
            "learning_rate": 1e-3,
            "batch_size": 256,
            "gamma": 0.99,
            "n_steps": 8192,
            "ent_coef": 0.0,
        },
    },
}


class SaveBestEpisodeRewardCallback(BaseCallback):
    def __init__(self, best_model_path, verbose=1):
        super().__init__(verbose)
        self.best_model_path = best_model_path
        self.best_reward_path = best_model_path + ".reward"
        self.current_episode_reward = 0.0
        self.best_episode_reward = -np.inf
        self.episode_count = 0
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
            self.episode_count += 1
            if self.current_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.current_episode_reward
                self.model.save(self.best_model_path)
                self._save_best_reward()
            self.current_episode_reward = 0.0
        return True


def get_ppo_profile(action_mode: str):
    profile = os.getenv("PPO_PROFILE")
    if not profile and len(sys.argv) > 1:
        profile = sys.argv[1]

    profile = (profile or "B").upper()
    presets = PPO_PRESETS[action_mode]
    if profile not in presets:
        print(f"알 수 없는 프로필 '{profile}' 입니다. B 프로필로 진행합니다. (사용 가능: A/B/C)")
        profile = "B"
    return profile


def get_carmaker_pid():
    target = "CarMaker.linux64"
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if target in proc.info["name"] or any(target in arg for arg in (proc.info["cmdline"] or [])):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class CarMaker4WIDEnv(gym.Env):
    def __init__(self, loop, action_mode: str, reward_mode: str):
        super().__init__()
        self.loop = loop
        self.client_sock = None
        self.last_obs = None
        self.step_count = 0
        self.episode_reward_sum = 0.0
        self.max_steps = int(os.getenv("EPISODE_MAX_STEPS", "500"))
        self.early_done_penalty = float(os.getenv("EARLY_DONE_PENALTY", "-50.0"))
        self.demand_torque = DEFAULT_DEMAND_TORQUE
        self.action_mode = action_mode
        self.reward_mode = reward_mode

        self.reset_req = asyncio.Event()
        self.ready_evt = asyncio.Event()
        self.stop_req = asyncio.Event()

        if self.action_mode == ACTION_MODE_RATIO3:
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def get_demand_torque(self, curr_v, v_diff):
        return float(self.demand_torque)

    def ratio_action_to_wheel_torques(self, action_ratio, demand_torque):
        left_torque_norm, left_ratio, right_ratio = np.asarray(action_ratio, dtype=np.float32)
        left_torque_norm = float(np.clip(left_torque_norm, -1.0, 1.0))
        left_ratio = float(np.clip(left_ratio, 0.0, 1.0))
        right_ratio = float(np.clip(right_ratio, 0.0, 1.0))

        left_total = left_torque_norm * WHEEL_TORQUE_LIMIT
        right_total = float(demand_torque) - left_total
        right_total = float(np.clip(right_total, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT))

        fl = left_total * left_ratio
        rl = left_total - fl
        fr = right_total * right_ratio
        rr = right_total - fr
        return np.array([fl, fr, rl, rr], dtype=np.float32)

    def compute_effort_term(self, action, wheel_torques, demand_torque):
        if self.action_mode == ACTION_MODE_RATIO3:
            torque_effort = float(np.sum(wheel_torques**2))
            norm = max(float(demand_torque) ** 2, 1e-6)
            return torque_effort / norm
        return float(np.sum(np.asarray(action, dtype=np.float32) ** 2))

    async def reset(self, seed=None, options=None):
        self.step_count = 0
        self.episode_reward_sum = 0.0
        self.reset_req.set()
        await self.ready_evt.wait()
        self.ready_evt.clear()
        return np.array(self.last_obs, dtype=np.float32), {}

    async def step(self, action):
        try:
            self.step_count += 1
            action = np.asarray(action, dtype=np.float32)
            curr_v, v_diff = self.last_obs if self.last_obs is not None else (0.0, 0.0)
            demand_torque = self.get_demand_torque(curr_v, v_diff)

            if self.action_mode == ACTION_MODE_RATIO3:
                wheel_torques = self.ratio_action_to_wheel_torques(action, demand_torque)
                cmd = wheel_torques / WHEEL_TORQUE_LIMIT
            else:
                wheel_torques = action * WHEEL_TORQUE_LIMIT
                cmd = action

            data = struct.pack("dddd", *map(float, cmd))
            await self.loop.run_in_executor(None, self.client_sock.sendall, data)

            raw_data = await self.loop.run_in_executor(None, self.client_sock.recv, 24)
            if not raw_data or len(raw_data) < 24:
                return np.zeros(2), 0, True, False, {}

            curr_v, v_diff, sim_state = struct.unpack("ddd", raw_data)
            self.last_obs = [curr_v, v_diff]

            velocity_penalty = ACTION_VELOCITY_WEIGHT * float(v_diff**2) / 1600.0
            effort_term = self.compute_effort_term(action, wheel_torques, demand_torque)
            effort_penalty = ACTION_EFFORT_WEIGHT * effort_term

            if self.reward_mode == REWARD_MODE_EFFORT_ONLY:
                reward = -effort_penalty
            else:
                reward = -(velocity_penalty + effort_penalty)

            terminated = sim_state != 8.0
            if terminated and self.step_count < self.max_steps:
                print(
                    f"[EARLY DONE] step={self.step_count} < max_steps={self.max_steps} -> penalty {self.early_done_penalty}"
                )
                reward += self.early_done_penalty

            self.episode_reward_sum += reward

            if self.action_mode == ACTION_MODE_RATIO3:
                print(
                    f"[RATIO] LeftTqNorm={action[0]:7.4f} | L_F/R={action[1]:7.4f} | R_F/R={action[2]:7.4f}"
                    f" | Demand={demand_torque:9.3f}"
                )
                print(
                    f"[TORQUE] FL={wheel_torques[0]:9.3f} | FR={wheel_torques[1]:9.3f}"
                    f" | RL={wheel_torques[2]:9.3f} | RR={wheel_torques[3]:9.3f}"
                )
            else:
                print(
                    f"[ACTION] FL={action[0]:7.4f} | FR={action[1]:7.4f}"
                    f" | RL={action[2]:7.4f} | RR={action[3]:7.4f}"
                )

            print(
                f"[REWARD-{self.reward_mode}] VDiff={velocity_penalty:9.4f} | Effort={effort_penalty:9.4f}"
                f" | Step={reward:9.4f} | Cumulative={self.episode_reward_sum:9.4f}"
            )

            return np.array(self.last_obs, dtype=np.float32), reward, terminated, False, {}
        except Exception as e:
            print(f"[STEP ERROR] {type(e).__name__}: {e}")
            return np.zeros(2), 0, True, False, {}


async def carmaker_orchestrator(env, simcontrol, variation, server_sock):
    loop = asyncio.get_running_loop()

    while True:
        await env.reset_req.wait()
        env.reset_req.clear()

        await simcontrol.connect()
        print("\n[Orchestrator] Starting New Episode...")

        if simcontrol.get_status() != cmapi.SimControlState.configure:
            await simcontrol.stop_sim()
            await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

        simcontrol.set_variation(variation.clone())
        await simcontrol.start_sim()
        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.running).wait()
        simcontrol.set_realtimefactor(100.0)

        env.client_sock, _ = await loop.run_in_executor(None, server_sock.accept)
        raw_obs = await loop.run_in_executor(None, env.client_sock.recv, 24)
        env.last_obs = struct.unpack("ddd", raw_obs)[:2]
        env.ready_evt.set()

        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

        if env.client_sock:
            env.client_sock.close()
            env.client_sock = None
        print("[Orchestrator] Episode Finished & Cleaned up.")
        await simcontrol.disconnect()


class SyncBridge(gym.Env):
    def __init__(self, a_env, loop):
        self.a_env = a_env
        self.loop = loop
        self.action_space = a_env.action_space
        self.observation_space = a_env.observation_space

    def reset(self, **kwargs):
        return asyncio.run_coroutine_threadsafe(self.a_env.reset(**kwargs), self.loop).result()

    def step(self, action):
        return asyncio.run_coroutine_threadsafe(self.a_env.step(action), self.loop).result()

    def close(self):
        pass


def run_learning(env, action_mode: str):
    profile = get_ppo_profile(action_mode)
    ppo_kwargs = PPO_PRESETS[action_mode][profile]

    log_dir = str(BASE_DIR / resolve_tensorboard_dir(action_mode))
    os.makedirs(log_dir, exist_ok=True)

    model_basename = resolve_model_basename(action_mode)
    model_path = str(BASE_DIR / f"{model_basename}.zip")
    best_model_path = str(BASE_DIR / f"{model_basename}_best.zip")
    interrupt_model_path = str(BASE_DIR / f"{model_basename}_interrupt.zip")
    callback = SaveBestEpisodeRewardCallback(best_model_path)
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "200000"))

    print(f"--- ACTION_MODE: {action_mode} | PPO 프로필: {profile} ---")
    print(f"--- PPO 설정: {ppo_kwargs} ---")
    print(f"--- MODEL_BASENAME: {model_basename} ---")

    if os.path.exists(model_path):
        print(f"--- 기존 모델 '{model_path}'을(를) 불러와서 학습을 재개합니다. ---")
        model = PPO.load(model_path, env=env, device="cpu", verbose=1)
    else:
        print("--- 기존 모델이 없습니다. 새로운 모델을 생성합니다. ---")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=log_dir, **ppo_kwargs)

    model.n_steps = ppo_kwargs["n_steps"]
    model.batch_size = ppo_kwargs["batch_size"]
    model.gamma = ppo_kwargs["gamma"]
    model.learning_rate = ppo_kwargs["learning_rate"]
    model.ent_coef = ppo_kwargs["ent_coef"]
    model.tensorboard_log = log_dir

    print("clip_range:", model.clip_range)
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
    action_mode = resolve_action_mode()
    reward_mode = resolve_reward_mode(action_mode)
    testrun_name = os.getenv("TESTRUN_NAME", "testrun_test1")
    port = int(os.getenv("CM_PORT", "5555"))

    print(f"[CONFIG] ACTION_MODE={action_mode} | REWARD_MODE={reward_mode} | TESTRUN={testrun_name}")

    proj_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(proj_path)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", port))
    server_sock.listen(1)

    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=get_carmaker_pid(), description="Idle"))
    master.set_host("localhost")

    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(master)

    testrun = cmapi.Project.instance().load_testrun_parametrization(proj_path / f"Data/TestRun/{testrun_name}")
    variation = cmapi.Variation.create_from_testrun(testrun)

    loop = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop, action_mode=action_mode, reward_mode=reward_mode)

    cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))

    env_sync = SyncBridge(env_async, loop)
    await loop.run_in_executor(None, run_learning, env_sync, action_mode)

    print("Learning finished. Sending stop signal to orchestrator...")
    env_async.stop_req.set()
    loop.stop()


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())

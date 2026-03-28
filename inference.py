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

from experiment_modes import (
    ACTION_MODE_RATIO3,
    ACTION_MODE_ACTION4,
    RUN_MODE_GUI,
    RUN_MODE_HEADLESS,
    REWARD_MODE_EFFORT_ONLY,
    REWARD_MODE_VELOCITY_EFFORT,
    resolve_action_mode,
    resolve_reward_mode,
    resolve_model_basename,
    resolve_env_id,
    resolve_cm_port,
    resolve_run_mode,
)

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi


BASE_DIR = Path(__file__).resolve().parent


ACTION_VELOCITY_WEIGHT = float(os.getenv("ACTION_VELOCITY_WEIGHT", "10"))
ACTION_EFFORT_WEIGHT = float(os.getenv("ACTION_EFFORT_WEIGHT", "1.0"))
DEFAULT_DEMAND_TORQUE = float(os.getenv("DEFAULT_DEMAND_TORQUE", "0.0"))
WHEEL_TORQUE_LIMIT = float(os.getenv("WHEEL_TORQUE_LIMIT", "875.0"))


def get_carmaker_pid():
    target = "CarMaker"
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = proc.info.get("cmdline") or []
            if target in name or any(target in arg for arg in cmdline):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def build_master(run_mode: str, project_path: Path):
    if run_mode == RUN_MODE_HEADLESS:
        master = cmapi.CarMaker()
        executable_path = Path(os.getenv("CM_EXECUTABLE_PATH", str(project_path / "src/CarMaker.linux64")))
        master.set_executable_path(executable_path)
        return master

    pid = get_carmaker_pid()
    if pid is None:
        raise RuntimeError("실행 중인 CarMaker GUI 프로세스를 찾지 못했습니다.")
    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=pid, description="Idle"))
    master.set_host("localhost")
    return master


class CarMaker4WIDEnv(gym.Env):
    def __init__(self, loop, action_mode: str, reward_mode: str):
        super().__init__()
        self.loop = loop
        self.client_sock = None
        self.last_obs = None
        self.demand_torque = DEFAULT_DEMAND_TORQUE
        self.step_count = 0
        self.max_steps = int(os.getenv("EPISODE_MAX_STEPS", "500"))
        self.early_done_penalty = float(os.getenv("EARLY_DONE_PENALTY", "-50.0"))
        self.action_mode = action_mode
        self.reward_mode = reward_mode

        self.reset_req = asyncio.Event()
        self.ready_evt = asyncio.Event()

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
                return np.zeros(2), 0.0, True, False, {}

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
                reward += self.early_done_penalty

            return np.array(self.last_obs, dtype=np.float32), reward, terminated, False, {}
        except Exception:
            return np.zeros(2), 0.0, True, False, {}


async def carmaker_orchestrator(env, simcontrol, variation, server_sock):
    loop = asyncio.get_running_loop()
    while True:
        await env.reset_req.wait()
        env.reset_req.clear()

        await simcontrol.connect()

        if simcontrol.get_status() != cmapi.SimControlState.configure:
            await simcontrol.stop_sim()
            await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

        simcontrol.set_variation(variation.clone())
        await simcontrol.start_sim()
        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.running).wait()

        env.client_sock, _ = await loop.run_in_executor(None, server_sock.accept)
        raw_obs = await loop.run_in_executor(None, env.client_sock.recv, 24)

        env.last_obs = struct.unpack("ddd", raw_obs)[:2]
        env.ready_evt.set()

        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

        if env.client_sock:
            env.client_sock.close()
            env.client_sock = None
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


def resolve_model_path(action_mode: str) -> str:
    use_best = os.getenv("PPO_USE_BEST", "1") == "1"
    model_basename = resolve_model_basename(action_mode)
    default_model_name = f"{model_basename}{'_best' if use_best else ''}.zip"

    model_path_raw = os.getenv("MODEL_PATH", default_model_name)
    candidate = Path(model_path_raw)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    model_path = str(candidate)

    if not os.path.exists(model_path):
        fallback = str(BASE_DIR / f"{model_basename}.zip")
        if use_best and os.path.exists(fallback):
            print(f"[WARN] '{model_path}' not found. Fallback to '{fallback}'")
            model_path = fallback

    return model_path


def run_inference(env, action_mode: str):
    model_path = resolve_model_path(action_mode)
    episodes = int(os.getenv("EPISODES", "1"))

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step_count += 1
            done = bool(terminated or truncated)

        print(f"Episode {ep + 1}/{episodes} done | steps={step_count} | total_reward={total_reward:.3f}")


async def main():
    project_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(project_path)

    action_mode = resolve_action_mode()
    reward_mode = resolve_reward_mode(action_mode)
    run_mode = resolve_run_mode()
    testrun_name = os.getenv("TESTRUN_NAME", "testrun_test1")
    env_id = resolve_env_id()
    port = resolve_cm_port()

    print(
        f"[CONFIG] ACTION_MODE={action_mode} | REWARD_MODE={reward_mode}"
        f" | RUN_MODE={run_mode} | TESTRUN={testrun_name} | ENV_ID={env_id} | CM_PORT={port}"
    )

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", port))
    server_sock.listen(1)

    master = build_master(run_mode, project_path)

    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(master)

    testrun = cmapi.Project.instance().load_testrun_parametrization(project_path / f"Data/TestRun/{testrun_name}")
    variation = cmapi.Variation.create_from_testrun(testrun)

    loop = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop, action_mode=action_mode, reward_mode=reward_mode)

    cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))

    env_sync = SyncBridge(env_async, loop)
    await loop.run_in_executor(None, run_inference, env_sync, action_mode)


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())

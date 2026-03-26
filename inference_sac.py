import sys, asyncio, struct, socket, psutil
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import SAC

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi


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


class CarMaker4WIDEnv(gym.Env):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.client_sock = None
        self.last_obs = None

        self.reset_req = asyncio.Event()
        self.ready_evt = asyncio.Event()

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    async def reset(self, seed=None, options=None):
        self.reset_req.set()
        await self.ready_evt.wait()
        self.ready_evt.clear()
        return np.array(self.last_obs, dtype=np.float32), {}

    async def step(self, action):
        try:
            data = struct.pack("dddd", *map(float, action))
            await self.loop.run_in_executor(None, self.client_sock.sendall, data)

            raw_data = await self.loop.run_in_executor(None, self.client_sock.recv, 24)
            if not raw_data or len(raw_data) < 24:
                return np.zeros(2), 0.0, True, False, {}

            curr_v, v_diff, sim_state = struct.unpack("ddd", raw_data)
            self.last_obs = [curr_v, v_diff]

            reward = -float(v_diff**2) - 100.0 * np.sum(action**2)
            terminated = sim_state != 8.0
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


def run_inference(env):
    profile = (os.getenv("SAC_PROFILE", "B") or "B").upper()
    default_model_path = f"carmaker_sac_4wid_gui_{profile}.zip"
    model_path = os.getenv("MODEL_PATH", default_model_path)
    episodes = int(os.getenv("EPISODES", "1"))

    print(f"Loading SAC model from {model_path}")
    model = SAC.load(model_path, env=env, device="cpu")

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

    pid = get_carmaker_pid()
    if pid is None:
        raise RuntimeError("실행 중인 CarMaker GUI 프로세스를 찾지 못했습니다.")

    port = int(os.getenv("CM_PORT", "5555"))
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", port))
    server_sock.listen(1)

    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=pid, description="Idle"))
    master.set_host("localhost")

    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(master)

    # testrun = cmapi.Project.instance().load_testrun_parametrization(project_path / "Data/TestRun/testrun_test_straight")
    testrun = cmapi.Project.instance().load_testrun_parametrization(project_path / "Data/TestRun/testrun_test1")
    variation = cmapi.Variation.create_from_testrun(testrun)

    loop = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop)

    cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))

    env_sync = SyncBridge(env_async, loop)
    await loop.run_in_executor(None, run_inference, env_sync)


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())

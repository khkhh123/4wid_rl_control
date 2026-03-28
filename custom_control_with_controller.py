import sys, asyncio, threading, struct, socket, psutil
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi

def get_carmaker_pid():
    targets = ("CarMaker_5555.linux64", "CarMaker.linux64")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = proc.info.get('name') or ""
            cmdline = proc.info.get('cmdline') or []
            if any(t in name or any(t in arg for arg in cmdline) for t in targets):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

class CarMaker4WIDEnv(gym.Env):
    def __init__(self, loop, controller=None):
        super().__init__()
        self.loop = loop
        self.client_sock = None
        self.last_obs = None
        self.step_count = 0
        self.max_steps = int(os.getenv("EPISODE_MAX_STEPS", "500"))
        self.early_done_penalty = float(os.getenv("EARLY_DONE_PENALTY", "-1000000.0"))
        self.controller = controller
        self.reset_req = asyncio.Event()
        self.ready_evt = asyncio.Event()
        self.stop_req = asyncio.Event()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def action_to_torques(self, action):
        """Convert action [demand_ratio, left_ratio, right_ratio] to wheel torques [FL, FR, RL, RR]."""
        demand_ratio, left_ratio, right_ratio = action
        base_torque = abs(demand_ratio) * 0.1
        lr_ratio = (demand_ratio + 1) / 2
        left_torque = base_torque * (1 - lr_ratio)
        right_torque = base_torque * lr_ratio
        fl = left_torque * (1 - left_ratio) / 2
        rl = left_torque * (1 + left_ratio) / 2
        fr = right_torque * (1 - right_ratio) / 2
        rr = right_torque * (1 + right_ratio) / 2
        return np.array([fl, fr, rl, rr], dtype=np.float32)

    async def reset(self, seed=None, options=None):
        self.step_count = 0
        self.reset_req.set()
        await self.ready_evt.wait()
        self.ready_evt.clear()
        return np.array(self.last_obs, dtype=np.float32), {}

    async def step(self, action=None):
        try:
            self.step_count += 1
            # 제어기가 있으면 obs로부터 action 생성
            if self.controller is not None and self.last_obs is not None:
                action = self.controller(np.array(self.last_obs, dtype=np.float32))
            elif action is None:
                raise ValueError("Action must be provided if no controller is set.")
            torques = self.action_to_torques(action)
            data = struct.pack('dddd', *map(float, torques))
            await self.loop.run_in_executor(None, self.client_sock.sendall, data)
            raw_data = await self.loop.run_in_executor(None, self.client_sock.recv, 24)
            if not raw_data or len(raw_data) < 24:
                return np.zeros(2), 0, True, False, {}
            curr_v, v_diff, sim_state = struct.unpack('ddd', raw_data)
            self.last_obs = [curr_v, v_diff]
            action = np.asarray(action, dtype=np.float32)
            # ACTION_VELOCITY_WEIGHT와 ACTION_EFFORT_WEIGHT는 import된 상수에서 사용
            velocity_penalty = 10.0 * np.exp(-float(v_diff**2) / 1600)  # ACTION_VELOCITY_WEIGHT default
            effort_penalty = 1.0 * float(np.sum(action**2))  # ACTION_EFFORT_WEIGHT default
            weight_dist = 1.0
            reward = velocity_penalty - weight_dist * effort_penalty
            terminated = (sim_state != 8.0)
            if terminated and self.step_count < self.max_steps:
                reward += self.early_done_penalty
            return np.array(self.last_obs, dtype=np.float32), reward, terminated, False, {}
        except Exception as e:
            # 디버깅을 위해 예외를 출력해 즉시 원인을 확인할 수 있도록 한다.
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
        env.last_obs = struct.unpack('ddd', raw_obs)[:2]
        env.ready_evt.set()
        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()
        if env.client_sock:
            env.client_sock.close()
            env.client_sock = None
        print("[Orchestrator] Episode Finished & Cleaned up.")
        await simcontrol.disconnect()

class SyncBridge(gym.Env):
    def __init__(self, a_env, loop):
        self.a_env, self.loop = a_env, loop
        self.action_space, self.observation_space = a_env.action_space, a_env.observation_space
    def reset(self, **kwargs):
        return asyncio.run_coroutine_threadsafe(self.a_env.reset(**kwargs), self.loop).result()
    def step(self, action=None):
        return asyncio.run_coroutine_threadsafe(self.a_env.step(action), self.loop).result()
    def close(self):
        pass

def run_custom_control(env, episodes=1):
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        transitions = []  # (obs, action, reward, next_obs, done)
        while not done:
            curr_obs = obs.copy()
            # 내부 controller로 action 생성
            action = env.a_env.controller(np.array(curr_obs, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = obs.copy()
            done_flag = bool(terminated or truncated)
            transitions.append((curr_obs, action, reward, next_obs, done_flag))
            total_reward += float(reward)
            step_count += 1
            done = done_flag
        # numpy array로 변환 및 저장
        arr = np.array(transitions, dtype=object)
        np.save(f"custom_episode_{ep+1}.npy", arr)
        print(f"Episode {ep + 1}/{episodes} done | steps={step_count} | total_reward={total_reward:.3f} | saved: custom_episode_{ep+1}.npy")

def example_controller(obs):
    # 사용자가 원하는 제어 로직을 여기에 작성
    # 예시: 단순히 v_diff를 줄이기 위한 proportional control
    curr_v, v_diff = obs
    kp = 1000
    demand_ratio = np.clip((kp * v_diff) / 3000, -1.0, 1.0)
    left_ratio = 0.0
    right_ratio = 0.0
    action = np.array([demand_ratio, left_ratio, right_ratio], dtype=np.float32)
    return np.clip(action, -1, 1)

async def main():
    proj_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(proj_path)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('127.0.0.1', 5555))
    server_sock.listen(1)
    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=get_carmaker_pid(), description="Idle"))
    master.set_host("localhost")
    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(master)
    testrun = cmapi.Project.instance().load_testrun_parametrization(proj_path / "Data/TestRun/testrun_test1")
    variation = cmapi.Variation.create_from_testrun(testrun)
    loop = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop, controller=example_controller)  # controller 인자로 전달
    cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))
    env_sync = SyncBridge(env_async, loop)
    await loop.run_in_executor(None, run_custom_control, env_sync, 3)  # 3 에피소드 예시
    print("Custom control finished. Sending stop signal to orchestrator...")
    env_async.stop_req.set()

if __name__ == "__main__":
    cmapi.Task.run_main_task(main())

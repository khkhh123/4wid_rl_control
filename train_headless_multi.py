import os
import sys
import subprocess
import asyncio
import socket
import threading
import struct
from typing import Callable
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from time import monotonic

from train_with_cm_gui import (
    BASE_DIR,
    PPO_PRESETS,
    CarMaker4WIDEnv,
    SyncBridge,
    cmapi,
    get_ppo_profile,
)
from experiment_modes import resolve_num_workers
from experiment_modes import (
    RUN_MODE_HEADLESS,
    resolve_action_mode,
    resolve_reward_mode,
    resolve_model_basename,
    resolve_tensorboard_dir,
)


class SaveBestEpisodeRewardVecCallback(BaseCallback):
    def __init__(self, best_model_path: str, n_envs: int, verbose: int = 1):
        super().__init__(verbose)
        self.best_model_path = best_model_path
        self.best_reward_path = best_model_path + ".reward"
        self.best_episode_reward = -np.inf
        self.current_episode_rewards = np.zeros(n_envs, dtype=np.float64)
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

        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        self.current_episode_rewards += rewards

        done_indices = np.where(dones)[0]
        for idx in done_indices:
            episode_reward = float(self.current_episode_rewards[idx])
            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                self.model.save(self.best_model_path)
                self._save_best_reward()
            self.current_episode_rewards[idx] = 0.0
        return True


class SharedProgressCallback(BaseCallback):
    def __init__(self, log_every_steps: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.log_every_steps = max(1, int(log_every_steps))
        self._last_log_step = 0
        self._start_ts = monotonic()

    def _on_step(self) -> bool:
        current = int(self.num_timesteps)
        if (current - self._last_log_step) < self.log_every_steps:
            return True

        elapsed = max(monotonic() - self._start_ts, 1e-6)
        fps = current / elapsed
        print(f"[SHARED-PROGRESS] timesteps={current} | elapsed={elapsed:.1f}s | fps={fps:.2f}")
        self._last_log_step = current
        return True


class SharedHeartbeatCallback(BaseCallback):
    def __init__(self, interval_sec: float = 5.0, verbose: int = 1):
        super().__init__(verbose)
        self.interval_sec = max(0.5, float(interval_sec))
        self._start_ts = monotonic()
        self._last_ts = self._start_ts

    def _on_step(self) -> bool:
        now = monotonic()
        if (now - self._last_ts) < self.interval_sec:
            return True

        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        mean_reward = float(np.mean(rewards)) if rewards is not None else float("nan")
        done_count = int(np.sum(dones)) if dones is not None else -1

        elapsed = now - self._start_ts
        print(
            "\n=== SHARED-HEARTBEAT"
            f" elapsed={elapsed:.1f}s"
            f" timesteps={int(self.num_timesteps)}"
            f" mean_reward={mean_reward:.4f}"
            f" done_count={done_count}"
            " ==="
        )
        self._last_ts = now
        return True


class HeadlessWorkerEnv(gym.Env):
    def __init__(
        self,
        worker_id: int,
        action_mode: str,
        reward_mode: str,
        project_path: str,
        testrun_name: str,
        cm_port_base: int,
        cm_executable_path: str,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.action_mode = action_mode
        self.reward_mode = reward_mode
        self.project_path = Path(project_path)
        self.testrun_name = testrun_name
        self.cm_port = int(cm_port_base) + int(worker_id)
        self.cm_executable_path = Path(cm_executable_path)

        # CarMaker child process inherits this worker process env.
        # Force explicit per-worker port variables to avoid fallback to 5555.
        os.environ["ENV_ID"] = str(worker_id)
        os.environ["CM_PORT_BASE"] = str(cm_port_base)
        os.environ["CM_PORT"] = str(self.cm_port)

        self._loop = None
        self._thread = None
        self._setup_done = threading.Event()
        self._setup_error = None

        self.server_sock = None
        self.master = None
        self.simcontrol = None
        self.env_async = None
        self.env_sync = None
        self._orchestrator_task = None
        self.trace_enabled = os.getenv("SHARED_WORKER_TRACE", "1").strip().lower() not in {"0", "false", "off"}
        self._t0 = monotonic()

        self._start_backend()
        if self._setup_error is not None:
            raise RuntimeError(f"Worker {self.worker_id} setup failed: {self._setup_error}")

        self.action_space = self.env_sync.action_space
        self.observation_space = self.env_sync.observation_space

    def _wlog(self, msg: str):
        if not self.trace_enabled:
            return
        dt = monotonic() - self._t0
        print(f"### [W{self.worker_id} +{dt:7.3f}s] {msg}", flush=True)

    async def _headless_orchestrator(self, variation):
        loop = asyncio.get_running_loop()

        while True:
            self._wlog("wait reset_req")
            await self.env_async.reset_req.wait()
            self.env_async.reset_req.clear()
            self._wlog("reset_req received")

            self._wlog("start_and_connect begin")
            await self.simcontrol.start_and_connect()
            self._wlog("start_and_connect done")
            print(f"[SHARED-WORKER-{self.worker_id}] Orchestrator: Starting New Episode...")

            if self.simcontrol.get_status() != cmapi.SimControlState.configure:
                await self.simcontrol.stop_sim()
                print("@@@")
                # await self.simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()
                # print("!!!")

            self.simcontrol.set_variation(variation.clone())
            self._wlog("start_sim begin")
            # running_cond = self.simcontrol.create_simstate_condition(cmapi.ConditionSimState.running)
            await self.simcontrol.start_sim()
            # await running_cond.wait()
            self._wlog("running-state reached")
            self.simcontrol.set_realtimefactor(100.0)

            self._wlog("accept begin")
            self.env_async.client_sock, _ = await loop.run_in_executor(None, self.server_sock.accept)
            self._wlog("accept done")

            raw_obs = await loop.run_in_executor(None, self.env_async.client_sock.recv, 24)
            self.env_async.last_obs = struct.unpack("ddd", raw_obs)[:2]
            self._wlog("initial obs recv done")
            self.env_async.ready_evt.set()
            self._wlog("ready_evt set")

            idle_task = asyncio.create_task(
                self.simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()
            )
            next_reset_task = asyncio.create_task(self.env_async.reset_req.wait())
            done, pending = await asyncio.wait(
                {idle_task, next_reset_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if idle_task in done:
                self._wlog("sim idle reached")
            else:
                self._wlog("sim idle wait bypassed (next reset requested)")

            for task in pending:
                task.cancel()

            if self.env_async.client_sock:
                self.env_async.client_sock.close()
                self.env_async.client_sock = None

            print(f"[SHARED-WORKER-{self.worker_id}] Orchestrator: Episode Finished.")
            self._wlog("stop_and_disconnect begin")
            await self.simcontrol.stop_and_disconnect()
            self._wlog("stop_and_disconnect done")

    def _start_backend(self):
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._setup_done.wait(timeout=120)
        if not self._setup_done.is_set():
            raise RuntimeError(f"Worker {self.worker_id}: setup timeout")

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_setup())
        except Exception as e:
            self._setup_error = e
            self._setup_done.set()
            return

        self._setup_done.set()
        self._loop.run_forever()

        try:
            self._loop.run_until_complete(self._async_close())
        finally:
            self._loop.close()

    async def _async_setup(self):
        cmapi.Project.load(self.project_path)

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("127.0.0.1", self.cm_port))
        self.server_sock.listen(1)

        self.master = cmapi.CarMaker()
        self.master.set_executable_path(self.cm_executable_path)

        self.simcontrol = cmapi.SimControlInteractive()
        await self.simcontrol.set_master(self.master)

        testrun = cmapi.Project.instance().load_testrun_parametrization(
            self.project_path / f"Data/TestRun/{self.testrun_name}"
        )
        variation = cmapi.Variation.create_from_testrun(testrun)

        self.env_async = CarMaker4WIDEnv(self._loop, action_mode=self.action_mode, reward_mode=self.reward_mode)
        self.env_async.stop_req.clear()
        self._orchestrator_task = asyncio.create_task(self._headless_orchestrator(variation))
        self._orchestrator_task.add_done_callback(self._on_orchestrator_done)
        self.env_sync = SyncBridge(self.env_async, self._loop)

        print(
            f"[SHARED-WORKER-{self.worker_id}] RUN_MODE={RUN_MODE_HEADLESS} "
            f"| CM_PORT={self.cm_port} | EXEC={self.cm_executable_path.name} "
            f"| PY_PID={os.getpid()} | THREAD_ID={threading.get_ident()} "
            f"| MASTER_OBJ=0x{id(self.master):x} | SIMCONTROL_OBJ=0x{id(self.simcontrol):x}"
        )

    def _on_orchestrator_done(self, task):
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[SHARED-WORKER-{self.worker_id}] orchestrator callback error: {e}")
            return
        if exc is not None:
            print(f"[SHARED-WORKER-{self.worker_id}] orchestrator crashed: {type(exc).__name__}: {exc}")

    async def _async_close(self):
        if self._orchestrator_task is not None:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._orchestrator_task = None

        if self.simcontrol is not None:
            try:
                await self.simcontrol.disconnect()
            except Exception:
                pass

        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None

    def reset(self, **kwargs):
        t0 = monotonic()
        fut = asyncio.run_coroutine_threadsafe(self.env_async.reset(**kwargs), self._loop)
        out = fut.result()
        self._wlog(f"reset ok ({monotonic() - t0:.3f}s)")
        return out

    def step(self, action):
        t0 = monotonic()
        fut = asyncio.run_coroutine_threadsafe(self.env_async.step(action), self._loop)
        out = fut.result()
        dt = monotonic() - t0
        if dt > 1.0:
            self._wlog(f"step slow ({dt:.3f}s)")
        return out

    def close(self):
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=15)


def make_shared_worker_env(
    worker_id: int,
    action_mode: str,
    reward_mode: str,
    project_path: str,
    testrun_name: str,
    cm_port_base: int,
    cm_executable_path: str,
) -> Callable[[], gym.Env]:
    def _init():
        return HeadlessWorkerEnv(
            worker_id=worker_id,
            action_mode=action_mode,
            reward_mode=reward_mode,
            project_path=project_path,
            testrun_name=testrun_name,
            cm_port_base=cm_port_base,
            cm_executable_path=cm_executable_path,
        )

    return _init


def build_worker_env(worker_id: int):
    env = os.environ.copy()
    env.setdefault("CM_RUN_MODE", "headless")
    env["ENV_ID"] = str(worker_id)
    env.setdefault("CM_PORT_BASE", "5555")
    env.setdefault("CM_EXECUTABLE_PATH", "/home/khkhh/CM_Projects/test1/src/CarMaker_multi.linux64")
    env.setdefault("NUM_WORKERS", env.get("NUM_WORKERS", "1"))

    # Default: isolate each worker's model/log files to avoid write races.
    model_mode = env.setdefault("MULTI_MODEL_MODE", "per_worker").strip().lower()
    if model_mode == "per_worker":
        action_mode = env.get("ACTION_MODE", "ratio3").strip().lower()
        if action_mode == "action4":
            base = "carmaker_ppo_4wid_action4"
            tb = "ppo_carmaker_action4_tensorboard"
        else:
            base = "carmaker_ppo_4wid_lefttorque_ratio3"
            tb = "ppo_carmaker_lefttorque_ratio3_tensorboard"
        env.setdefault("MODEL_BASENAME", f"{base}_w{worker_id}")
        env.setdefault("TB_DIRNAME", f"{tb}_w{worker_id}")

    return env


def run_shared_training(num_workers: int, profile: str):
    action_mode = resolve_action_mode()
    reward_mode = resolve_reward_mode(action_mode)
    ppo_kwargs = PPO_PRESETS[action_mode][profile]

    project_path = os.getenv("CM_PROJECT_PATH", "/home/khkhh/CM_Projects/test1")
    testrun_name = os.getenv("TESTRUN_NAME", "testrun_test1")
    cm_port_base = int(os.getenv("CM_PORT_BASE", "5555"))
    cm_executable_path = os.getenv(
        "CM_EXECUTABLE_PATH", "/home/khkhh/CM_Projects/test1/src/CarMaker_multi.linux64"
    )
    progress_every_steps = int(os.getenv("SHARED_PROGRESS_EVERY_STEPS", "200"))
    heartbeat_sec = float(os.getenv("SHARED_HEARTBEAT_SEC", "5"))

    model_basename = resolve_model_basename(action_mode)
    tb_dir = resolve_tensorboard_dir(action_mode)
    log_dir = str(BASE_DIR / tb_dir)
    os.makedirs(log_dir, exist_ok=True)

    model_path = str(BASE_DIR / f"{model_basename}.zip")
    best_model_path = str(BASE_DIR / f"{model_basename}_best.zip")
    interrupt_model_path = str(BASE_DIR / f"{model_basename}_interrupt.zip")
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "200000"))

    env_fns = [
        make_shared_worker_env(
            worker_id=i,
            action_mode=action_mode,
            reward_mode=reward_mode,
            project_path=project_path,
            testrun_name=testrun_name,
            cm_port_base=cm_port_base,
            cm_executable_path=cm_executable_path,
        )
        for i in range(num_workers)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    best_callback = SaveBestEpisodeRewardVecCallback(best_model_path=best_model_path, n_envs=num_workers)
    progress_callback = SharedProgressCallback(log_every_steps=progress_every_steps)
    heartbeat_callback = SharedHeartbeatCallback(interval_sec=heartbeat_sec)
    callback = CallbackList([best_callback, progress_callback, heartbeat_callback])

    print(
        f"[SHARED] ACTION_MODE={action_mode} | REWARD_MODE={reward_mode} "
        f"| PPO_PROFILE={profile} | NUM_WORKERS={num_workers}"
    )
    print(f"[SHARED] PPO settings: {ppo_kwargs}")
    print(f"[SHARED] MODEL_BASENAME={model_basename}")
    print(
        f"[SHARED] CM_PROJECT_PATH={project_path} | TESTRUN_NAME={testrun_name} "
        f"| CM_EXECUTABLE_PATH={cm_executable_path} | CM_PORT_BASE={cm_port_base}"
    )
    print(f"[SHARED] PROGRESS_LOG_EVERY={progress_every_steps} timesteps")
    print(f"[SHARED] HEARTBEAT_EVERY={heartbeat_sec:.1f}s")

    try:
        if os.path.exists(model_path):
            print(f"[SHARED] Loading existing model: {model_path}")
            model = PPO.load(model_path, env=vec_env, device="cpu", verbose=1)
        else:
            print("[SHARED] No model found. Creating a new PPO model.")
            model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", tensorboard_log=log_dir, **ppo_kwargs)

        model.n_steps = ppo_kwargs["n_steps"]
        model.batch_size = ppo_kwargs["batch_size"]
        model.gamma = ppo_kwargs["gamma"]
        model.learning_rate = ppo_kwargs["learning_rate"]
        model.ent_coef = ppo_kwargs["ent_coef"]
        model.tensorboard_log = log_dir

        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("[SHARED] Training interrupted. Saving interrupt model...")
        model.save(interrupt_model_path)
        print(f"[SHARED] Interrupt model saved to {interrupt_model_path}")
    finally:
        try:
            model.save(model_path)
            print(f"[SHARED] Model saved to {model_path}")
        except Exception:
            pass
        try:
            vec_env.close()
        except Exception as e:
            print(f"[SHARED] vec_env close warning: {type(e).__name__}: {e}")


def run_independent_training(num_workers: int, profile: str, project_dir: Path):
    train_script = project_dir / "train_with_cm_gui.py"
    model_mode = os.getenv("MULTI_MODEL_MODE", "per_worker")

    print(
        f"[MULTI-INDEPENDENT] Launching {num_workers} headless workers"
        f" | PPO_PROFILE={profile} | MULTI_MODEL_MODE={model_mode}"
    )
    procs = []

    for worker_id in range(num_workers):
        env = build_worker_env(worker_id)
        cmd = [sys.executable, str(train_script), profile]
        print(
            f"[WORKER-{worker_id}] CM_PORT={int(env.get('CM_PORT_BASE', '5555')) + worker_id}"
            f" | ENV_ID={worker_id} | MODEL_BASENAME={env.get('MODEL_BASENAME', '(default)')}"
        )
        procs.append(subprocess.Popen(cmd, cwd=str(project_dir), env=env))

    exit_codes = []
    try:
        for p in procs:
            exit_codes.append(p.wait())
    except KeyboardInterrupt:
        print("[MULTI-INDEPENDENT] KeyboardInterrupt: terminating workers...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            if p.poll() is None:
                p.wait(timeout=10)

    non_zero = [c for c in exit_codes if c != 0]
    if non_zero:
        raise SystemExit(1)


def main():
    num_workers = resolve_num_workers(default_value=max(1, (os.cpu_count() or 2) // 2))
    profile = get_ppo_profile(resolve_action_mode())
    train_mode = os.getenv("MULTI_TRAIN_MODE", "shared").strip().lower()
    project_dir = Path(__file__).resolve().parent

    if train_mode == "independent":
        run_independent_training(num_workers=num_workers, profile=profile, project_dir=project_dir)
        return
    if train_mode != "shared":
        print(f"[WARN] Unknown MULTI_TRAIN_MODE='{train_mode}'. Fallback to 'shared'.")

    run_shared_training(num_workers=num_workers, profile=profile)


if __name__ == "__main__":
    main()

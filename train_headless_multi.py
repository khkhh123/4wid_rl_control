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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from time import monotonic

from train_with_cm_gui import (
    PPO_PRESETS,
    SAC_PRESETS,
    TD3_PRESETS,
    WHEEL_TORQUE_LIMIT,
    MODEL_BASENAME,
    CarMaker4WIDEnv,
    SyncBridge,
    recv_exact,
    cmapi,
    get_ppo_profile,
)
from experiment_modes import resolve_num_workers
from experiment_modes import RUN_MODE_HEADLESS

PROJECT_DIR = Path(__file__).resolve().parent

class SaveBestEpisodeRewardVecCallback(BaseCallback):
    def __init__(self, best_model_path: str, latest_model_path: str, n_envs: int, verbose: int = 1):
        super().__init__(verbose)
        self.best_model_path = best_model_path
        self.latest_model_path = latest_model_path
        self.best_reward_path = best_model_path + ".reward"
        self.best_episode_reward = -np.inf
        self.current_episode_rewards = np.zeros(n_envs, dtype=np.float64)
        self.current_yaw_penalties = np.zeros(n_envs, dtype=np.float64)
        self.current_effort_penalties = np.zeros(n_envs, dtype=np.float64)
        self.current_saturation_penalties = np.zeros(n_envs, dtype=np.float64)
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
        infos = self.locals.get("infos")
        if rewards is None or dones is None:
            return True

        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        self.current_episode_rewards += rewards

        done_indices = np.where(dones)[0]
        for idx in done_indices:
            self.episode_count += 1
            episode_reward = float(self.current_episode_rewards[idx])
            # 항목별 누적 (info에서 읽기)
            ep_info = infos[idx].get("episode", {}) if infos else {}
            yaw_pen = ep_info.get("yaw_penalty", 0.0)
            effort_pen = ep_info.get("effort_penalty", 0.0)
            sat_pen = ep_info.get("saturation_penalty", 0.0)
            # 매 에피소드 latest 저장
            self.model.save(self.latest_model_path)
            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                self.model.save(self.best_model_path)
                self._save_best_reward()
                if self.verbose:
                    print(f"[BEST] episode={self.episode_count} reward={self.best_episode_reward:.3f}")
            # TensorBoard 로깅
            self.logger.record("episode/reward", episode_reward)
            self.logger.record("episode/count", self.episode_count)
            self.logger.record("episode/best_reward", self.best_episode_reward)
            self.logger.record("episode/yaw_penalty", yaw_pen)
            self.logger.record("episode/effort_penalty", effort_pen)
            self.logger.record("episode/saturation_penalty", sat_pen)
            self.logger.dump(self.num_timesteps)
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
        project_path: str,
        testrun_name: str,
        cm_port_base: int,
        cm_executable_path: str,
    ):
        super().__init__()
        self.worker_id = worker_id
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

    async def _make_simcontrol(self):
        """매 에피소드마다 fresh한 master/simcontrol 인스턴스를 생성해 이전 PID 캐시 문제를 회피."""
        master = cmapi.CarMaker()
        master.set_executable_path(self.cm_executable_path)
        # CM 프로세스에 포트 관련 환경변수를 명시적으로 전달
        # (env 상속에 의존하지 않고 cmapi API로 직접 지정)
        master.set_environment_variables({
            "ENV_ID": str(self.worker_id),
            "CM_PORT_BASE": str(self.cm_port - self.worker_id),  # base
            "CM_PORT": str(self.cm_port),
        })
        simcontrol = cmapi.SimControlInteractive()
        await simcontrol.set_master(master)
        return master, simcontrol

    async def _headless_orchestrator(self, variation):
        loop = asyncio.get_running_loop()

        # Worker lifetime 동안 CarMaker 인스턴스를 재사용한다.
        # (에피소드마다 재생성/재연결하지 않음)
        master, simcontrol = await self._make_simcontrol()
        self.master = master
        self.simcontrol = simcontrol
        self._wlog("connect begin (worker lifetime)")
        await asyncio.wait_for(simcontrol.start_and_connect(), timeout=20.0)
        self._wlog("connect done (worker lifetime)")

        while True:
            self._wlog("wait reset_req")
            await self.env_async.reset_req.wait()
            self.env_async.reset_req.clear()
            self._wlog("reset_req received")

            if self.env_async.stop_req.is_set():
                self._wlog("stop requested")
                break

            self.env_async.client_sock = None

            try:
                self._wlog(f"simcontrol status = {simcontrol.get_status()}")
                print(f"[SHARED-WORKER-{self.worker_id}] Orchestrator: Starting New Episode...")

                simcontrol.set_variation(variation.clone())
                self._wlog("start_sim begin")
                await asyncio.sleep(1.0)  # CM이 안정적으로 시작될 때까지 대기
                await simcontrol.start_sim()
                simcontrol.set_realtimefactor(100.0)

                self._wlog(f"accept begin (port={self.cm_port})")
                self.env_async.client_sock, addr = await asyncio.wait_for(
                    loop.run_in_executor(None, self.server_sock.accept), timeout=9.8
                )
                self._wlog(f"accept done from {addr}")

                raw_obs = await asyncio.wait_for(
                    loop.run_in_executor(None, recv_exact, self.env_async.client_sock, self.env_async.state_recv_bytes),
                    timeout=30.0,
                )
                if not raw_obs or len(raw_obs) < self.env_async.state_recv_bytes:
                    raise RuntimeError("initial observation recv failed")
                _, obs = self.env_async._parse_assessment_state(raw_obs)
                obs.append(self.env_async.last_total_torque)
                obs.append(self.env_async.last_steering_cmd)
                obs.append(self.env_async.last_yaw_rate_sq_error)
                self.env_async.last_obs = obs

                self._wlog("initial obs recv done")
                self.env_async.ready_evt.set()
                self._wlog("ready_evt set - waiting for sim idle or reset_req")

                # idle 조건 대기: CM 자연 종료 시 reset_req로 탈출
                idle_task = asyncio.ensure_future(
                    simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()
                )
                reset_task = asyncio.ensure_future(self.env_async.reset_req.wait())
                done_set, pending = await asyncio.wait(
                    [idle_task, reset_task], timeout=300.0, return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                if not done_set:
                    self._wlog("timeout waiting for sim idle")
                elif idle_task in done_set:
                    self._wlog("sim idle reached")
                else:
                    self._wlog("reset_req fired before sim idle (CM closed connection)")

            except Exception as e:
                self._wlog(f"orchestrator error: {type(e).__name__}: {e}")
                self.env_async.last_obs = np.zeros(self.env_async.observation_space.shape, dtype=np.float32)
                self.env_async.ready_evt.set()
            finally:
                if self.env_async.client_sock:
                    self.env_async.client_sock.close()
                    self.env_async.client_sock = None

                print(f"[SHARED-WORKER-{self.worker_id}] Orchestrator: Episode Finished.")
                self._wlog("episode stop_sim begin")
                try:
                    await simcontrol.stop_sim()
                except Exception as e:
                    self._wlog(f"stop_sim error (ignored): {e}")
                self._wlog("episode stop_sim done")

        # Orchestrator 종료 시점에만 disconnect
        self._wlog("worker shutdown disconnect begin")
        try:
            await simcontrol.stop_sim()
        except Exception:
            pass
        try:
            await simcontrol.disconnect()
        except Exception as e:
            self._wlog(f"disconnect error (ignored): {e}")
        self._wlog("worker shutdown disconnect done")

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
        self.server_sock.settimeout(9.8)  # accept() timeout

        testrun = cmapi.Project.instance().load_testrun_parametrization(
            self.project_path / f"Data/TestRun/{self.testrun_name}"
        )
        variation = cmapi.Variation.create_from_testrun(testrun)

        self.env_async = CarMaker4WIDEnv(self._loop)
        self.env_async.stop_req.clear()
        self._orchestrator_task = asyncio.create_task(self._headless_orchestrator(variation))
        self._orchestrator_task.add_done_callback(self._on_orchestrator_done)
        self.env_sync = SyncBridge(self.env_async, self._loop)

        print(
            f"[SHARED-WORKER-{self.worker_id}] RUN_MODE={RUN_MODE_HEADLESS} "
            f"| CM_PORT={self.cm_port} | EXEC={self.cm_executable_path.name} "
            f"| PY_PID={os.getpid()} | THREAD_ID={threading.get_ident()}"
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
    project_path: str,
    testrun_name: str,
    cm_port_base: int,
    cm_executable_path: str,
) -> Callable[[], gym.Env]:
    def _init():
        return HeadlessWorkerEnv(
            worker_id=worker_id,
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
    env.setdefault("CM_PORT_BASE", "5556")
    env.setdefault("CM_EXECUTABLE_PATH", "/home/khkhh/CM_Projects/assessment_of_4wid_ver14/src/CarMaker.linux64")
    env.setdefault("NUM_WORKERS", env.get("NUM_WORKERS", "1"))

    # Default: isolate each worker's model/log files to avoid write races.
    model_mode = env.setdefault("MULTI_MODEL_MODE", "per_worker").strip().lower()
    if model_mode == "per_worker":
        base = os.getenv("MODEL_BASENAME", "carmaker_sac_4wid_dyc")
        tb = os.getenv("TB_DIRNAME", "carmaker_sac_4wid_dyc_tensorboard")
        env.setdefault("MODEL_BASENAME", f"{base}_w{worker_id}")
        env.setdefault("TB_DIRNAME", f"{tb}_w{worker_id}")

    return env


def run_shared_training(num_workers: int, profile: str):
    algo = os.getenv("ALGO", "ppo").strip().lower()
    if algo not in {"ppo", "sac", "td3"}:
        raise ValueError(f"ALGO는 'ppo', 'sac', 'td3' 이어야 합니다. 받은 값: {algo}")

    if algo == "ppo":
        algo_kwargs = PPO_PRESETS[profile]
        model_class = PPO
    elif algo == "sac":
        algo_kwargs = SAC_PRESETS[profile]
        model_class = SAC
    else:
        algo_kwargs = TD3_PRESETS[profile]
        model_class = TD3

    project_path = os.getenv("CM_PROJECT_PATH", "/home/khkhh/CM_Projects/assessment_of_4wid_ver14")
    testrun_name = os.getenv("TESTRUN_NAME", "test1")
    cm_port_base = int(os.getenv("CM_PORT_BASE", "5556"))
    cm_executable_path = os.getenv(
        "CM_EXECUTABLE_PATH",
        "/home/khkhh/CM_Projects/assessment_of_4wid_ver14/src/CarMaker.linux64",
    )
    progress_every_steps = int(os.getenv("SHARED_PROGRESS_EVERY_STEPS", "200"))
    heartbeat_sec = float(os.getenv("SHARED_HEARTBEAT_SEC", "5"))

    model_basename = os.getenv("MODEL_BASENAME", MODEL_BASENAME)
    tb_dir = os.getenv("TENSORBOARD_LOG", f"{model_basename}_tensorboard")
    log_dir = str(PROJECT_DIR / tb_dir)
    os.makedirs(log_dir, exist_ok=True)

    model_path = str(PROJECT_DIR / f"{model_basename}.zip")
    best_model_path = str(PROJECT_DIR / f"{model_basename}_best.zip")
    latest_model_path = str(PROJECT_DIR / f"{model_basename}_latest.zip")
    interrupt_model_path = str(PROJECT_DIR / f"{model_basename}_interrupt.zip")
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "5000000"))

    env_fns = [
        make_shared_worker_env(
            worker_id=i,
            project_path=project_path,
            testrun_name=testrun_name,
            cm_port_base=cm_port_base,
            cm_executable_path=cm_executable_path,
        )
        for i in range(num_workers)
    ]
    vec_env = VecMonitor(
        SubprocVecEnv(env_fns, start_method="spawn"),
        info_keywords=("yaw_penalty", "effort_penalty", "saturation_penalty"),
    )
    best_callback = SaveBestEpisodeRewardVecCallback(
        best_model_path=best_model_path, latest_model_path=latest_model_path, n_envs=num_workers
    )
    progress_callback = SharedProgressCallback(log_every_steps=progress_every_steps)
    heartbeat_callback = SharedHeartbeatCallback(interval_sec=heartbeat_sec)
    callback = CallbackList([best_callback, progress_callback, heartbeat_callback])

    print(f"[SHARED] ALGO={algo.upper()} | PROFILE={profile} | NUM_WORKERS={num_workers}")
    print(f"[SHARED] ALGO settings: {algo_kwargs}")
    print(f"[SHARED] MODEL_BASENAME={model_basename} | TENSORBOARD_LOG={log_dir}")
    print(
        f"[SHARED] CM_PROJECT_PATH={project_path} | TESTRUN_NAME={testrun_name} "
        f"| CM_EXECUTABLE_PATH={cm_executable_path} | CM_PORT_BASE={cm_port_base}"
    )
    print(f"[SHARED] PROGRESS_LOG_EVERY={progress_every_steps} timesteps | HEARTBEAT_EVERY={heartbeat_sec:.1f}s")

    try:
        # LOAD_MODEL_PATH: 로드할 체크포인트 경로 (저장은 model_path/best_model_path 유지)
        load_path = os.getenv("LOAD_MODEL_PATH", "").strip()
        if load_path and os.path.exists(load_path):
            print(f"[SHARED] 체크포인트 로드: '{load_path}' -> 저장 경로: '{model_path}'")
            model = model_class.load(load_path, env=vec_env, device="cpu", verbose=1, tensorboard_log=log_dir)
        elif os.path.exists(model_path):
            print(f"[SHARED] Loading existing model: {model_path}")
            model = model_class.load(model_path, env=vec_env, device="cpu", verbose=1, tensorboard_log=log_dir)
        else:
            if load_path:
                print(f"[WARN] LOAD_MODEL_PATH='{load_path}' 파일 없음. 새 모델로 시작합니다.")
            print(f"[SHARED] No model found. Creating a new {algo.upper()} model.")
            model = model_class("MlpPolicy", vec_env, verbose=1, device="cpu", tensorboard_log=log_dir, **algo_kwargs)

        if algo == "ppo":
            model.n_steps = algo_kwargs["n_steps"]
            model.batch_size = algo_kwargs["batch_size"]
            model.gamma = algo_kwargs["gamma"]
            model.learning_rate = algo_kwargs["learning_rate"]
            model.ent_coef = algo_kwargs["ent_coef"]
        elif algo == "sac":
            model.batch_size = algo_kwargs["batch_size"]
            model.gamma = algo_kwargs["gamma"]
            model.learning_rate = algo_kwargs["learning_rate"]
            model.ent_coef = algo_kwargs["ent_coef"]
            model.tau = algo_kwargs["tau"]
            model.buffer_size = algo_kwargs["buffer_size"]
            model.learning_starts = algo_kwargs["learning_starts"]
            model.gradient_steps = algo_kwargs["gradient_steps"]
        else:  # td3
            model.batch_size = algo_kwargs["batch_size"]
            model.gamma = algo_kwargs["gamma"]
            model.learning_rate = algo_kwargs["learning_rate"]
            model.tau = algo_kwargs["tau"]
            model.buffer_size = algo_kwargs["buffer_size"]
            model.learning_starts = algo_kwargs["learning_starts"]
            model.gradient_steps = algo_kwargs["gradient_steps"]
            model.policy_delay = algo_kwargs["policy_delay"]
            model.target_policy_noise = algo_kwargs["target_policy_noise"]
            model.target_noise_clip = algo_kwargs["target_noise_clip"]
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
            f"[WORKER-{worker_id}] CM_PORT={int(env.get('CM_PORT_BASE', '5556')) + worker_id}"
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
    num_workers = resolve_num_workers(default_value=1)
    profile = get_ppo_profile()
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

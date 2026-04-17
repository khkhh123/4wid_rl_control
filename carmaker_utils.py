"""
CarMaker 4WID RL 인프라 유틸리티.

이 파일은 소켓 통신, 컨트롤러, 콜백, 오케스트레이터 등
수정이 거의 필요 없는 인프라 코드를 담고 있습니다.
보상 함수·시나리오·알고리즘 프리셋을 바꾸려면 train_with_cm_gui.py를 수정하세요.
"""

import os
import sys
import asyncio
import socket
import struct
from pathlib import Path

import psutil
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi  # noqa: E402  (CarMaker Python API)

# TensorBoard 중간 로깅 주기 (env var로 덮어쓰기 가능)
_LOG_INTERVAL_STEPS = int(os.getenv("LOG_INTERVAL_STEPS", "500"))


# ──────────────────────────────────────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────────────────────────────────────

def wrap_to_pi(angle: float) -> float:
    """각도를 [-π, π] 범위로 정규화."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def recv_exact(sock_obj: socket.socket, nbytes: int) -> bytes:
    """소켓에서 정확히 nbytes를 수신. 연결 끊김 시 빈 bytes 반환."""
    chunks = []
    received = 0
    while received < nbytes:
        chunk = sock_obj.recv(nbytes - received)
        if not chunk:
            return b""
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def get_carmaker_pid() -> int | None:
    """실행 중인 CarMaker.linux64 프로세스의 PID를 반환. 없으면 None."""
    target = "CarMaker.linux64"
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = proc.info.get("cmdline") or []
            if target in name or any(target in arg for arg in cmdline):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 상위 제어기 (속도 PI + Stanley 조향)
# ──────────────────────────────────────────────────────────────────────────────

class PISpeedController:
    """속도 오차 → 총 구동 토크 변환 PI 제어기."""

    def __init__(self):
        self.kp             = float(os.getenv("CTRL_KP",         "1000.0"))
        self.ki             = float(os.getenv("CTRL_KI",         "200.0"))
        self.kd             = float(os.getenv("CTRL_KD",         "1.0"))
        self.integral_limit = float(os.getenv("INTEGRAL_LIMIT",  "2000.0"))
        wheel_limit         = float(os.getenv("WHEEL_TORQUE_LIMIT", "630.0"))
        self._total_limit   = float(os.getenv("TOTAL_TORQUE_LIMIT", str(wheel_limit * 4.0)))
        self.integral       = 0.0
        self.prev_error     = 0.0

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute_total_torque(self, v_diff_mps: float, dt: float) -> float:
        self.integral = float(np.clip(
            self.integral + v_diff_mps * dt,
            -self.integral_limit,
            self.integral_limit,
        ))
        derivative = (v_diff_mps - self.prev_error) / max(dt, 1e-9)
        torque = self.kp * v_diff_mps + self.ki * self.integral + self.kd * derivative
        self.prev_error = v_diff_mps
        return float(np.clip(torque, -self._total_limit, self._total_limit))


class StanleySteeringController:
    """Stanley 횡방향 제어 + 동역학 기반 reference yaw rate 계산."""

    def __init__(self):
        self.g1            = float(os.getenv("STANLEY_G1",           "27"))
        self.g2            = float(os.getenv("STANLEY_G2",           "5.0"))
        self.k             = float(os.getenv("STANLEY_K",            "5.0"))
        self.scale_factor  = float(os.getenv("STANLEY_SCALE",        "1.0"))
        self.mass_kg       = float(os.getenv("VEH_MASS_KG",          "2065.03"))
        self.a_m           = float(os.getenv("VEH_A_M",              "1.169"))
        self.b_m           = float(os.getenv("VEH_B_M",              "1.801"))
        _default_wb        = str(self.a_m + self.b_m)
        self.wheelbase     = float(os.getenv("STANLEY_WHEELBASE_M",  _default_wb))  # 축거 (m)
        # self.cf            = float(os.getenv("VEH_CF",               "180000.0")) * 2
        # self.cr            = float(os.getenv("VEH_CR",               "120000.0")) * 2
        self.cf = float(os.getenv("VEH_CF", 143000.0))
        self.cr = float(os.getenv("VEH_CR", 143000.0))
        self.steer_ratio   = float(os.getenv("VEH_STEER_RATIO",      "16.0"))
        self.mu            = float(os.getenv("VEH_MU",               "0.95"))
        self.g             = 9.81
        self.min_turn_radius_m = float(os.getenv("VEH_MIN_TURN_RADIUS_M", "10.0"))

    def compute(
        self,
        veh_x: float, veh_y: float, veh_yaw: float,
        ref_x: float, ref_y: float, ref_yaw: float,
        vel: float,
    ):
        # 전륜 축 위치 기준으로 횡방향 오차 계산
        L_f = self.a_m
        fx  = veh_x + L_f * np.cos(veh_yaw)
        fy  = veh_y + L_f * np.sin(veh_yaw)

        cross_track_err  = (ref_y - fy) * np.cos(ref_yaw) - (ref_x - fx) * np.sin(ref_yaw)
        heading_err      = wrap_to_pi(ref_yaw - veh_yaw)
        heading_err      = float(np.clip(heading_err, -np.pi / 2, np.pi / 2))

        vel_term_heading = 20.0 / (float(vel) + 1.0)
        vel_term_cte     = (self.k * cross_track_err) / (float(vel) / 10.0 + 2.2) ** 2
        steering_cmd     = self.scale_factor * (
            self.g1 * heading_err * vel_term_heading + self.g2 * vel_term_cte
        )
        return float(steering_cmd), float(heading_err), float(cross_track_err)

    def compute_reference_yaw_rate_both(self, veh_speed: float, steering_cmd: float):
        """(yawrate_ideal_비클램프, yawrate_클램프) 반환."""
        vx = abs(float(veh_speed))
        if vx < 1e-6:
            return 0.0, 0.0
        delta_f = float(steering_cmd) / max(self.steer_ratio, 1e-6)
        ab = self.a_m + self.b_m
        numerator   = vx * self.cf * self.cr * ab * delta_f
        denominator = (self.cf * self.cr * (ab ** 2)) + (
            self.mass_kg * (vx ** 2) * (self.b_m * self.cr - self.a_m * self.cf)
        )
        yawrate_ideal = 0.0 if abs(denominator) < 1e-9 else numerator / denominator
        yawrate_max = min(
            abs(self.mu * self.g / vx),
            abs(vx / max(self.min_turn_radius_m, 1e-6)),
        )
        clamped = float(min(abs(yawrate_ideal), yawrate_max) * np.sign(delta_f))
        return float(yawrate_ideal), clamped


# ──────────────────────────────────────────────────────────────────────────────
# 시나리오 프로파일 (CSV 기반 경로 추종)
# ──────────────────────────────────────────────────────────────────────────────

class ScenarioProfile:
    """CSV에서 시나리오 reference state를 읽어 선형 보간으로 제공."""

    def __init__(self):
        self.time_arr  = None
        self.x_arr     = None
        self.y_arr     = None
        self.speed_arr = None
        self.yaw_arr   = None
        self.last_idx  = 0

    def reset(self):
        self.last_idx = 0

    def load_csv(self, csv_path: Path) -> bool:
        if not csv_path.exists():
            return False
        try:
            data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Failed to parse scenario CSV: {csv_path} ({e})")
            return False
        if data is None or data.size < 2:
            print(f"[WARN] Scenario CSV has insufficient rows: {csv_path}")
            return False
        names    = list(data.dtype.names or [])
        name_map = {n.lower(): n for n in names}
        if any(k not in name_map for k in ["time", "x", "y", "speed", "yaw"]):
            print(f"[WARN] Scenario CSV missing columns (Time/X/Y/Speed/Yaw): {csv_path}")
            return False
        self.time_arr  = np.asarray(data[name_map["time"]],  dtype=np.float64)
        self.x_arr     = np.asarray(data[name_map["x"]],     dtype=np.float64)
        self.y_arr     = np.asarray(data[name_map["y"]],     dtype=np.float64)
        self.speed_arr = np.asarray(data[name_map["speed"]], dtype=np.float64)
        self.yaw_arr   = np.asarray(data[name_map["yaw"]],   dtype=np.float64)
        self.last_idx  = 0
        print(f"[SCENARIO] loaded: {csv_path.name} ({self.time_arr.size} points)")
        return True

    def get_reference_state(self, sim_time: float):
        if self.time_arr is None or self.time_arr.size < 2:
            return None
        if sim_time < self.time_arr[0] or sim_time > self.time_arr[-1]:
            return None
        i = int(self.last_idx)
        n = int(self.time_arr.size)
        while i + 1 < n and self.time_arr[i + 1] < sim_time:
            i += 1
        if i + 1 >= n:
            i = n - 2
        self.last_idx = i
        t0    = self.time_arr[i]
        t1    = self.time_arr[i + 1]
        alpha = 0.0 if t1 <= t0 else (sim_time - t0) / (t1 - t0)
        x     = float(self.x_arr[i]     + alpha * (self.x_arr[i + 1]     - self.x_arr[i]))
        y     = float(self.y_arr[i]     + alpha * (self.y_arr[i + 1]     - self.y_arr[i]))
        speed = float(self.speed_arr[i] + alpha * (self.speed_arr[i + 1] - self.speed_arr[i])) / 3.6
        yaw   = float(self.yaw_arr[i]   + alpha * (self.yaw_arr[i + 1]   - self.yaw_arr[i]))
        return {"x": x, "y": y, "speed_mps": speed, "yaw": yaw}


# ──────────────────────────────────────────────────────────────────────────────
# SB3 콜백 (GUI 단일 환경용)
# ──────────────────────────────────────────────────────────────────────────────

class SaveBestEpisodeRewardCallback(BaseCallback):
    """
    에피소드 종료마다 최고 보상 모델을 저장하고 TensorBoard에 기록.
    중간 단계 로깅: LOG_INTERVAL_STEPS 마다 step_reward / ep_reward_so_far 기록.
    """

    def __init__(self, best_model_path: str, latest_model_path: str | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.best_model_path      = best_model_path
        self.latest_model_path    = latest_model_path
        self.best_reward_path     = best_model_path + ".reward"
        self.current_episode_reward = 0.0
        self.best_episode_reward  = -np.inf
        self.episode_count        = 0
        self._log_interval        = int(os.getenv("LOG_INTERVAL_STEPS", str(_LOG_INTERVAL_STEPS)))
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
        dones   = self.locals.get("dones")
        infos   = self.locals.get("infos")
        if rewards is None or dones is None:
            return True

        self.current_episode_reward += float(rewards[0])

        # 중간 단계 로깅
        if self._log_interval > 0 and self.num_timesteps % self._log_interval == 0:
            self.logger.record("train/step_reward",      float(rewards[0]))
            self.logger.record("train/ep_reward_so_far", self.current_episode_reward)
            self.logger.dump(self.num_timesteps)

        if bool(dones[0]):
            self.episode_count += 1
            if self.latest_model_path:
                self.model.save(self.latest_model_path)
            if self.current_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.current_episode_reward
                self.model.save(self.best_model_path)
                self._save_best_reward()
                if self.verbose:
                    print(f"[BEST] episode={self.episode_count} reward={self.best_episode_reward:.3f} -> {self.best_model_path}")
            self.logger.record("episode/reward",       self.current_episode_reward)
            self.logger.record("episode/count",        self.episode_count)
            self.logger.record("episode/best_reward",  self.best_episode_reward)
            ep_info = infos[0].get("episode", {}) if infos else {}
            if ep_info:
                self.logger.record("episode/yaw_penalty",        ep_info.get("yaw_penalty",        0.0))
                self.logger.record("episode/effort_penalty",     ep_info.get("effort_penalty",     0.0))
                self.logger.record("episode/saturation_penalty", ep_info.get("saturation_penalty", 0.0))
                self.logger.record("episode/energy_penalty",     ep_info.get("energy_penalty",     0.0))
            self.logger.dump(self.num_timesteps)
            if self.verbose:
                yaw_pen = ep_info.get("yaw_penalty",    0.0) if ep_info else 0.0
                eff_pen = ep_info.get("effort_penalty", 0.0) if ep_info else 0.0
                sat_pen = ep_info.get("saturation_penalty", 0.0) if ep_info else 0.0
                eng_pen = ep_info.get("energy_penalty", 0.0) if ep_info else 0.0
                print(f"[EP {self.episode_count}] reward={self.current_episode_reward:.3f} "
                      f"| yaw={yaw_pen:.3f} effort={eff_pen:.3f} sat={sat_pen:.3f} energy={eng_pen:.3f}")
            self.current_episode_reward = 0.0
        return True


# ──────────────────────────────────────────────────────────────────────────────
# SyncBridge: async env → sync (SB3용)
# ──────────────────────────────────────────────────────────────────────────────

class SyncBridge(gym.Env):
    """비동기 CarMaker4WIDEnv를 동기 인터페이스로 래핑 (stable_baselines3 호환)."""

    def __init__(self, a_env, loop):
        self.a_env             = a_env
        self.loop              = loop
        self.action_space      = a_env.action_space
        self.observation_space = a_env.observation_space

    def reset(self, **kwargs):
        return asyncio.run_coroutine_threadsafe(self.a_env.reset(**kwargs), self.loop).result()

    def step(self, action):
        return asyncio.run_coroutine_threadsafe(self.a_env.step(action), self.loop).result()

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# CarMaker 오케스트레이터 (GUI 모드 전용)
# ──────────────────────────────────────────────────────────────────────────────

async def carmaker_orchestrator(env, simcontrol, variation, server_sock):
    """
    GUI 모드에서 에피소드마다 CarMaker 시뮬레이션을 시작/종료하는 비동기 루프.
    env.reset_req 이벤트를 기다렸다가 새 시뮬레이션을 시작하고
    idle 상태가 되면 소켓을 닫는다.
    """
    loop = asyncio.get_running_loop()
    realtime_factor = float(os.getenv("REALTIME_FACTOR", "100.0"))

    while True:
        await env.reset_req.wait()
        env.reset_req.clear()

        await simcontrol.start_and_connect()
        print("\n[Orchestrator] Starting New Episode...")
        simcontrol.set_variation(variation.clone())
        print(simcontrol.get_status())
        await asyncio.sleep(1.0)
        await simcontrol.start_sim()
        simcontrol.set_realtimefactor(realtime_factor)

        env.client_sock, _ = await loop.run_in_executor(None, server_sock.accept)
        raw_obs = await loop.run_in_executor(None, recv_exact, env.client_sock, env.state_recv_bytes)

        from train_with_cm_gui import OBS_SCALE
        _, obs = env._parse_assessment_state(raw_obs)
        obs.append(env.last_total_torque)
        obs.append(env.last_steering_cmd)
        obs.append(env.last_yaw_rate_sq_error)
        env.last_obs = (np.asarray(obs, dtype=np.float32) / OBS_SCALE).tolist()
        env.ready_evt.set()

        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

        if env.client_sock:
            print("[Orchestrator] Closing client socket...")
            env.client_sock.close()
            env.client_sock = None
        print("[Orchestrator] Episode Finished & Cleaned up.")
        await simcontrol.disconnect()


# ──────────────────────────────────────────────────────────────────────────────
# 모터 효율맵
# ──────────────────────────────────────────────────────────────────────────────

class MotorMap:
    """모터 토크 한계 및 효율 맵 로더. batt_net_power 계산에 필요한 메서드만 포함."""

    def __init__(self, map_path: Path):
        self.map_path  = map_path
        self.loaded    = False
        self.spd_rpm   = None
        self.trq_max_nm = None
        self.eff_spd_rpm = None
        self.eff_trq_nm  = None
        self.eff_map_pct = None

    def load(self) -> bool:
        if not self.map_path.exists():
            print(f"[WARN] Motor map not found: {self.map_path}")
            return False
        try:
            import scipy.io as sio
            from scipy.interpolate import RegularGridInterpolator
            m = sio.loadmat(self.map_path)
        except Exception as e:
            print(f"[WARN] Failed to load motor map: {e}")
            return False
        try:
            self.spd_rpm     = np.asarray(m["spd_rpm"],    dtype=np.float64).reshape(-1)
            self.trq_max_nm  = np.asarray(m["trq_max_Nm"], dtype=np.float64).reshape(-1)
            self.eff_spd_rpm = np.asarray(m["eff_spd_rpm"], dtype=np.float64).reshape(-1)
            self.eff_trq_nm  = np.asarray(m["trq_Nm"],      dtype=np.float64).reshape(-1)
            eff_raw = np.asarray(m.get("calc_eff", m.get("original_eff")), dtype=np.float64)
            if eff_raw.ndim != 2:
                raise ValueError("Efficiency map must be 2D")
            self.eff_map_pct = eff_raw  # shape: (n_trq, n_spd)
            # 2D 보간기를 미리 빌드 → efficiency_pct 호출 시 단일 C 호출로 처리
            self._eff_interp = RegularGridInterpolator(
                (self.eff_trq_nm, self.eff_spd_rpm),
                self.eff_map_pct,
                method="linear",
                bounds_error=False,
                fill_value=None,  # 범위 밖은 경계값으로 클램프
            )
            self.loaded = True
            print(f"[INFO] Motor map loaded: {self.map_path}")
            return True
        except Exception as e:
            print(f"[WARN] Invalid motor map contents: {e}")
            return False

    def max_torque_nm(self, rpm: float) -> float:
        if not self.loaded:
            return float("inf")
        r = abs(float(rpm))
        return float(np.interp(r, self.spd_rpm, self.trq_max_nm,
                               left=self.trq_max_nm[0], right=self.trq_max_nm[-1]))

    def efficiency_pct(self, rpm: float, torque_nm: float) -> float:
        if not self.loaded:
            return 100.0
        t = abs(float(torque_nm))
        if t <= 1e-9:
            return 100.0
        t_clip = float(np.clip(t, self.eff_trq_nm[0],  self.eff_trq_nm[-1]))
        r_clip = float(np.clip(abs(float(rpm)), self.eff_spd_rpm[0], self.eff_spd_rpm[-1]))
        eff = float(self._eff_interp([[t_clip, r_clip]])[0])
        return float(np.clip(eff, 1.0, 100.0))


def compute_batt_net_power(wheel_torques: np.ndarray, rotv: np.ndarray, motor_map: "MotorMap") -> float:
    """wheel_torques(Nm) × rotv(rad/s) + 모터 효율맵 → 배터리 순 소비 전력(W)."""
    batt_drive = 0.0
    batt_regen = 0.0
    for w in range(4):
        omega     = abs(float(rotv[w]))
        rpm       = omega * 60.0 / (2.0 * np.pi)
        tq        = float(wheel_torques[w])
        tq_max    = motor_map.max_torque_nm(rpm)
        sat_drive = min(max(tq, 0.0), tq_max)
        sat_regen = min(max(-tq, 0.0), tq_max)
        if sat_drive > 1e-9:
            eff = max(motor_map.efficiency_pct(rpm, sat_drive) / 100.0, 1e-3)
            batt_drive += (sat_drive * omega) / eff
        if sat_regen > 1e-9:
            eff = max(motor_map.efficiency_pct(rpm, sat_regen) / 100.0, 1e-3)
            batt_regen += (sat_regen * omega) * eff
    return batt_drive - batt_regen


def compute_batt_net_power_batch(
    tq_matrix: np.ndarray,   # (N, 4) wheel torques
    omega_vec: np.ndarray,   # (N,)   wheel angular velocity (rad/s, same for all 4)
    motor_map: "MotorMap",
) -> np.ndarray:
    """
    벡터화된 배터리 순 소비 전력 계산. (N,) → batt_net[i] = batt_drive[i] - batt_regen[i]
    motor_map 미로드 시 효율 100% 가정 (scalar compute와 동일 fallback).
    """
    N = len(omega_vec)
    omega_abs = np.abs(omega_vec)
    rpm_vec   = omega_abs * (60.0 / (2.0 * np.pi))
    batt      = np.zeros(N, dtype=np.float64)

    if motor_map.loaded:
        tq_max_vec = np.interp(rpm_vec, motor_map.spd_rpm, motor_map.trq_max_nm)
        r_clip = np.clip(rpm_vec, motor_map.eff_spd_rpm[0], motor_map.eff_spd_rpm[-1])
    else:
        tq_max_vec = np.full(N, np.inf)
        r_clip     = None

    for w in range(4):
        tq    = tq_matrix[:, w].astype(np.float64)
        drive = np.clip(tq,  0.0, tq_max_vec)
        regen = np.clip(-tq, 0.0, tq_max_vec)

        d_mask = drive > 1e-9
        r_mask = regen > 1e-9

        if motor_map.loaded:
            if d_mask.any():
                t_d = np.clip(drive[d_mask], motor_map.eff_trq_nm[0], motor_map.eff_trq_nm[-1])
                eff = np.clip(motor_map._eff_interp(
                    np.column_stack([t_d, r_clip[d_mask]])) / 100.0, 1e-3, 1.0)
                batt[d_mask] += (drive[d_mask] * omega_abs[d_mask]) / eff
            if r_mask.any():
                t_r = np.clip(regen[r_mask], motor_map.eff_trq_nm[0], motor_map.eff_trq_nm[-1])
                eff = np.clip(motor_map._eff_interp(
                    np.column_stack([t_r, r_clip[r_mask]])) / 100.0, 1e-3, 1.0)
                batt[r_mask] -= (regen[r_mask] * omega_abs[r_mask]) * eff
        else:
            batt[d_mask] += drive[d_mask] * omega_abs[d_mask]
            batt[r_mask] -= regen[r_mask] * omega_abs[r_mask]

    return batt

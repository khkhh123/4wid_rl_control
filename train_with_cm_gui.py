import sys
import asyncio
import struct
import socket
import psutil
import os
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi


# 액션/보상/제어 파라미터
ACTION_EFFORT_WEIGHT = float(os.getenv("ACTION_EFFORT_WEIGHT", "1e-0"))
YAW_RATE_WEIGHT = float(os.getenv("YAW_RATE_WEIGHT", "1e+2"))
SATURATION_PENALTY_WEIGHT = float(os.getenv("SATURATION_PENALTY_WEIGHT", "1e-2"))
YAW_RATE_NORM_EPS = float(os.getenv("YAW_RATE_NORM_EPS", "0.01"))

WHEEL_TORQUE_LIMIT = float(os.getenv("WHEEL_TORQUE_LIMIT", "630.0"))
TOTAL_TORQUE_LIMIT = float(os.getenv("TOTAL_TORQUE_LIMIT", str(WHEEL_TORQUE_LIMIT * 4.0)))

CONTROL_DT = float(os.getenv("CONTROL_DT", "0.05"))

# 상위 속도 PI 제어기 파라미터
CTRL_KP = float(os.getenv("CTRL_KP", "1000.0"))
CTRL_KI = float(os.getenv("CTRL_KI", "200.0"))
CTRL_KD = float(os.getenv("CTRL_KD", "1.0"))
INTEGRAL_LIMIT = float(os.getenv("INTEGRAL_LIMIT", "2000.0"))

# 시나리오 로드 경로
SCENARIO_DIR = os.getenv("SCENARIO_DIR", "scenarios").strip()
SCENARIO_CSV_PATH = os.getenv("SCENARIO_CSV_PATH", "").strip()

# Open-loop steering mode (scenario 대체)
OPEN_LOOP_STEER = os.getenv("OPEN_LOOP_STEER", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
OPEN_LOOP_STEER_AMP_DEG = float(os.getenv("OPEN_LOOP_STEER_AMP_DEG", "60.0"))
OPEN_LOOP_STEER_PERIOD_S = float(os.getenv("OPEN_LOOP_STEER_PERIOD_S", "4.0"))
OPEN_LOOP_STEER_START_S = float(os.getenv("OPEN_LOOP_STEER_START_S", "0.0"))
OPEN_LOOP_STEER_CYCLES = max(0, int(float(os.getenv("OPEN_LOOP_STEER_CYCLES", "700.0"))))
OPEN_LOOP_SPEED_KPH = float(os.getenv("OPEN_LOOP_SPEED_KPH", "80.0"))
OPEN_LOOP_SPEED_MODE = os.getenv("OPEN_LOOP_SPEED_MODE", "mixed").strip().lower()  # constant | mixed
OPEN_LOOP_SPEED_MIN_KPH = float(os.getenv("OPEN_LOOP_SPEED_MIN_KPH", "0.0"))
OPEN_LOOP_SPEED_MAX_KPH = float(os.getenv("OPEN_LOOP_SPEED_MAX_KPH", "60.0"))
OPEN_LOOP_ACCEL_DURATION_S = float(os.getenv("OPEN_LOOP_ACCEL_DURATION_S", "10.0"))
OPEN_LOOP_CRUISE_HIGH_DURATION_S = float(os.getenv("OPEN_LOOP_CRUISE_HIGH_DURATION_S", "40.0"))
OPEN_LOOP_DECEL_DURATION_S = float(os.getenv("OPEN_LOOP_DECEL_DURATION_S", "5.0"))
OPEN_LOOP_CRUISE_LOW_DURATION_S = float(os.getenv("OPEN_LOOP_CRUISE_LOW_DURATION_S", "5.0"))

# Stanley 제어기 파라미터 (custom_controller 기준)
STANLEY_G1 = float(os.getenv("STANLEY_G1", "30.0"))
STANLEY_G2 = float(os.getenv("STANLEY_G2", "5.0"))
STANLEY_K = float(os.getenv("STANLEY_K", "1.0"))
STANLEY_SCALE = float(os.getenv("STANLEY_SCALE", "10.0"))

# 차량 동역학 파라미터 (reference yaw rate 계산용)
VEH_MASS_KG = float(os.getenv("VEH_MASS_KG", "2065.03"))
VEH_A_M = float(os.getenv("VEH_A_M", "1.169"))       # 무게중심 ~ 전축 거리 [m]
VEH_B_M = float(os.getenv("VEH_B_M", "1.801"))       # 무게중심 ~ 후축 거리 [m]
VEH_CF = float(os.getenv("VEH_CF", "180000.0"))*2    # 전축 코너링 강성 [N/rad] (pre-scale)
VEH_CR = float(os.getenv("VEH_CR", "120000.0"))*2    # 후축 코너링 강성 [N/rad] (pre-scale)
VEH_STEER_RATIO = float(os.getenv("VEH_STEER_RATIO", "16.0"))  # 조향비
VEH_MU = float(os.getenv("VEH_MU", "0.95"))        # 노면 마찰계수
VEH_MIN_TURN_RADIUS_M = float(os.getenv("VEH_MIN_TURN_RADIUS_M", "10.0"))  # 최소 선회반경 [m]

# MODEL_BASENAME = os.getenv("MODEL_BASENAME", "carmaker_ppo_4wid_dyc").strip()
MODEL_BASENAME = os.getenv("MODEL_BASENAME", "carmaker_td3_4wid_dyc").strip()

PPO_PRESETS = {
	"A": {
		"learning_rate": 1e-4,
		"batch_size": 256,
		"gamma": 0.99,
		"n_steps": 1024,
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
}

SAC_PRESETS = {
	"A": {
		"learning_rate": 3e-4,
		"batch_size": 256,
		"gamma": 0.99,
		"ent_coef": "auto",
		"tau": 0.005,
		"buffer_size": 300000,
		"learning_starts": 5000,
		"train_freq": (1, "step"),
		"gradient_steps": 1,
	},
	"B": {
		"learning_rate": 1e-4,
		"batch_size": 512,
		"gamma": 0.995,
		"ent_coef": "auto",
		"tau": 0.005,
		"buffer_size": 500000,
		"learning_starts": 10000,
		"train_freq": (1, "step"),
		"gradient_steps": 1,
	},
	"C": {
		"learning_rate": 3e-4,
		"batch_size": 256,
		"gamma": 0.99,
		"ent_coef": "auto_0.1",
		"tau": 0.01,
		"buffer_size": 300000,
		"learning_starts": 5000,
		"train_freq": (1, "step"),
		"gradient_steps": 2,
	},
}

TD3_PRESETS = {
	"A": {
		"learning_rate": 3e-4,
		"batch_size": 256,
		"gamma": 0.99,
		"tau": 0.005,
		"buffer_size": 300000,
		"learning_starts": 5000,
		"train_freq": (1, "step"),
		"gradient_steps": 1,
		"policy_delay": 2,
		"target_policy_noise": 0.2,
		"target_noise_clip": 0.5,
	},
	"B": {
		"learning_rate": 1e-4,
		"batch_size": 512,
		"gamma": 0.995,
		"tau": 0.005,
		"buffer_size": 500000,
		"learning_starts": 10000,
		"train_freq": (1, "step"),
		"gradient_steps": 1,
		"policy_delay": 2,
		"target_policy_noise": 0.2,
		"target_noise_clip": 0.5,
	},
	"C": {
		"learning_rate": 3e-4,
		"batch_size": 256,
		"gamma": 0.99,
		"tau": 0.01,
		"buffer_size": 300000,
		"learning_starts": 5000,
		"train_freq": (1, "step"),
		"gradient_steps": 2,
		"policy_delay": 2,
		"target_policy_noise": 0.1,
		"target_noise_clip": 0.5,
	},
}


class PISpeedController:
	def __init__(self):
		self.kp = CTRL_KP
		self.ki = CTRL_KI
		self.kd = CTRL_KD
		self.integral_limit = INTEGRAL_LIMIT
		self.integral = 0.0
		self.prev_error = 0.0

	def reset(self):
		self.integral = 0.0
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
		return float(np.clip(torque, -TOTAL_TORQUE_LIMIT, TOTAL_TORQUE_LIMIT))


class StanleySteeringController:
	def __init__(self):
		self.g1 = STANLEY_G1
		self.g2 = STANLEY_G2
		self.k = STANLEY_K
		self.scale_factor = STANLEY_SCALE
		# 동역학 파라미터
		self.mass_kg = VEH_MASS_KG
		self.a_m = VEH_A_M
		self.b_m = VEH_B_M
		self.cf = VEH_CF
		self.cr = VEH_CR
		self.steer_ratio = VEH_STEER_RATIO
		self.mu = VEH_MU
		self.g = 9.81
		self.min_turn_radius_m = VEH_MIN_TURN_RADIUS_M

	def compute(
		self,
		veh_x: float,
		veh_y: float,
		veh_yaw: float,
		ref_x: float,
		ref_y: float,
		ref_yaw: float,
		vel: float,
	):
		cross_track_err = (ref_y - veh_y) * np.cos(ref_yaw) - (ref_x - veh_x) * np.sin(ref_yaw)
		heading_err = float(np.clip(ref_yaw - veh_yaw, -np.pi / 2, np.pi / 2))

		vel_term_heading = 20.0 / (float(vel) + 20.0)
		vel_term_cte = (self.k * cross_track_err) / (float(vel) + 3.0)
		steering_cmd = self.scale_factor * (self.g1 * heading_err * vel_term_heading + self.g2 * vel_term_cte)
		return float(steering_cmd), float(heading_err), float(cross_track_err)

	def compute_reference_yaw_rate(self, veh_speed: float, steering_cmd: float) -> float:
		vx = abs(float(veh_speed))
		if vx < 1e-6:
			return 0.0

		# Stanley output is steering-wheel angle, convert to front-wheel angle.
		delta_f = float(steering_cmd) / max(self.steer_ratio, 1e-6)

		ab = self.a_m + self.b_m
		numerator = vx * self.cf * self.cr * ab * delta_f
		denominator = (self.cf * self.cr * (ab**2)) + (
			self.mass_kg * (vx**2) * (self.b_m * self.cr - self.a_m * self.cf)
		)
		if abs(denominator) < 1e-9:
			yawrate_ideal = 0.0
		else:
			yawrate_ideal = numerator / denominator

		yawrate_max_friction = abs(self.mu * self.g / vx)
		yawrate_max_minturnrad = abs(vx / max(self.min_turn_radius_m, 1e-6))
		yawrate_max = min(yawrate_max_friction, yawrate_max_minturnrad)

		return float(min(abs(yawrate_ideal), yawrate_max) * np.sign(delta_f))

	def compute_reference_yaw_rate_both(self, veh_speed: float, steering_cmd: float):
		"""Returns (yawrate_ideal_unclamped, yawrate_clamped)."""
		vx = abs(float(veh_speed))
		if vx < 1e-6:
			return 0.0, 0.0
		delta_f = float(steering_cmd) / max(self.steer_ratio, 1e-6)
		ab = self.a_m + self.b_m
		numerator = vx * self.cf * self.cr * ab * delta_f
		denominator = (self.cf * self.cr * (ab**2)) + (
			self.mass_kg * (vx**2) * (self.b_m * self.cr - self.a_m * self.cf)
		)
		if abs(denominator) < 1e-9:
			yawrate_ideal = 0.0
		else:
			yawrate_ideal = numerator / denominator
		yawrate_max_friction = abs(self.mu * self.g / vx)
		yawrate_max_minturnrad = abs(vx / max(self.min_turn_radius_m, 1e-6))
		yawrate_max = min(yawrate_max_friction, yawrate_max_minturnrad)
		clamped = float(min(abs(yawrate_ideal), yawrate_max) * np.sign(delta_f))
		return float(yawrate_ideal), clamped


class ScenarioProfile:
	def __init__(self):
		self.time_arr = None
		self.x_arr = None
		self.y_arr = None
		self.speed_arr = None
		self.yaw_arr = None
		self.last_idx = 0

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

		names = list(data.dtype.names or [])
		name_map = {n.lower(): n for n in names}
		required = ["time", "x", "y", "speed", "yaw"]
		if any(k not in name_map for k in required):
			print(f"[WARN] Scenario CSV missing columns (Time/X/Y/Speed/Yaw): {csv_path}")
			return False

		self.time_arr = np.asarray(data[name_map["time"]], dtype=np.float64)
		self.x_arr = np.asarray(data[name_map["x"]], dtype=np.float64)
		self.y_arr = np.asarray(data[name_map["y"]], dtype=np.float64)
		self.speed_arr = np.asarray(data[name_map["speed"]], dtype=np.float64)
		self.yaw_arr = np.asarray(data[name_map["yaw"]], dtype=np.float64)
		self.last_idx = 0

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
		t0 = self.time_arr[i]
		t1 = self.time_arr[i + 1]
		alpha = 0.0 if t1 <= t0 else (sim_time - t0) / (t1 - t0)

		x = float(self.x_arr[i] + alpha * (self.x_arr[i + 1] - self.x_arr[i]))
		y = float(self.y_arr[i] + alpha * (self.y_arr[i + 1] - self.y_arr[i]))
		speed_mps = float(self.speed_arr[i] + alpha * (self.speed_arr[i + 1] - self.speed_arr[i])) / 3.6
		yaw = float(self.yaw_arr[i] + alpha * (self.yaw_arr[i + 1] - self.yaw_arr[i]))
		dt_seg = t1 - t0
		yaw_rate = wrap_to_pi(float(self.yaw_arr[i + 1] - self.yaw_arr[i])) / dt_seg if dt_seg > 1e-9 else 0.0

		return {"x": x, "y": y, "speed_mps": speed_mps, "yaw": yaw}

class SaveBestEpisodeRewardCallback(BaseCallback):
	def __init__(self, best_model_path, latest_model_path=None, verbose=1):
		super().__init__(verbose)
		self.best_model_path = best_model_path
		self.latest_model_path = latest_model_path
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
		infos = self.locals.get("infos")
		if rewards is None or dones is None:
			return True

		self.current_episode_reward += float(rewards[0])
		if bool(dones[0]):
			self.episode_count += 1
			# 매 에피소드마다 최신 모델 저장
			if self.latest_model_path:
				self.model.save(self.latest_model_path)
			# 최고 보상 갱신 시 best 모델 저장
			if self.current_episode_reward > self.best_episode_reward:
				self.best_episode_reward = self.current_episode_reward
				self.model.save(self.best_model_path)
				self._save_best_reward()
				if self.verbose:
					print(f"[BEST] episode={self.episode_count} reward={self.best_episode_reward:.3f} -> {self.best_model_path}")
			# TensorBoard 로깅
			self.logger.record("episode/reward", self.current_episode_reward)
			self.logger.record("episode/count", self.episode_count)
			self.logger.record("episode/best_reward", self.best_episode_reward)
			# 보상 항목별 로깅
			ep_info = infos[0].get("episode", {}) if infos else {}
			if ep_info:
				self.logger.record("episode/yaw_penalty", ep_info.get("yaw_penalty", 0.0))
				self.logger.record("episode/effort_penalty", ep_info.get("effort_penalty", 0.0))
				self.logger.record("episode/saturation_penalty", ep_info.get("saturation_penalty", 0.0))
			self.logger.dump(self.num_timesteps)
			if self.verbose:
				print(f"[EP {self.episode_count}] reward={self.current_episode_reward:.3f}")
			self.current_episode_reward = 0.0
		return True

def get_ppo_profile():
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
	for proc in psutil.process_iter(["pid", "name", "cmdline"]):
		try:
			name = proc.info.get("name") or ""
			cmdline = proc.info.get("cmdline") or []
			if target in name or any(target in arg for arg in cmdline):
				return proc.info["pid"]
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			continue
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


class CarMaker4WIDEnv(gym.Env):
	def __init__(self, loop):
		super().__init__()
		self.loop = loop
		self.client_sock = None
		self.last_obs = None
		self.last_full_state = None
		self.last_action_info = {}
		self.step_count = 0
		self.episode_reward_sum = 0.0
		self.episode_yaw_penalty_sum = 0.0
		self.episode_effort_penalty_sum = 0.0
		self.episode_saturation_penalty_sum = 0.0
		self.max_steps = int(os.getenv("EPISODE_MAX_STEPS", "1000"))
		self.early_done_penalty = float(os.getenv("EARLY_DONE_PENALTY", "-1000.0"))

		self.control_dt = max(CONTROL_DT, 1e-3)
		self.prev_yaw = None

		self.speed_controller = PISpeedController()
		self.stanley_controller = StanleySteeringController()

		self.base_dir = Path(__file__).resolve().parent
		self.use_open_loop_steer = OPEN_LOOP_STEER
		self.open_loop_amp_rad = float(np.deg2rad(OPEN_LOOP_STEER_AMP_DEG))
		self.open_loop_period_s = float(OPEN_LOOP_STEER_PERIOD_S)
		self.open_loop_start_s = float(OPEN_LOOP_STEER_START_S)
		self.open_loop_cycles = int(OPEN_LOOP_STEER_CYCLES)
		self.open_loop_speed_mps = float(OPEN_LOOP_SPEED_KPH) / 3.6
		self.open_loop_speed_mode = OPEN_LOOP_SPEED_MODE
		self.open_loop_speed_min_mps = float(OPEN_LOOP_SPEED_MIN_KPH) / 3.6
		self.open_loop_speed_max_mps = float(OPEN_LOOP_SPEED_MAX_KPH) / 3.6
		self.open_loop_accel_duration_s = max(0.0, float(OPEN_LOOP_ACCEL_DURATION_S))
		self.open_loop_cruise_high_duration_s = max(0.0, float(OPEN_LOOP_CRUISE_HIGH_DURATION_S))
		self.open_loop_decel_duration_s = max(0.0, float(OPEN_LOOP_DECEL_DURATION_S))
		self.open_loop_cruise_low_duration_s = max(0.0, float(OPEN_LOOP_CRUISE_LOW_DURATION_S))
		self.scenario_files = find_scenario_csv_files(self.base_dir)
		self.scenario_idx = -1
		self.current_scenario_path = None
		self.scenario = ScenarioProfile()
		self.scenario_global_step = 0

		self.curr_x = 0.0
		self.curr_y = 0.0
		self.curr_yaw = 0.0
		self.curr_speed_mps = 0.0
		self.last_total_torque = 0.0
		self.last_steering_cmd = 0.0
		self.last_yaw_rate_sq_error = 0.0

		self.reset_req = asyncio.Event()
		self.ready_evt = asyncio.Event()
		self.stop_req = asyncio.Event()

		# RL action: [left_total_bias, left_ratio, right_ratio]
		# left_total_bias: 좌우 총토크 편차 바이어스
		#   left_total = demand_trq/2 + bias*(WHEEL_TORQUE_LIMIT*2)
		#   right_total = demand_trq/2 - bias*(WHEEL_TORQUE_LIMIT*2)
		# left_fr_bias/right_fr_bias in [-1, 1]: -1=rear 100%, +1=front 100%, 0=50/50
		self.action_space = spaces.Box(
			low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
			high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
			dtype=np.float32,
		)
		# obs: [speed_kph, yaw_rate, ax, ay, total_torque, steering_wheel_angle, yaw_rate_sq_error]
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
		self.state_recv_bytes = 23 * 8
		self.action_send_fmt = "ddddd"

	def _advance_scenario(self):
		if self.use_open_loop_steer:
			self.current_scenario_path = None
			self.scenario = ScenarioProfile()
			self.scenario_global_step = 0
			print(
				"[SCENARIO] OPEN_LOOP_STEER=1. "
				f"Using sine steering (amp={OPEN_LOOP_STEER_AMP_DEG:.1f}deg, period={self.open_loop_period_s:.2f}s, "
				f"start={self.open_loop_start_s:.2f}s, cycles={self.open_loop_cycles}) "
				f"with OPEN_LOOP_SPEED_MODE={self.open_loop_speed_mode}."
			)
			return

		if not self.scenario_files:
			self.current_scenario_path = None
			self.scenario = ScenarioProfile()
			self.scenario_global_step = 0
			print("[SCENARIO] no CSV files found under scenarios. Steering=0, speed=fixed.")
			return

		self.scenario_idx = (self.scenario_idx + 1) % len(self.scenario_files)
		path = self.scenario_files[self.scenario_idx]
		if self.scenario.load_csv(path):
			self.current_scenario_path = path
			self.scenario.reset()
			self.scenario_global_step = 0
		else:
			self.current_scenario_path = None
			self.scenario = ScenarioProfile()
			self.scenario_global_step = 0
			print(f"[WARN] invalid scenario file: {path}. Fallback to fixed speed + zero steer.")

	def _scenario_finished(self) -> bool:
		if self.scenario.time_arr is None or self.scenario.time_arr.size == 0:
			return False
		sim_time = float(self.scenario_global_step) * self.control_dt
		return sim_time > float(self.scenario.time_arr[-1])

	def _target_speed_mps(self, ref_state):
		if self.use_open_loop_steer:
			return self._compute_open_loop_target_speed_mps(float(self.scenario_global_step) * self.control_dt)
		if ref_state is None:
			return 0.0
		return float(ref_state["speed_mps"])

	def _compute_open_loop_target_speed_mps(self, sim_time: float) -> float:
		mode = (self.open_loop_speed_mode or "constant").lower()
		if mode != "mixed":
			return self.open_loop_speed_mps

		v_low = float(min(self.open_loop_speed_min_mps, self.open_loop_speed_max_mps))
		v_high = float(max(self.open_loop_speed_min_mps, self.open_loop_speed_max_mps))

		t_acc = self.open_loop_accel_duration_s
		t_high = self.open_loop_cruise_high_duration_s
		t_dec = self.open_loop_decel_duration_s
		t_low = self.open_loop_cruise_low_duration_s
		cycle = t_acc + t_high + t_dec + t_low
		if cycle <= 1e-9:
			return self.open_loop_speed_mps

		phase = float(sim_time) - self.open_loop_start_s
		if phase < 0.0:
			return v_low
		phase = phase % cycle

		if phase < t_acc:
			alpha = 1.0 if t_acc <= 1e-9 else (phase / t_acc)
			return float(v_low + alpha * (v_high - v_low))
		phase -= t_acc

		if phase < t_high:
			return v_high
		phase -= t_high

		if phase < t_dec:
			alpha = 1.0 if t_dec <= 1e-9 else (phase / t_dec)
			return float(v_high + alpha * (v_low - v_high))

		return v_low

	def _compute_open_loop_steering_cmd(self, sim_time: float) -> float:
		if not self.use_open_loop_steer:
			return 0.0
		if self.open_loop_cycles <= 0:
			return 0.0

		if abs(self.open_loop_period_s) > 1e-9:
			omega = (2.0 * np.pi) / self.open_loop_period_s
		else:
			omega = 0.0

		phase_time = float(sim_time) - self.open_loop_start_s
		total_window = max(self.open_loop_period_s, 0.0) * float(self.open_loop_cycles)
		if 0.0 <= phase_time <= total_window:
			return float(self.open_loop_amp_rad * np.sin(omega * phase_time))
		return 0.0

	def _parse_assessment_state(self, raw_data: bytes):
		state = struct.unpack("d" * 23, raw_data)
		self.last_full_state = state
		self.curr_x = float(state[0])
		self.curr_y = float(state[1])
		yaw = float(state[2])
		speed_mps = float(state[3])

		wheel_rotv = [float(state[i]) for i in range(4, 8)]
		if self.prev_yaw is None:
			yaw_rate = 0.0
		else:
			yaw_rate = wrap_to_pi(yaw - self.prev_yaw) / self.control_dt
		self.prev_yaw = yaw

		self.curr_yaw = yaw
		self.curr_speed_mps = speed_mps

		ax = float(state[21])
		ay = float(state[22])
		obs = [speed_mps * 3.6, yaw_rate, ax, ay]
		return state, obs

	def _build_assessment_action(self, action):
		action = np.asarray(action, dtype=np.float32)
		left_total_bias = float(np.clip(action[0], -1.0, 1.0))
		left_fr_bias = float(np.clip(action[1], -1.0, 1.0))
		right_fr_bias = float(np.clip(action[2], -1.0, 1.0))

		sim_time = float(self.scenario_global_step) * self.control_dt
		ref_state = None
		if not self.use_open_loop_steer:
			ref_state = self.scenario.get_reference_state(sim_time)
			if ref_state is None and self.scenario.time_arr is not None and self.scenario.time_arr.size > 0:
				# Hold last scenario sample after end to avoid abrupt zero-target behavior.
				ref_state = self.scenario.get_reference_state(float(self.scenario.time_arr[-1]))

		target_speed_mps = self._target_speed_mps(ref_state)
		v_diff = target_speed_mps - self.curr_speed_mps
		total_torque = self.speed_controller.compute_total_torque(v_diff_mps=v_diff, dt=self.control_dt)
		self.last_total_torque = float(total_torque)

		steering_cmd = 0.0
		if self.use_open_loop_steer:
			steering_cmd = self._compute_open_loop_steering_cmd(sim_time)
		elif ref_state is not None:
			steering_cmd, _, _ = self.stanley_controller.compute(
				veh_x=self.curr_x,
				veh_y=self.curr_y,
				veh_yaw=self.curr_yaw,
				ref_x=ref_state["x"],
				ref_y=ref_state["y"],
				ref_yaw=ref_state["yaw"],
				vel=self.curr_speed_mps,
			)

		half_demand_trq = 0.5 * float(total_torque)
		lr_offset_max = WHEEL_TORQUE_LIMIT * 2.0
		lr_offset = left_total_bias * lr_offset_max
		print(half_demand_trq)
		left_total = half_demand_trq + lr_offset
		right_total = half_demand_trq - lr_offset

		left_half = 0.5 * left_total
		right_half = 0.5 * right_total
		fl = left_half + (left_fr_bias * left_half)
		rl = left_half - (left_fr_bias * left_half)
		fr = right_half + (right_fr_bias * right_half)
		rr = right_half - (right_fr_bias * right_half)

		# Wheel torque saturation
		wheel_torques = np.array([fl, fr, rl, rr], dtype=np.float32)
		wheel_torques_sat = np.clip(wheel_torques, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
		wheel_saturation_penalty = float(np.sum(np.abs(wheel_torques - wheel_torques_sat)))
		# 조향 기반 동역학 reference yaw rate
		ref_yaw_rate_base, ref_yaw_rate = self.stanley_controller.compute_reference_yaw_rate_both(
			veh_speed=self.curr_speed_mps, steering_cmd=steering_cmd
		)
		self.last_action_info = {
			"total_torque": float(total_torque),
			"torque_fl": float(wheel_torques_sat[0]),
			"torque_fr": float(wheel_torques_sat[1]),
			"torque_rl": float(wheel_torques_sat[2]),
			"torque_rr": float(wheel_torques_sat[3]),
			"steering_cmd": float(steering_cmd),
			"ref_yaw_rate_base": float(ref_yaw_rate_base),
			"ref_yaw_rate": float(ref_yaw_rate),
		}
		self.last_steering_cmd = float(steering_cmd)
		return (wheel_torques_sat[0], wheel_torques_sat[1], wheel_torques_sat[2], wheel_torques_sat[3], steering_cmd), wheel_torques_sat, v_diff, target_speed_mps, wheel_saturation_penalty, ref_yaw_rate

	async def reset(self, seed=None, options=None):
		print("\n[ENV] Resetting environment...")
		self.step_count = 0
		self.episode_reward_sum = 0.0
		self.episode_yaw_penalty_sum = 0.0
		self.episode_effort_penalty_sum = 0.0
		self.episode_saturation_penalty_sum = 0.0
		self.prev_yaw = None
		self.last_total_torque = 0.0
		self.last_steering_cmd = 0.0
		self.last_yaw_rate_sq_error = 0.0
		self.speed_controller.reset()
		if self.current_scenario_path is None or self._scenario_finished():
			self._advance_scenario()

		if self.client_sock is None:
			# 활성 연결 없음 -> 오케스트레이터에 새 시뮬레이션 요청
			print("[ENV] No active connection. Requesting new simulation...")
			self.reset_req.set()
			await self.ready_evt.wait()
			self.ready_evt.clear()
		else:
			# 시뮬레이션 계속 실행 중 -> 내부 상태만 리셋
			print("[ENV] Simulation still running. Resetting internal state only.")

		print("[ENV] Environment ready.")
		return np.array(self.last_obs, dtype=np.float32), {}

	async def step(self, action):
		try:
			self.step_count += 1
			self.scenario_global_step += 1

			action_cmd, wheel_torques, v_diff, target_speed_mps, wheel_saturation_penalty, ref_yaw_rate = self._build_assessment_action(action)

			data = struct.pack(self.action_send_fmt, *action_cmd)
			await self.loop.run_in_executor(None, self.client_sock.sendall, data)

			raw_data = await self.loop.run_in_executor(None, recv_exact, self.client_sock, self.state_recv_bytes)
			if not raw_data or len(raw_data) < self.state_recv_bytes:
				print("[ENV] Connection lost. Marking socket as dead.")
				self.client_sock = None
				info = {
					"yaw_penalty": float(self.episode_yaw_penalty_sum),
					"effort_penalty": float(self.episode_effort_penalty_sum),
					"saturation_penalty": float(self.episode_saturation_penalty_sum),
					"episode": {
						"r": float(self.episode_reward_sum),
						"l": int(self.step_count),
						"yaw_penalty": float(self.episode_yaw_penalty_sum),
						"effort_penalty": float(self.episode_effort_penalty_sum),
						"saturation_penalty": float(self.episode_saturation_penalty_sum),
					},
				}
				return np.zeros(self.observation_space.shape[0]), 0.0, True, False, info

			_, obs = self._parse_assessment_state(raw_data)
			obs.append(self.last_total_torque)
			obs.append(self.last_steering_cmd)
			obs.append(self.last_yaw_rate_sq_error)
			self.last_obs = obs
			sim_state = 8.0

			actual_yaw_rate = float(obs[1])
			yaw_rate_sq_error = float((actual_yaw_rate - ref_yaw_rate) ** 2)
			self.last_yaw_rate_sq_error = yaw_rate_sq_error
			obs[-1] = yaw_rate_sq_error
			print(actual_yaw_rate - ref_yaw_rate)
			yaw_rate_den = max(abs(float(ref_yaw_rate)), YAW_RATE_NORM_EPS)
			yaw_rate_norm_error = (actual_yaw_rate - ref_yaw_rate) / yaw_rate_den
			yaw_rate_err_penalty = YAW_RATE_WEIGHT * (yaw_rate_norm_error ** 2)
			effort_penalty = ACTION_EFFORT_WEIGHT * float(np.sum((wheel_torques / WHEEL_TORQUE_LIMIT) ** 2))
			saturation_penalty = SATURATION_PENALTY_WEIGHT * wheel_saturation_penalty

			reward = -yaw_rate_err_penalty - effort_penalty - saturation_penalty
			# print(yaw_rate_err_penalty)
			# print(effort_penalty)
			# print(saturation_penalty)
			terminated = sim_state != 8.0
			truncated = self.step_count >= self.max_steps
			done = terminated or truncated
			self.episode_reward_sum += float(reward)
			self.episode_yaw_penalty_sum += float(yaw_rate_err_penalty)
			self.episode_effort_penalty_sum += float(effort_penalty)
			self.episode_saturation_penalty_sum += float(saturation_penalty)

			if terminated and not truncated:
				print(f"[EARLY DONE] step={self.step_count} < max_steps={self.max_steps} -> penalty {self.early_done_penalty}")
				reward += self.early_done_penalty

			# print(
			# 	f"[ACTION] FL={action_cmd[0]:9.3f} | FR={action_cmd[1]:9.3f} "
			# 	f"| RL={action_cmd[2]:9.3f} | RR={action_cmd[3]:9.3f} | STR={action_cmd[4]:8.4f}"
			# )
			# print(
			# 	f"[TARGET] target_speed_mps={target_speed_mps:7.3f} "
			# 	f"| curr_speed_mps={self.curr_speed_mps:7.3f}"
			# )
			# print(
			# 	f"[REWARD] V_diff: {velocity_penalty:9.4f}"
			# 	f"| Effort: {effort_penalty:9.4f} | Total: {reward:9.4f}"
			# )

			info = {}
			if done:
				info["yaw_penalty"] = float(self.episode_yaw_penalty_sum)
				info["effort_penalty"] = float(self.episode_effort_penalty_sum)
				info["saturation_penalty"] = float(self.episode_saturation_penalty_sum)
				info["episode"] = {
					"r": float(self.episode_reward_sum),
					"l": int(self.step_count),
					"yaw_penalty": float(self.episode_yaw_penalty_sum),
					"effort_penalty": float(self.episode_effort_penalty_sum),
					"saturation_penalty": float(self.episode_saturation_penalty_sum),
				}

			return np.array(self.last_obs, dtype=np.float32), reward, terminated, truncated, info

		except Exception as e:
			print(f"[STEP ERROR] {type(e).__name__}: {e}")
			self.client_sock = None
			info = {
				"yaw_penalty": float(self.episode_yaw_penalty_sum),
				"effort_penalty": float(self.episode_effort_penalty_sum),
				"saturation_penalty": float(self.episode_saturation_penalty_sum),
				"episode": {
					"r": float(self.episode_reward_sum),
					"l": int(self.step_count),
					"yaw_penalty": float(self.episode_yaw_penalty_sum),
					"effort_penalty": float(self.episode_effort_penalty_sum),
					"saturation_penalty": float(self.episode_saturation_penalty_sum),
				},
			}
			return np.zeros(self.observation_space.shape[0]), 0.0, True, False, info


async def carmaker_orchestrator(env, simcontrol, variation, server_sock):
	loop = asyncio.get_running_loop()
	while True:
		await env.reset_req.wait()
		env.reset_req.clear()
        
		await simcontrol.start_and_connect()
        
		print("\n[Orchestrator] Starting New Episode...")

		simcontrol.set_variation(variation.clone())
		# await simcontrol.create_simstate_condition(cmapi.ConditionSimState.running).wait()
		# await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()
		print(simcontrol.get_status())
		await asyncio.sleep(1.0)  # 시뮬레이터가 안정적으로 시작될 때까지 잠시 대기
		await simcontrol.start_sim()
		simcontrol.set_realtimefactor(float(os.getenv("REALTIME_FACTOR", "100.0")))

		env.client_sock, _ = await loop.run_in_executor(None, server_sock.accept)
		raw_obs = await loop.run_in_executor(None, recv_exact, env.client_sock, env.state_recv_bytes)

		_, obs = env._parse_assessment_state(raw_obs)
		obs.append(env.last_total_torque)
		obs.append(env.last_steering_cmd)
		obs.append(env.last_yaw_rate_sq_error)
		env.last_obs = obs
		env.ready_evt.set()

		await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

		if env.client_sock:
			print("[Orchestrator] Closing client socket...")
			env.client_sock.close()
			env.client_sock = None
		print("[Orchestrator] Episode Finished & Cleaned up.")
		await simcontrol.stop_sim()

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


def run_learning(env):
	env = Monitor(env, info_keywords=("yaw_penalty", "effort_penalty", "saturation_penalty"))
	algo = os.getenv("ALGO", "ppo").strip().lower()
	if algo not in {"ppo", "sac", "td3"}:
		raise ValueError(f"ALGO는 'ppo', 'sac', 'td3' 이어야 합니다. 받은 값: {algo}")

	profile = get_ppo_profile()
	if algo == "ppo":
		algo_kwargs = PPO_PRESETS[profile]
		model_class = PPO
	elif algo == "sac":
		algo_kwargs = SAC_PRESETS[profile]
		model_class = SAC
	else:
		algo_kwargs = TD3_PRESETS[profile]
		model_class = TD3

	model_path = f"{MODEL_BASENAME}.zip"
	best_model_path = f"{MODEL_BASENAME}_best.zip"
	latest_model_path = f"{MODEL_BASENAME}.zip"
	interrupt_model_path = f"{MODEL_BASENAME}_interrupt.zip"
	tensorboard_log_dir = os.getenv("TENSORBOARD_LOG", f"{MODEL_BASENAME}_tensorboard")
	callback = SaveBestEpisodeRewardCallback(best_model_path, latest_model_path=latest_model_path)

	print(f"--- ALGO: {algo.upper()} | 프로필: {profile} ---")
	print(f"--- 설정: {algo_kwargs} ---")
	print(f"--- TensorBoard 로그 경로: {tensorboard_log_dir} ---")

	# LOAD_MODEL_PATH: 로드할 체크포인트 경로 (저장은 MODEL_BASENAME 경로 유지)
	load_path = os.getenv("LOAD_MODEL_PATH", "").strip()
	if load_path and os.path.exists(load_path):
		print(f"--- 체크포인트 로드: '{load_path}' -> 저장 경로: '{model_path}' ---")
		model = model_class.load(load_path, env=env, device="cpu", verbose=1, tensorboard_log=tensorboard_log_dir)
	elif os.path.exists(model_path):
		print(f"--- 기존 모델 '{model_path}'을(를) 불러와서 학습을 재개합니다. ---")
		model = model_class.load(model_path, env=env, device="cpu", verbose=1, tensorboard_log=tensorboard_log_dir)
	else:
		if load_path:
			print(f"[WARN] LOAD_MODEL_PATH='{load_path}' 파일 없음. 새 모델로 시작합니다.")
		print(f"--- 기존 모델이 없습니다. 새로운 {algo.upper()} 모델을 생성합니다. ---")
		model = model_class("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=tensorboard_log_dir, **algo_kwargs)

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

	total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "1000000"))
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
	proj_path = Path(os.getenv("CM_PROJECT_PATH", "/home/khkhh/CM_Projects/assessment_of_4wid_ver14"))
	cmapi.Project.load(proj_path)

	cm_port = int(os.getenv("CM_PORT", "5555"))
	server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_sock.bind(("127.0.0.1", cm_port))
	server_sock.listen(1)

	pid = get_carmaker_pid()
	if pid is None:
		raise RuntimeError("실행 중인 CarMaker GUI 프로세스를 찾지 못했습니다.")

	master = cmapi.ApoServer()
	master.set_sinfo(cmapi.ApoServerInfo(pid=pid, description="Idle"))
	master.set_host("localhost")

	simcontrol = cmapi.SimControlInteractive()
	await simcontrol.set_master(master)

	testrun_name = os.getenv("TESTRUN_NAME", "test1")
	testrun = cmapi.Project.instance().load_testrun_parametrization(proj_path / f"Data/TestRun/{testrun_name}")
	variation = cmapi.Variation.create_from_testrun(testrun)

	loop = asyncio.get_running_loop()
	env_async = CarMaker4WIDEnv(loop)
	print(
		f"[CONFIG] CONTROL_DT={env_async.control_dt:.3f}s | CM_PORT={cm_port} | TESTRUN={testrun_name} "
	)
	if OPEN_LOOP_STEER:
		print(
			f"[CONFIG] OPEN_LOOP_STEER=1 | AMP_DEG={OPEN_LOOP_STEER_AMP_DEG:.1f} | "
			f"PERIOD_S={OPEN_LOOP_STEER_PERIOD_S:.2f} | START_S={OPEN_LOOP_STEER_START_S:.2f} | "
			f"CYCLES={OPEN_LOOP_STEER_CYCLES} | SPEED_MODE={OPEN_LOOP_SPEED_MODE} | "
			f"SPEED_KPH={OPEN_LOOP_SPEED_KPH:.1f} | SPEED_MIN_KPH={OPEN_LOOP_SPEED_MIN_KPH:.1f} | "
			f"SPEED_MAX_KPH={OPEN_LOOP_SPEED_MAX_KPH:.1f}"
		)
	print(
		f"[CONFIG] SCENARIO_DIR={SCENARIO_DIR} | SCENARIO_CSV_PATH={SCENARIO_CSV_PATH or '(auto from scenarios)'} "
		f"| CarMakerPID={pid}"
	)

	cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))

	env_sync = SyncBridge(env_async, loop)
	await loop.run_in_executor(None, run_learning, env_sync)

	print("Learning finished. Sending stop signal to orchestrator...")
	env_async.stop_req.set()
	loop.stop()


if __name__ == "__main__":
	cmapi.Task.run_main_task(main())

"""
CarMaker GUI + PPO/SAC/TD3 학습 스크립트 (단일 환경).

사용법:
    python3 train_with_cm_gui.py [A|B|C]   # 알고리즘 프리셋 선택

주요 환경변수:
    ALGO                : ppo | sac | td3  (기본: ppo)
    PPO_PROFILE         : A | B | C        (기본: B)
    CM_PROJECT_PATH     : CarMaker 프로젝트 경로
    TESTRUN_NAME        : TestRun 이름
    CM_PORT             : 소켓 포트         (기본: 5555)
    REALTIME_FACTOR     : 실시간 배율       (기본: 100.0)
    TOTAL_TIMESTEPS     : 총 학습 스텝 수   (기본: 1_000_000)
    LOAD_MODEL_PATH     : 이어서 학습할 모델 경로
"""

import os
import sys
import asyncio
import socket
import struct
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

# CarMaker API
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi  # noqa: E402

from carmaker_utils import (  # noqa: E402
    wrap_to_pi,
    recv_exact,
    get_carmaker_pid,
    PISpeedController,
    StanleySteeringController,
    ScenarioProfile,
    SaveBestEpisodeRewardCallback,
    SyncBridge,
    carmaker_orchestrator,
    MotorMap,
    compute_batt_net_power,
    compute_batt_net_power_batch,
)

# ─────────────────────────────────────────────────────────────────────────────
# [1] 보상 함수 파라미터  ◀  자주 수정
# ─────────────────────────────────────────────────────────────────────────────
YAW_RATE_WEIGHT           = float(os.getenv("YAW_RATE_WEIGHT",          "10"))
ACTION_EFFORT_WEIGHT      = float(os.getenv("ACTION_EFFORT_WEIGHT",     "0"))
SATURATION_PENALTY_WEIGHT = float(os.getenv("SATURATION_PENALTY_WEIGHT", "0"))
YAW_RATE_NORM_EPS         = float(os.getenv("YAW_RATE_NORM_EPS",        "0.01"))

# ─────────────────────────────────────────────────────────────────────────────
# [2] 시나리오 / 에피소드 설정  ◀  자주 수정
# ─────────────────────────────────────────────────────────────────────────────
# CSV 경로 설정: SCENARIO_CSV_PATH 지정 시 단일 파일, 아니면 SCENARIO_DIR 디렉토리 탐색
SCENARIO_DIR      = os.getenv("SCENARIO_DIR",      "scenarios").strip()
SCENARIO_CSV_PATH = os.getenv("SCENARIO_CSV_PATH", "").strip()

# Open-loop 조향 모드 (시나리오 CSV 대신 정현파 조향 + 속도 프로파일로 학습)
OPEN_LOOP_STEER             = os.getenv("OPEN_LOOP_STEER", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
OPEN_LOOP_STEER_AMP_DEG     = float(os.getenv("OPEN_LOOP_STEER_AMP_DEG",   "60.0"))
OPEN_LOOP_STEER_PERIOD_S    = float(os.getenv("OPEN_LOOP_STEER_PERIOD_S",  "4.0"))
OPEN_LOOP_STEER_START_S     = float(os.getenv("OPEN_LOOP_STEER_START_S",   "0.0"))
OPEN_LOOP_STEER_CYCLES      = max(0, int(float(os.getenv("OPEN_LOOP_STEER_CYCLES", "700.0"))))
OPEN_LOOP_SPEED_KPH         = float(os.getenv("OPEN_LOOP_SPEED_KPH",       "80.0"))
OPEN_LOOP_SPEED_MODE        = os.getenv("OPEN_LOOP_SPEED_MODE", "mixed").strip().lower()  # constant | mixed
OPEN_LOOP_SPEED_MIN_KPH     = float(os.getenv("OPEN_LOOP_SPEED_MIN_KPH",   "0.0"))
OPEN_LOOP_SPEED_MAX_KPH     = float(os.getenv("OPEN_LOOP_SPEED_MAX_KPH",   "60.0"))
OPEN_LOOP_ACCEL_DURATION_S  = float(os.getenv("OPEN_LOOP_ACCEL_DURATION_S",       "10.0"))
OPEN_LOOP_CRUISE_HIGH_DURATION_S = float(os.getenv("OPEN_LOOP_CRUISE_HIGH_DURATION_S", "40.0"))
OPEN_LOOP_DECEL_DURATION_S  = float(os.getenv("OPEN_LOOP_DECEL_DURATION_S",       "5.0"))
OPEN_LOOP_CRUISE_LOW_DURATION_S  = float(os.getenv("OPEN_LOOP_CRUISE_LOW_DURATION_S",  "5.0"))

# ─────────────────────────────────────────────────────────────────────────────
# [3] 알고리즘 프리셋  ◀  자주 수정
# ─────────────────────────────────────────────────────────────────────────────
# 스크립트 위치 기준 절대 경로로 고정 (CarMaker가 cwd를 바꿔도 올바른 위치에 저장됨)
_SCRIPT_DIR = Path(__file__).resolve().parent
# MODEL_BASENAME = os.getenv("MODEL_BASENAME", "carmaker_ppo_4wid_dyc").strip()
_MODEL_BASENAME = os.getenv("MODEL_BASENAME", "carmaker_sac_4wid_dyc_C_cl").strip()

# ── 커리큘럼 학습 단계 (Reward Curriculum 방식) ────────────────────────────
# CURRICULUM_STAGE=0 : 커리큘럼 없음 (ENERGY_LOSS_WEIGHT 기본 0.1)
# CURRICULUM_STAGE=1 : Stage 1 – DYC 전용  (ENERGY_LOSS_WEIGHT 기본 0.03 → 직선 action=0 인센티브)
# CURRICULUM_STAGE=2 : Stage 2 – 에너지 추가 (ENERGY_LOSS_WEIGHT 기본 2.0 → fr_bias 학습 유도)
CURRICULUM_STAGE = int(os.getenv("CURRICULUM_STAGE", "0"))
_MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "").strip()
if _MODEL_OUTPUT_DIR:
    _base = Path(_MODEL_OUTPUT_DIR)
    if not _base.is_absolute():
        _base = _SCRIPT_DIR / _base
    _base.mkdir(parents=True, exist_ok=True)
    MODEL_BASENAME = str(_base / _MODEL_BASENAME)
else:
    MODEL_BASENAME = str(_SCRIPT_DIR / _MODEL_BASENAME)

# ENERGY_LOSS_WEIGHT: Stage 1=0.03 (직선 action=0 인센티브), Stage 2=2.0 (에너지 학습), 0=0.1 (기존)
# 환경변수로 덮어쓰기 가능
_ENERGY_DEFAULT = "0.03" if CURRICULUM_STAGE == 1 else ("2.0" if CURRICULUM_STAGE == 2 else "0.1")
ENERGY_LOSS_WEIGHT = float(os.getenv("ENERGY_LOSS_WEIGHT", _ENERGY_DEFAULT))

PPO_PRESETS = {
    "A": {"learning_rate": 1e-4,  "batch_size": 256, "gamma": 0.99,  "n_steps": 1024, "ent_coef": 0.1},
    "B": {"learning_rate": 3e-4,  "batch_size": 256, "gamma": 0.99,  "n_steps": 4096, "ent_coef": 0.0},
    "C": {"learning_rate": 1e-3,  "batch_size": 256, "gamma": 0.99,  "n_steps": 8192, "ent_coef": 0.0},
}

SAC_PRESETS = {
    "A": {
        "learning_rate": 3e-4, "batch_size": 256,  "gamma": 0.99,  "ent_coef": "auto",
        "tau": 0.005, "buffer_size": 300000, "learning_starts": 5000,
        "train_freq": (1, "step"), "gradient_steps": 1,
    },
    "B": {
        "learning_rate": 1e-4, "batch_size": 512,  "gamma": 0.995, "ent_coef": "auto",
        "tau": 0.005, "buffer_size": 500000, "learning_starts": 10000,
        "train_freq": (1, "step"), "gradient_steps": 1,
    },
    "C": {
        "learning_rate": 3e-4, "batch_size": 256,  "gamma": 0.99,   "ent_coef": "auto",
        "tau": 0.005, "buffer_size": 1500000, "learning_starts": 5000,
        "train_freq": (1, "step"), "gradient_steps": 1, "policy_kwargs": {"net_arch": [256, 256]},
    },
}

TD3_PRESETS = {
    "A": {
        "learning_rate": 3e-4, "batch_size": 256,  "gamma": 0.99,  "tau": 0.005,
        "buffer_size": 300000, "learning_starts": 5000,
        "train_freq": (1, "step"), "gradient_steps": 1,
        "policy_delay": 2, "target_policy_noise": 0.2, "target_noise_clip": 0.5,
    },
    "B": {
        "learning_rate": 1e-4, "batch_size": 512,  "gamma": 0.995, "tau": 0.005,
        "buffer_size": 500000, "learning_starts": 10000,
        "train_freq": (1, "step"), "gradient_steps": 1,
        "policy_delay": 2, "target_policy_noise": 0.2, "target_noise_clip": 0.5,
    },
    "C": {
        "learning_rate": 3e-4, "batch_size": 256,  "gamma": 0.99,  "tau": 0.01,
        "buffer_size": 300000, "learning_starts": 5000,
        "train_freq": (1, "step"), "gradient_steps": 2,
        "policy_delay": 2, "target_policy_noise": 0.1, "target_noise_clip": 0.5,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# [4] 차량·제어 파라미터  ◀  고급 설정 (보통 수정 불필요)
# ─────────────────────────────────────────────────────────────────────────────
WHEEL_TORQUE_LIMIT = float(os.getenv("WHEEL_TORQUE_LIMIT", "630.0"))
TOTAL_TORQUE_LIMIT = float(os.getenv("TOTAL_TORQUE_LIMIT", str(WHEEL_TORQUE_LIMIT * 4.0)))
CONTROL_DT         = float(os.getenv("CONTROL_DT",         "0.05"))

# 속도 PI 제어기: CTRL_KP / CTRL_KI / CTRL_KD / INTEGRAL_LIMIT (env var, carmaker_utils에서 읽음)
# Stanley 조향 제어기: STANLEY_G1 / G2 / K / SCALE (env var, carmaker_utils에서 읽음)
# 차량 동역학: VEH_MASS_KG / VEH_A_M / VEH_B_M / VEH_CF / VEH_CR / VEH_STEER_RATIO / VEH_MU (env var)

# observation 정규화 스케일 (각 obs를 이 값으로 나누면 대략 [-1, 1] 범위)
OBS_SCALE = np.array([
    float(os.getenv("OBS_SCALE_SPEED",      "100.0")),   # speed_kph         (0 ~ 200)
    float(os.getenv("OBS_SCALE_YAW_RATE",     "0.4")),   # yaw_rate  rad/s   (-2 ~ 2)
    float(os.getenv("OBS_SCALE_AX",          "5.0")),   # ax        m/s²    (-15 ~ 15)
    float(os.getenv("OBS_SCALE_AY",          "5.0")),   # ay        m/s²    (-15 ~ 15)
    float(os.getenv("OBS_SCALE_TORQUE",    "2520.0")),   # total_torque Nm   (-2520 ~ 2520)
    float(os.getenv("OBS_SCALE_STEER",       "9.425")),    # steering_cmd rad  (-0.7 ~ 0.7)
    float(os.getenv("OBS_SCALE_YAW_SQ_ERR",  "0.16")),   # yaw_rate_sq_error (0 ~ 16→sqrt→4)
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 시나리오 CSV 파일 목록 검색 (SCENARIO_DIR / SCENARIO_CSV_PATH 참조)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# CarMaker 4WID 환경
# ─────────────────────────────────────────────────────────────────────────────

class CarMaker4WIDEnv(gym.Env):
    """
    4WID DYC (Direct Yaw-rate Control) RL 환경.

    액션: [left_total_bias, left_fr_bias, right_fr_bias]  각 [-1, 1]
      - left_total_bias  : 좌/우 총토크 편차 바이어스
      - left/right_fr_bias: 전/후축 토크 배분 바이어스 (-1=후100%, +1=전100%)

    관측: [speed_kph, yaw_rate, ax, ay, total_torque, steering_cmd, yaw_rate_sq_error]
    """

    def __init__(self, loop):
        super().__init__()
        self.loop = loop

        # 소켓·상태
        self.client_sock        = None
        self.last_obs           = None
        self.last_full_state    = None
        self.last_action_info   = {}

        # 에피소드 카운터
        self.step_count                  = 0
        self.episode_reward_sum          = 0.0
        self.episode_yaw_penalty_sum     = 0.0
        self.episode_effort_penalty_sum  = 0.0
        self.episode_saturation_penalty_sum = 0.0

        # 에피소드 설정
        # EPISODE_MAX_STEPS 미지정 시 시나리오 CSV의 마지막 time 기준 자동 계산됨
        # 수동 지정: WLTP=36000, HWFET=15300, FTP75=49480
        self._episode_max_steps_override = int(os.getenv("EPISODE_MAX_STEPS", "0"))
        self.max_steps        = self._episode_max_steps_override if self._episode_max_steps_override > 0 else 36000
        self.early_done_penalty = float(os.getenv("EARLY_DONE_PENALTY", "-50.0"))
        self.control_dt       = max(CONTROL_DT, 1e-3)

        # 내부 상태
        self.prev_yaw              = None
        self.curr_x                = 0.0
        self.curr_y                = 0.0
        self.curr_yaw              = 0.0
        self.curr_speed_mps        = 0.0
        self.last_total_torque     = 0.0
        self.last_steering_cmd     = 0.0
        self.last_yaw_rate_sq_error = 0.0

        # 제어기
        self.speed_controller   = PISpeedController()
        self.stanley_controller = StanleySteeringController()
        mg_map_path = os.getenv("MG_MAP_PATH", str(Path(__file__).resolve().parent / "scaled_mg_map.mat"))
        self.motor_map = MotorMap(Path(mg_map_path))
        self.motor_map.load()
        self.last_rotv = np.zeros(4, dtype=np.float64)

        # 시나리오
        self.base_dir          = Path(__file__).resolve().parent
        self.use_open_loop_steer = OPEN_LOOP_STEER
        self.open_loop_amp_rad  = float(np.deg2rad(OPEN_LOOP_STEER_AMP_DEG))
        self.open_loop_period_s = float(OPEN_LOOP_STEER_PERIOD_S)
        self.open_loop_start_s  = float(OPEN_LOOP_STEER_START_S)
        self.open_loop_cycles   = int(OPEN_LOOP_STEER_CYCLES)
        self.open_loop_speed_mps      = float(OPEN_LOOP_SPEED_KPH) / 3.6
        self.open_loop_speed_mode     = OPEN_LOOP_SPEED_MODE
        self.open_loop_speed_min_mps  = float(OPEN_LOOP_SPEED_MIN_KPH) / 3.6
        self.open_loop_speed_max_mps  = float(OPEN_LOOP_SPEED_MAX_KPH) / 3.6
        self.open_loop_accel_duration_s       = max(0.0, float(OPEN_LOOP_ACCEL_DURATION_S))
        self.open_loop_cruise_high_duration_s = max(0.0, float(OPEN_LOOP_CRUISE_HIGH_DURATION_S))
        self.open_loop_decel_duration_s       = max(0.0, float(OPEN_LOOP_DECEL_DURATION_S))
        self.open_loop_cruise_low_duration_s  = max(0.0, float(OPEN_LOOP_CRUISE_LOW_DURATION_S))
        self.scenario_files        = find_scenario_csv_files(self.base_dir)
        self.scenario_idx          = -1
        self.current_scenario_path = None
        self.scenario              = ScenarioProfile()
        self.scenario_global_step  = 0

        # 동기화 이벤트
        self.reset_req   = asyncio.Event()
        self.ready_evt   = asyncio.Event()
        self.stop_req    = asyncio.Event()
        # truncation 후 "Simulation still running" tail 구간 표시:
        # True이면 연결 끊김을 조기 중단이 아닌 자연 종료로 처리.
        self._in_sim_tail = False

        # 액션·관측 공간
        # action_space는 항상 3D [-1,1] 유지 (Stage 1에서도 동일)
        # Stage 1에서 fr_bias(action[1,2])는 _build_assessment_action에서 0으로 강제됨
        # (high=low=0 으로 설정하면 SAC Gaussian scale=0 → NaN 발생)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.state_recv_bytes  = 23 * 8
        self.action_send_fmt   = "ddddd"

    # ── 보상 함수 ──────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        actual_yaw_rate: float,
        ref_yaw_rate: float,
        wheel_torques: np.ndarray,
        wheel_saturation_penalty: float,
        total_torque: float = 0.0,
    ):
        """
        ════════════════════════════════════════════════
        보상 함수  ◀  수정 포인트
        ════════════════════════════════════════════════
        반환: (reward, yaw_penalty, effort_penalty, saturation_penalty, energy_penalty)
        energy_penalty: 양수 = RL이 균등분배보다 에너지 더 씀 (reward 감소), 음수 = 절감 (reward 증가)
        """
        # ref 기준 정규화 + clamp:
        #   직진(ref≈0) → denom=EPS → 작은 모멘트도 엄격하게 패널티
        #   코너링(ref 큼) → ref가 분모 → 상대 오차
        #   물리적 한계 구간(norm_err 폭발) → clamp=2.0으로 상한 제한
        yaw_den        = max(abs(ref_yaw_rate), YAW_RATE_NORM_EPS)
        yaw_norm_err   = min(abs((actual_yaw_rate - ref_yaw_rate) / yaw_den), 2.0)
        yaw_penalty    = YAW_RATE_WEIGHT * yaw_norm_err ** 2
        effort_penalty = ACTION_EFFORT_WEIGHT * float(np.sum((wheel_torques / WHEEL_TORQUE_LIMIT) ** 2))
        sat_penalty    = SATURATION_PENALTY_WEIGHT * wheel_saturation_penalty
        # 에너지: 등분배(algo0) 대비 상대 절감량으로 정규화
        ref_torques    = np.full(4, total_torque / 4.0, dtype=np.float64)
        batt_rl        = compute_batt_net_power(wheel_torques, self.last_rotv, self.motor_map)
        batt_ref       = compute_batt_net_power(ref_torques,   self.last_rotv, self.motor_map)
        energy_saving  = (batt_ref - batt_rl) / max(abs(batt_ref), 1.0)  # 양수=절감, 음수=낭비
        energy_penalty = -ENERGY_LOSS_WEIGHT * energy_saving              # 보상에 더하기 위해 부호 반전
        reward = -(yaw_penalty + effort_penalty + sat_penalty + energy_penalty)
        return reward, yaw_penalty, effort_penalty, sat_penalty, energy_penalty

    # ── 시나리오 관리 ──────────────────────────────────────────────────────────

    def _advance_scenario(self):
        if self.use_open_loop_steer:
            self.current_scenario_path = None
            self.scenario = ScenarioProfile()
            self.scenario_global_step  = 0
            print(
                "[SCENARIO] OPEN_LOOP_STEER=1. "
                f"Sine steering (amp={OPEN_LOOP_STEER_AMP_DEG:.1f}deg, "
                f"period={self.open_loop_period_s:.2f}s, start={self.open_loop_start_s:.2f}s, "
                f"cycles={self.open_loop_cycles}) speed_mode={self.open_loop_speed_mode}."
            )
            return
        if not self.scenario_files:
            self.current_scenario_path = None
            self.scenario = ScenarioProfile()
            self.scenario_global_step  = 0
            print("[SCENARIO] no CSV files found. Steering=0, speed=fixed.")
            return
        self.scenario_idx = (self.scenario_idx + 1) % len(self.scenario_files)
        path = self.scenario_files[self.scenario_idx]
        if self.scenario.load_csv(path):
            self.current_scenario_path = path
            self.scenario.reset()
            self.scenario_global_step  = 0
            # 시나리오 지속 시간으로 max_steps 자동 설정 (EPISODE_MAX_STEPS 미지정 시)
            if self._episode_max_steps_override <= 0 and \
               self.scenario.time_arr is not None and self.scenario.time_arr.size > 0:
                self.max_steps = int(np.ceil(float(self.scenario.time_arr[-1]) / self.control_dt))
                print(f"[SCENARIO] max_steps={self.max_steps} "
                      f"({self.scenario.time_arr[-1]:.1f}s / {self.control_dt}s)")
        else:
            self.current_scenario_path = None
            self.scenario = ScenarioProfile()
            self.scenario_global_step  = 0
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
        v_low  = float(min(self.open_loop_speed_min_mps, self.open_loop_speed_max_mps))
        v_high = float(max(self.open_loop_speed_min_mps, self.open_loop_speed_max_mps))
        t_acc  = self.open_loop_accel_duration_s
        t_high = self.open_loop_cruise_high_duration_s
        t_dec  = self.open_loop_decel_duration_s
        t_low  = self.open_loop_cruise_low_duration_s
        cycle  = t_acc + t_high + t_dec + t_low
        if cycle <= 1e-9:
            return self.open_loop_speed_mps
        phase = float(sim_time) - self.open_loop_start_s
        if phase < 0.0:
            return v_low
        phase = phase % cycle
        if phase < t_acc:
            return float(v_low + (1.0 if t_acc <= 1e-9 else phase / t_acc) * (v_high - v_low))
        phase -= t_acc
        if phase < t_high:
            return v_high
        phase -= t_high
        if phase < t_dec:
            return float(v_high + (1.0 if t_dec <= 1e-9 else phase / t_dec) * (v_low - v_high))
        return v_low

    def _compute_open_loop_steering_cmd(self, sim_time: float) -> float:
        if not self.use_open_loop_steer or self.open_loop_cycles <= 0:
            return 0.0
        omega = (2.0 * np.pi) / self.open_loop_period_s if abs(self.open_loop_period_s) > 1e-9 else 0.0
        phase_time   = float(sim_time) - self.open_loop_start_s
        total_window = max(self.open_loop_period_s, 0.0) * float(self.open_loop_cycles)
        if 0.0 <= phase_time <= total_window:
            return float(self.open_loop_amp_rad * np.sin(omega * phase_time))
        return 0.0

    # ── 상태 파싱 / 액션 빌드 ──────────────────────────────────────────────────

    def _parse_assessment_state(self, raw_data: bytes):
        state = struct.unpack("d" * 23, raw_data)
        self.last_full_state = state
        self.curr_x     = float(state[0])
        self.curr_y     = float(state[1])
        yaw             = float(state[2])
        speed_mps       = float(state[3])
        if self.prev_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_rate = wrap_to_pi(yaw - self.prev_yaw) / self.control_dt
        self.prev_yaw       = yaw
        self.curr_yaw       = yaw
        self.curr_speed_mps = speed_mps
        ax = float(state[21])
        ay = float(state[22])
        self.last_rotv = np.array(state[4:8], dtype=np.float64)
        obs = [speed_mps * 3.6, yaw_rate, ax, ay]
        return state, obs

    def _build_assessment_action(self, action):
        action         = np.asarray(action, dtype=np.float32)
        left_total_bias = float(np.clip(action[0], -1.0, 1.0))
        left_fr_bias    = float(np.clip(action[1], -1.0, 1.0))
        right_fr_bias   = float(np.clip(action[2], -1.0, 1.0))

        sim_time  = float(self.scenario_global_step) * self.control_dt
        ref_state = None
        if not self.use_open_loop_steer:
            ref_state = self.scenario.get_reference_state(sim_time)
            if ref_state is None and self.scenario.time_arr is not None and self.scenario.time_arr.size > 0:
                ref_state = self.scenario.get_reference_state(float(self.scenario.time_arr[-1]))

        target_speed_mps = self._target_speed_mps(ref_state)
        v_diff       = target_speed_mps - self.curr_speed_mps
        total_torque = self.speed_controller.compute_total_torque(v_diff_mps=v_diff, dt=self.control_dt)
        self.last_total_torque = float(total_torque)

        steering_cmd = 0.0
        if self.use_open_loop_steer:
            steering_cmd = self._compute_open_loop_steering_cmd(sim_time)
        elif ref_state is not None:
            steering_cmd, _, _ = self.stanley_controller.compute(
                veh_x=self.curr_x, veh_y=self.curr_y, veh_yaw=self.curr_yaw,
                ref_x=ref_state["x"], ref_y=ref_state["y"], ref_yaw=ref_state["yaw"],
                vel=self.curr_speed_mps,
            )

        half    = 0.5 * float(total_torque)
        lr_off  = left_total_bias * WHEEL_TORQUE_LIMIT * 2.0
        lh      = 0.5 * (half + lr_off)
        rh      = 0.5 * (half - lr_off)
        fl      = lh + left_fr_bias   * lh
        rl      = lh - left_fr_bias   * lh
        fr      = rh + right_fr_bias  * rh
        rr      = rh - right_fr_bias  * rh

        wheel_torques     = np.array([fl, fr, rl, rr], dtype=np.float32)
        wheel_torques_sat = np.clip(wheel_torques, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
        wheel_saturation_penalty = float(np.sum(np.abs(wheel_torques - wheel_torques_sat)))

        ref_yaw_rate_base, ref_yaw_rate = self.stanley_controller.compute_reference_yaw_rate_both(
            veh_speed=self.curr_speed_mps, steering_cmd=steering_cmd
        )
        self.last_action_info = {
            "total_torque":     float(total_torque),
            "torque_fl":        float(wheel_torques_sat[0]),
            "torque_fr":        float(wheel_torques_sat[1]),
            "torque_rl":        float(wheel_torques_sat[2]),
            "torque_rr":        float(wheel_torques_sat[3]),
            "steering_cmd":     float(steering_cmd),
            "ref_yaw_rate_base": float(ref_yaw_rate_base),
            "ref_yaw_rate":     float(ref_yaw_rate),
        }
        self.last_steering_cmd = float(steering_cmd)
        return (
            (wheel_torques_sat[0], wheel_torques_sat[1],
             wheel_torques_sat[2], wheel_torques_sat[3], steering_cmd),
            wheel_torques_sat, v_diff, target_speed_mps, wheel_saturation_penalty, ref_yaw_rate,
        )

    # ── episode info 헬퍼 ──────────────────────────────────────────────────────

    def _make_episode_info(self) -> dict:
        return {
            "yaw_penalty":        float(self.episode_yaw_penalty_sum),
            "effort_penalty":     float(self.episode_effort_penalty_sum),
            "saturation_penalty": float(self.episode_saturation_penalty_sum),
            "energy_penalty":     float(self.episode_energy_penalty_sum),
            "episode": {
                "r":                  float(self.episode_reward_sum),
                "l":                  int(self.step_count),
                "yaw_penalty":        float(self.episode_yaw_penalty_sum),
                "effort_penalty":     float(self.episode_effort_penalty_sum),
                "saturation_penalty": float(self.episode_saturation_penalty_sum),
                "energy_penalty":     float(self.episode_energy_penalty_sum),
            },
        }

    # ── reset / step ───────────────────────────────────────────────────────────

    async def reset(self, seed=None, options=None):
        print("\n[ENV] Resetting environment...")
        self.step_count                   = 0
        self.episode_reward_sum           = 0.0
        self.episode_yaw_penalty_sum      = 0.0
        self.episode_effort_penalty_sum   = 0.0
        self.episode_saturation_penalty_sum = 0.0
        self.episode_energy_penalty_sum   = 0.0
        self.prev_yaw                     = None
        self.last_total_torque            = 0.0
        self.last_steering_cmd            = 0.0
        self.last_yaw_rate_sq_error       = 0.0
        self.speed_controller.reset()

        if self.client_sock is None:
            # 활성 연결 없음 → 오케스트레이터에 새 시뮬레이션 요청
            self._in_sim_tail = False
            print("[ENV] No active connection. Requesting new simulation...")
            self._advance_scenario()
            self.reset_req.set()
            await self.ready_evt.wait()
            self.ready_evt.clear()
        else:
            # 시뮬레이션 계속 실행 중 → 내부 상태만 리셋
            # (이전 에피소드 max_steps truncation 후 CM이 아직 돌고 있는 tail 구간)
            self._in_sim_tail = True
            print("[ENV] Simulation still running (tail). Resetting internal state only.")

        print("[ENV] Environment ready.")
        return np.array(self.last_obs, dtype=np.float32), {}

    async def step(self, action):
        try:
            self.step_count           += 1
            self.scenario_global_step += 1

            action_cmd, wheel_torques, _, _, wheel_saturation_penalty, ref_yaw_rate = \
                self._build_assessment_action(action)

            data = struct.pack(self.action_send_fmt, *action_cmd)
            await self.loop.run_in_executor(None, self.client_sock.sendall, data)

            raw_data = await self.loop.run_in_executor(
                None, recv_exact, self.client_sock, self.state_recv_bytes
            )

            if not raw_data or len(raw_data) < self.state_recv_bytes:
                # 소켓 종료 → 조기 중단 vs 자연 종료 판정
                self.client_sock = None
                if self.step_count >= self.max_steps or self._in_sim_tail:
                    was_tail = self._in_sim_tail
                    self._in_sim_tail = False
                    conn_reward = 0.0
                    terminated, truncated = False, True
                    print(f"[ENV] 시뮬 자연 종료 (step={self.step_count}/{self.max_steps}, tail={was_tail})")
                else:
                    conn_reward = self.early_done_penalty
                    terminated, truncated = True, False
                    print(f"[ENV] 조기 중단 패널티: {self.early_done_penalty:.1f} (step={self.step_count}/{self.max_steps})")
                self.episode_reward_sum += conn_reward
                return np.zeros(self.observation_space.shape[0]), conn_reward, terminated, truncated, self._make_episode_info()

            # 정상 스텝
            _, obs = self._parse_assessment_state(raw_data)
            obs.append(self.last_total_torque)
            obs.append(self.last_steering_cmd)
            actual_yaw_rate     = float(obs[1])
            yaw_rate_sq_error   = float((actual_yaw_rate - ref_yaw_rate) ** 2)
            self.last_yaw_rate_sq_error = yaw_rate_sq_error
            obs.append(yaw_rate_sq_error)
            obs = (np.array(obs, dtype=np.float32) / OBS_SCALE).tolist()
            self.last_obs = obs

            reward, yaw_pen, effort_pen, sat_pen, energy_pen = self._compute_reward(
                actual_yaw_rate, ref_yaw_rate, wheel_torques, wheel_saturation_penalty,
                float(self.last_total_torque),
            )

            terminated = False  # 시뮬 상태는 소켓 끊김으로만 감지
            truncated  = self.step_count >= self.max_steps
            done       = terminated or truncated

            self.episode_reward_sum             += float(reward)
            self.episode_yaw_penalty_sum        += float(yaw_pen)
            self.episode_effort_penalty_sum     += float(effort_pen)
            self.episode_saturation_penalty_sum += float(sat_pen)
            self.episode_energy_penalty_sum     += float(energy_pen)

            info = self._make_episode_info() if done else {}
            return np.array(self.last_obs, dtype=np.float32), reward, terminated, truncated, info

        except Exception as e:
            print(f"[STEP ERROR] {type(e).__name__}: {e} (step={self.step_count}/{self.max_steps})")
            self.client_sock = None

            if self.step_count >= self.max_steps or self._in_sim_tail:
                # 정상 truncation 또는 tail 구간 자연 종료
                was_tail = self._in_sim_tail
                self._in_sim_tail = False
                reward = 0.0
                terminated, truncated = False, True
                print(f"[STEP] 자연 종료 (exception, tail={was_tail}). Normal truncation.")
            elif self.last_obs is not None and self.last_action_info:
                # 조기 중단: 마지막 관측 기반 보상 + 조기 중단 패널티
                actual_yaw_rate = float(self.last_obs[1])
                ref_yaw_rate_ex = float(self.last_action_info.get("ref_yaw_rate", 0.0))
                torques_ex = np.array([
                    self.last_action_info.get("torque_fl", 0.0),
                    self.last_action_info.get("torque_fr", 0.0),
                    self.last_action_info.get("torque_rl", 0.0),
                    self.last_action_info.get("torque_rr", 0.0),
                ], dtype=np.float32)
                step_r, _, _, _, _ = self._compute_reward(actual_yaw_rate, ref_yaw_rate_ex, torques_ex, 0.0)
                reward = step_r + self.early_done_penalty
                terminated, truncated = True, False
                print(f"[STEP] 조기 중단 패널티 적용: {reward:.1f}")
            else:
                reward = self.early_done_penalty
                terminated, truncated = True, False
                print(f"[STEP] 조기 중단 패널티 적용 (obs 없음): {reward:.1f}")

            self.episode_reward_sum += reward
            return np.zeros(self.observation_space.shape[0]), reward, terminated, truncated, self._make_episode_info()


# ─────────────────────────────────────────────────────────────────────────────
# 알고리즘 프리셋 선택
# ─────────────────────────────────────────────────────────────────────────────

def get_ppo_profile():
    profile = os.getenv("PPO_PROFILE")
    if not profile and len(sys.argv) > 1:
        profile = sys.argv[1]
    profile = (profile or "B").upper()
    if profile not in PPO_PRESETS:
        print(f"알 수 없는 프로필 '{profile}'. B 프로필로 진행합니다.")
        profile = "B"
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# 리플레이 버퍼 시딩 (algo2/algo3 CSV → SAC/TD3 버퍼)
# ─────────────────────────────────────────────────────────────────────────────

def seed_replay_buffer_from_csv(model, rb_dir: str) -> int:
    """
    output/rb/ 의 algo2/algo3 CSV를 읽어 replay buffer에 직접 주입.
    환경변수 RB_SEED_DIR 로 활성화. SAC/TD3 전용.
    steering_cmd 컬럼 없는 구버전 파일은 자동 스킵.
    """
    import glob
    import pandas as pd

    # 상대경로면 스크립트 위치 기준 절대경로로 변환 (CarMaker가 cwd를 바꿔도 동작)
    rb_path = Path(rb_dir)
    if not rb_path.is_absolute():
        rb_path = _SCRIPT_DIR / rb_path
    rb_dir = str(rb_path)

    # 에너지 계산용 모터맵 (batt_rl vs batt_ref 비교에 사용)
    _WHEEL_R = 0.33  # 타이어 반경(m) 근사 - rotv = speed_mps / WHEEL_R
    _mm = MotorMap(Path(os.getenv("MG_MAP_PATH", str(_SCRIPT_DIR / "scaled_mg_map.mat"))))
    _mm.load()

    files = sorted(
        glob.glob(os.path.join(rb_dir, "*_algo2_*.csv")) +
        glob.glob(os.path.join(rb_dir, "*_algo3_*.csv"))
    )
    if not files:
        print(f"[SEED] algo2/algo3 CSV 없음: {rb_dir}")
        return 0

    all_obs, all_nxt, all_act, all_rew, all_don = [], [], [], [], []

    for fpath in files:
        df = pd.read_csv(fpath)
        if "steering_cmd" not in df.columns:
            print(f"[SEED] steering_cmd 없음 (구버전), 스킵: {os.path.basename(fpath)}")
            continue
        if len(df) < 2:
            continue

        # ── obs ─────────────────────────────────────────────────────────────
        total_torque = (df["sat_drive_torque_total_nm"].values
                        - df["sat_regen_torque_total_nm"].values)
        yaw_rate = df["veh_yaw_rate"].values
        ref_yr   = df["ref_yaw_rate"].values
        obs = (np.column_stack([
            df["veh_speed"].values * 3.6,       # speed_kph
            yaw_rate,                            # yaw_rate
            df["veh_ax"].values,                 # ax
            df["veh_ay"].values,                 # ay
            total_torque,                        # total_torque
            df["steering_cmd"].values,           # steering_cmd
            (yaw_rate - ref_yr) ** 2,            # yaw_rate_sq_error
        ]) / OBS_SCALE).astype(np.float32)

        # ── action 역산 ──────────────────────────────────────────────────────
        fl = df["torque_fl"].values
        fr = df["torque_fr"].values
        rl = df["torque_rl"].values
        rr = df["torque_rr"].values
        lt = fl + rl
        rt = fr + rr
        ltb = (lt - rt) / (4.0 * WHEEL_TORQUE_LIMIT)
        lfb = np.where(np.abs(lt) > 1e-6, (fl - rl) / lt, 0.0)
        rfb = np.where(np.abs(rt) > 1e-6, (fr - rr) / rt, 0.0)
        act = np.clip(np.column_stack([ltb, lfb, rfb]), -1.0, 1.0).astype(np.float32)

        # ── reward ───────────────────────────────────────────────────────────
        wt   = np.column_stack([fl, fr, rl, rr])
        yden       = np.maximum(np.abs(ref_yr), YAW_RATE_NORM_EPS)
        yaw_norm   = np.clip(np.abs((yaw_rate - ref_yr) / yden), 0.0, 2.0)
        yaw_pen    = YAW_RATE_WEIGHT * yaw_norm ** 2
        effort_pen = ACTION_EFFORT_WEIGHT * np.sum((wt / WHEEL_TORQUE_LIMIT) ** 2, axis=1)
        # 에너지: 동일 총 요구토크를 등분배했을 때 vs RL(algo2/3) 실제 배분 비교
        # rotv는 현재 속도로 근사 (omega ≈ v / r_wheel, 4바퀴 동일 가정)
        speed_mps = df["veh_speed"].values
        omega_vec = speed_mps / _WHEEL_R
        total_tq  = fl + fr + rl + rr
        tq_rl_mat  = np.column_stack([fl, fr, rl, rr]).astype(np.float64)
        tq_ref_mat = np.column_stack([total_tq / 4.0] * 4).astype(np.float64)
        batt_rl_arr  = compute_batt_net_power_batch(tq_rl_mat,  omega_vec, _mm)
        batt_ref_arr = compute_batt_net_power_batch(tq_ref_mat, omega_vec, _mm)
        energy_saving = (batt_ref_arr - batt_rl_arr) / np.maximum(np.abs(batt_ref_arr), 1.0)
        energy_pen = (-ENERGY_LOSS_WEIGHT * energy_saving).astype(np.float32)
        rew = -(yaw_pen + effort_pen + energy_pen).astype(np.float32)

        # ── t → t+1 페어링 ──────────────────────────────────────────────────
        n = len(obs) - 1
        done = np.zeros(n, dtype=np.float32)
        done[-1] = 1.0  # 에피소드 끝

        all_obs.append(obs[:-1])
        all_nxt.append(obs[1:])
        all_act.append(act[:-1])
        all_rew.append(rew[:-1])
        all_don.append(done)

    if not all_obs:
        print("[SEED] 유효한 파일 없음")
        return 0

    obs_a = np.concatenate(all_obs)
    nxt_a = np.concatenate(all_nxt)
    act_a = np.concatenate(all_act)
    rew_a = np.concatenate(all_rew)
    don_a = np.concatenate(all_don)

    total   = len(obs_a)
    buf     = model.replay_buffer
    buf_cap = buf.buffer_size

    if total > buf_cap:
        idx   = np.random.choice(total, buf_cap, replace=False)
        idx.sort()
        obs_a, nxt_a = obs_a[idx], nxt_a[idx]
        act_a, rew_a = act_a[idx], rew_a[idx]
        don_a = don_a[idx]
        total = buf_cap
        print(f"[SEED] 서브샘플 → {total:,}")

    # 버퍼 배열에 직접 쓰기 (n_envs=1 가정)
    buf.observations[:total, 0, :]      = obs_a
    buf.next_observations[:total, 0, :] = nxt_a
    buf.actions[:total, 0, :]           = act_a
    buf.rewards[:total, 0]              = rew_a.reshape(-1, 1) if buf.rewards.ndim == 3 else rew_a
    buf.dones[:total, 0]                = don_a
    # timeouts: 에피소드 경계(done=1)를 timeout(truncation)으로 표시
    if hasattr(buf, "timeouts"):
        buf.timeouts[:total, 0] = don_a
    buf.pos  = total % buf_cap
    buf.full = total >= buf_cap

    print(f"[SEED] {total:,}개 transition 주입 완료 ({len(files)}개 파일)")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def run_learning(env):
    env  = Monitor(env, info_keywords=("yaw_penalty", "effort_penalty", "saturation_penalty", "energy_penalty"))
    algo = os.getenv("ALGO", "ppo").strip().lower()
    if algo not in {"ppo", "sac", "td3"}:
        raise ValueError(f"ALGO는 'ppo', 'sac', 'td3' 이어야 합니다. 받은 값: {algo}")

    profile = get_ppo_profile()
    if algo == "ppo":
        algo_kwargs  = PPO_PRESETS[profile]
        model_class  = PPO
    elif algo == "sac":
        algo_kwargs  = SAC_PRESETS[profile]
        model_class  = SAC
    else:
        algo_kwargs  = TD3_PRESETS[profile]
        model_class  = TD3

    # 커리큘럼 단계별 모델 경로 및 기본 timesteps 설정
    if CURRICULUM_STAGE in (1, 2):
        _basename = f"{MODEL_BASENAME}_s{CURRICULUM_STAGE}"
        print(f"--- [CURRICULUM] Stage {CURRICULUM_STAGE} ---")
    else:
        _basename = MODEL_BASENAME

    model_path          = f"{_basename}.zip"
    best_model_path     = f"{_basename}_best.zip"
    latest_model_path   = f"{_basename}.zip"
    interrupt_model_path = f"{_basename}_interrupt.zip"
    tensorboard_log_dir = os.getenv("TENSORBOARD_LOG", f"{_basename}_tensorboard")
    callback = SaveBestEpisodeRewardCallback(best_model_path, latest_model_path=latest_model_path)

    print(f"--- ALGO: {algo.upper()} | 프로필: {profile} ---")
    print(f"--- 설정: {algo_kwargs} ---")
    print(f"--- TensorBoard 로그 경로: {tensorboard_log_dir} ---")

    load_path = os.getenv("LOAD_MODEL_PATH", "").strip()
    # Stage 2: LOAD_MODEL_PATH 미지정 시 Stage 1 best 모델 자동 탐색
    if not load_path and CURRICULUM_STAGE == 2:
        s1_best = f"{MODEL_BASENAME}_s1_best.zip"
        if os.path.exists(s1_best):
            load_path = s1_best
            print(f"[CURRICULUM] Stage 2: Stage 1 best 자동 로드: '{load_path}'")
        else:
            print(f"[CURRICULUM] Stage 2: Stage 1 best 없음 ({s1_best}), 새 모델로 시작합니다.")

    if load_path and os.path.exists(load_path):
        print(f"--- 체크포인트 로드: '{load_path}' -> 저장 경로: '{model_path}' ---")
        model = model_class.load(load_path, env=env, device="cuda", verbose=1, tensorboard_log=tensorboard_log_dir)
    elif os.path.exists(model_path):
        print(f"--- 기존 모델 '{model_path}'을(를) 불러와서 학습을 재개합니다. ---")
        model = model_class.load(model_path, env=env, device="cuda", verbose=1, tensorboard_log=tensorboard_log_dir)
    else:
        if load_path:
            print(f"[WARN] LOAD_MODEL_PATH='{load_path}' 파일 없음. 새 모델로 시작합니다.")
        print(f"--- 기존 모델이 없습니다. 새로운 {algo.upper()} 모델을 생성합니다. ---")
        model = model_class("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=tensorboard_log_dir, **algo_kwargs)

    # 로드한 경우 프리셋 하이퍼파라미터를 덮어씀
    if algo == "ppo":
        model.n_steps        = algo_kwargs["n_steps"]
        model.batch_size     = algo_kwargs["batch_size"]
        model.gamma          = algo_kwargs["gamma"]
        model.learning_rate  = algo_kwargs["learning_rate"]
        model.ent_coef       = algo_kwargs["ent_coef"]
    elif algo == "sac":
        model.batch_size     = algo_kwargs["batch_size"]
        model.gamma          = algo_kwargs["gamma"]
        model.learning_rate  = algo_kwargs["learning_rate"]
        model.ent_coef       = algo_kwargs["ent_coef"]
        model.tau            = algo_kwargs["tau"]
        model.buffer_size    = algo_kwargs["buffer_size"]
        model.learning_starts = algo_kwargs["learning_starts"]
        model.gradient_steps = algo_kwargs["gradient_steps"]
    else:  # td3
        model.batch_size          = algo_kwargs["batch_size"]
        model.gamma               = algo_kwargs["gamma"]
        model.learning_rate       = algo_kwargs["learning_rate"]
        model.tau                 = algo_kwargs["tau"]
        model.buffer_size         = algo_kwargs["buffer_size"]
        model.learning_starts     = algo_kwargs["learning_starts"]
        model.gradient_steps      = algo_kwargs["gradient_steps"]
        model.policy_delay        = algo_kwargs["policy_delay"]
        model.target_policy_noise = algo_kwargs["target_policy_noise"]
        model.target_noise_clip   = algo_kwargs["target_noise_clip"]

    # 리플레이 버퍼 시딩 (SAC/TD3 전용, RB_SEED_DIR 설정 시)
    rb_seed_dir = os.getenv("RB_SEED_DIR", "").strip()
    if rb_seed_dir and algo in {"sac", "td3"}:
        n_seeded = seed_replay_buffer_from_csv(model, rb_seed_dir)
        if n_seeded > 0:
            # 버퍼가 이미 채워졌으므로 warm-up 대기 없이 즉시 gradient update 시작
            model.learning_starts = 0
            print(f"[SEED] learning_starts → 0 (버퍼 {n_seeded:,}개 pre-filled)")

    # Stage 1 기본 max step: 50만, 그 외: 100만 (TOTAL_TIMESTEPS 환경변수로 덮어쓰기 가능)
    _default_ts = "500000" if CURRICULUM_STAGE == 1 else "1000000"
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", _default_ts))
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("학습 중단 요청됨. 모델 저장 중...")
        model.save(interrupt_model_path)
        print(f"중단 시점 모델이 '{interrupt_model_path}'에 저장되었습니다.")
    finally:
        model.save(model_path)
        print(f"모델이 '{model_path}'에 저장되었습니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    proj_path = Path(os.getenv("CM_PROJECT_PATH", "/home/khkhh/CM_Projects/assessment_of_4wid_ver14"))
    cmapi.Project.load(proj_path)

    cm_port     = int(os.getenv("CM_PORT", "5555"))
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
    testrun      = cmapi.Project.instance().load_testrun_parametrization(
        proj_path / f"Data/TestRun/{testrun_name}"
    )
    variation = cmapi.Variation.create_from_testrun(testrun)

    loop      = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop)
    print(f"[CONFIG] CONTROL_DT={env_async.control_dt:.3f}s | CM_PORT={cm_port} | TESTRUN={testrun_name}")
    if OPEN_LOOP_STEER:
        print(
            f"[CONFIG] OPEN_LOOP_STEER=1 | AMP_DEG={OPEN_LOOP_STEER_AMP_DEG:.1f} | "
            f"PERIOD_S={OPEN_LOOP_STEER_PERIOD_S:.2f} | START_S={OPEN_LOOP_STEER_START_S:.2f} | "
            f"CYCLES={OPEN_LOOP_STEER_CYCLES} | SPEED_MODE={OPEN_LOOP_SPEED_MODE} | "
            f"SPEED_KPH={OPEN_LOOP_SPEED_KPH:.1f} | MIN_KPH={OPEN_LOOP_SPEED_MIN_KPH:.1f} | "
            f"MAX_KPH={OPEN_LOOP_SPEED_MAX_KPH:.1f}"
        )
    print(
        f"[CONFIG] SCENARIO_DIR={SCENARIO_DIR} | "
        f"SCENARIO_CSV_PATH={SCENARIO_CSV_PATH or '(auto from scenarios)'} | PID={pid}"
    )

    cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))

    env_sync = SyncBridge(env_async, loop)
    await loop.run_in_executor(None, run_learning, env_sync)

    print("Learning finished. Sending stop signal to orchestrator...")
    env_async.stop_req.set()
    loop.stop()


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())

#!/usr/bin/env python3
import asyncio
import csv
import os
import socket
import struct
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import cvxpy as cp

from torque_algorithms import TorqueDistributionAlgorithms

BASE_DIR = Path(__file__).resolve().parent


# ============================================================================
# CONFIGURATION - Project Paths, Ports, and Defaults
# ============================================================================

# CarMaker API Configuration
CMAPI_PY_PATH = os.getenv("CMAPI_PY_PATH", "/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")

# Default Project and TestRun Paths
DEFAULT_PROJECT_PATH = "/home/khkhh/CM_Projects/assessment_of_4wid_ver14"
DEFAULT_TESTRUN_NAME = "test1"

# Network Configuration
DEFAULT_PORT = 5555
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CM_OBS_DIM = int(os.getenv("CM_OBS_DIM", "23"))

# CarMaker Process Configuration
CARMAKER_PROCESS_TARGET = "CarMaker.linux64"
DEFAULT_CARMAKER_EXECUTABLE_SUBPATH = "bin/CarMaker.linux64"

# Simulation Configuration
DEFAULT_EPISODES = 1
DEFAULT_CONTROL_DT = 0.05
DEFAULT_MAX_STEPS_BY_CYCLE = {
    "HWFET": 76500,
    "FTP": 247400,
    "FTP75": 247400,
    "WLTP": 180000,
}
DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS_BY_CYCLE["HWFET"]
DEFAULT_REALTIME_FACTOR = 200.0
DEFAULT_LOG_EVERY = 20
DEFAULT_LOG_FLUSH_EVERY = 200
DEFAULT_LOG_OUTPUT_DIR = os.getenv("LOG_OUTPUT_DIR", str(BASE_DIR / "output"))

# Run Modes
RUN_MODE_GUI = "GUI"
RUN_MODE_HEADLESS = "HEADLESS"

# Simulation State
SIM_RUNNING_STATE = 8.0

# Scenario Configuration
DEFAULT_SCENARIO_CSV_PATH = os.getenv("SCENARIO_CSV_PATH", "/home/khkhh/Projects/scenario_gen_multi/data/WLTP_scenario_wlines.csv")  # CSV 시나리오 경로
DEFAULT_MG_MAP_PATH = os.getenv("MG_MAP_PATH", str(BASE_DIR / "scaled_mg_map.mat"))
DEFAULT_GLOBAL_OPTIM_MAP_PATH = os.getenv("GLOBAL_OPTIM_MAP_PATH", str(BASE_DIR / "global_optim_map.mat"))

# ============================================================================
# Load CarMaker Python API
# ============================================================================
sys.path.append(CMAPI_PY_PATH)
import cmapi


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def infer_cycle_from_scenario_path(scenario_csv_path: Optional[Path]) -> Optional[str]:
    if scenario_csv_path is None:
        return None

    name = scenario_csv_path.name.upper()
    if "HWFET" in name:
        return "HWFET"
    if "FTP75" in name:
        return "FTP75"
    if "FTP" in name:
        return "FTP"
    if "WLTP" in name:
        return "WLTP"
    return None


def get_carmaker_pid() -> Optional[int]:
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = proc.info.get("cmdline") or []
            if CARMAKER_PROCESS_TARGET in name or any(CARMAKER_PROCESS_TARGET in arg for arg in cmdline):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def build_master(run_mode: str, project_path: Path):
    if run_mode == RUN_MODE_HEADLESS:
        master = cmapi.CarMaker()
        executable_path = Path(os.getenv("CM_EXECUTABLE_PATH", str(project_path / DEFAULT_CARMAKER_EXECUTABLE_SUBPATH)))
        master.set_executable_path(executable_path)
        return master

    pid = get_carmaker_pid()
    if pid is None:
        raise RuntimeError("Could not find a running CarMaker GUI process.")

    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=pid, description="Idle"))
    master.set_host("localhost")
    return master


@dataclass
class PIDState:
    integral: float = 0.0
    last_error: float = 0.0


@dataclass
class YawMomentState:
    x_hat: np.ndarray
    p_cov: np.ndarray
    prev_yawrate_ref: float = 0.0


class RequiredYawMomentController:
    """EKF lateral-force observer + sliding-mode yaw-moment controller."""

    def __init__(self):
        self.smc_k = env_float("YAWM_SMC_K", 2.5)
        self.smc_eps = env_float("YAWM_SMC_EPS", 0.2)
        self.yaw_moment_limit = env_float("YAWM_LIMIT_NM", 10000.0)

        # EKF model parameters
        self.mass_kg = env_float("YREF_MASS_KG", 2065.03)
        self.iz = env_float("YREF_IZ", 3637.526)
        self.lf = env_float("YREF_A_M", 1.169)
        self.lr = env_float("YREF_B_M", 1.801)

        # EKF tuning (same baseline as user-provided MATLAB code)
        self.qc = np.diag(
            [
                env_float("EKF_Q_R", 0.01),
                env_float("EKF_Q_FYF", 1_000_000.0),
                env_float("EKF_Q_FYR", 1_000_000.0),
            ]
        ).astype(np.float64)
        self.r = np.diag(
            [
                env_float("EKF_R_AY", 0.01),
                env_float("EKF_R_R", 0.01),
            ]
        ).astype(np.float64)
        self.h = np.array(
            [
                [0.0, 1.0 / self.mass_kg, 1.0 / self.mass_kg],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        x0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        p0 = np.diag(
            [
                env_float("EKF_P0_R", 1.0),
                env_float("EKF_P0_FYF", 10000.0),
                env_float("EKF_P0_FYR", 10000.0),
            ]
        ).astype(np.float64)
        self.state = YawMomentState(x_hat=x0, p_cov=p0, prev_yawrate_ref=0.0)

        # Reference yaw-rate model parameters
        self.steer_ratio = env_float("STEER_RATIO", 16.0)
        self.mass_kg = env_float("YREF_MASS_KG", 2065.03)
        self.a_m = env_float("YREF_A_M", 1.169)
        self.b_m = env_float("YREF_B_M", 1.801)
        self.cf = env_float("YREF_CF", 180000.0)*2
        self.cr = env_float("YREF_CR", 120000.0)*2
        self.mu = env_float("YREF_MU", 0.95)
        self.g = env_float("YREF_G", 9.81)
        self.min_turn_radius_m = env_float("YREF_MIN_TURN_RADIUS_M", 10.0)

    def reset(self) -> None:
        self.state = YawMomentState(
            x_hat=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            p_cov=np.diag([1.0, 10000.0, 10000.0]).astype(np.float64),
            prev_yawrate_ref=0.0,
        )

    def _ekf_step(self, ay_meas: float, gamma_meas: float, dt: float) -> np.ndarray:
        a_c = np.array(
            [
                [0.0, self.lf / self.iz, -self.lr / self.iz],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        phi = np.eye(3, dtype=np.float64) + a_c * dt
        qd = self.qc * dt

        x_pred = phi @ self.state.x_hat
        p_pred = phi @ self.state.p_cov @ phi.T + qd

        z = np.array([float(ay_meas), float(gamma_meas)], dtype=np.float64)
        s = self.h @ p_pred @ self.h.T + self.r
        k_gain = p_pred @ self.h.T @ np.linalg.inv(s)
        y_tilde = z - (self.h @ x_pred)

        x_upd = x_pred + k_gain @ y_tilde
        p_upd = (np.eye(3, dtype=np.float64) - k_gain @ self.h) @ p_pred

        self.state.x_hat = x_upd
        self.state.p_cov = p_upd
        return x_upd

    def compute_required_yaw_moment(
        self,
        ay_meas: float,
        veh_yaw_rate: float,
        yawrate_ref: float,
        yawrate_ref_dot: float,
        dt: float,
    ) -> float:
        x_hat = self._ekf_step(ay_meas=ay_meas, gamma_meas=veh_yaw_rate, dt=dt)
        fy_f = float(x_hat[1])
        fy_r = float(x_hat[2])
        se = float(veh_yaw_rate) - float(yawrate_ref)

        if self.smc_eps > 1e-9:
            ratio = se / self.smc_eps
            smc_term = float(np.tanh(ratio)**3)
        else:
            smc_term = float(np.sign(se))

        yawrate_ref_term = self.iz * float(yawrate_ref_dot)
        # print(yawrate_ref_term)
        yawrate_ref_term = float(np.clip(yawrate_ref_term, -self.yaw_moment_limit, self.yaw_moment_limit))
        # print(yawrate_ref_term)
        req_yaw_moment = (
            -self.lf * fy_f
            + self.lr * fy_r
            - self.smc_k * smc_term * self.iz
            + yawrate_ref_term
        )
        # req_yaw_moment = 0
        return float(np.clip(req_yaw_moment, -self.yaw_moment_limit, self.yaw_moment_limit))

    def compute_reference_yaw_rate(self, veh_speed: float, steering_cmd: float) -> float:
        vx = abs(float(veh_speed))
        if vx < 1e-6:
            return 0.0

        # Stanley output is steering-wheel angle, convert to front-wheel angle.
        delta_f = float(steering_cmd) / max(self.steer_ratio, 1e-6)

        ab = self.a_m + self.b_m
        
        numerator = vx * self.cf * self.cr * ab * delta_f
        denominator = (self.cf * self.cr * (ab ** 2)) + (self.mass_kg * (vx ** 2) * (self.b_m * self.cr - self.a_m * self.cf))
        if abs(denominator) < 1e-9:
            yawrate_ideal = 0.0
        else:
            yawrate_ideal = numerator / denominator

        yawrate_max_friction = abs(self.mu * self.g / vx)
        yawrate_max_minturnrad = abs(vx / max(self.min_turn_radius_m, 1e-6))
        yawrate_max = min(yawrate_max_friction, yawrate_max_minturnrad)

        return float(min(abs(yawrate_ideal), yawrate_max) * np.sign(delta_f))


class PIDTorqueController:
    def __init__(self):
        self.kp = env_float("CTRL_KP", 1000.0)
        self.ki = env_float("CTRL_KI", 200.0)
        self.kd = env_float("CTRL_KD", 1.0)
        self.vdiff_sign = env_float("V_DIFF_SIGN", 1.0)

        self.total_torque_limit = env_float("TOTAL_TORQUE_LIMIT", 630.0*4)
        self.wheel_torque_limit = env_float("WHEEL_TORQUE_LIMIT", 630.0)

        self.front_ratio = env_float("FRONT_RATIO", 0.5)
        self.track_width_m = env_float("TRACK_WIDTH_M", 1.63)
        self.wheel_radius_m = env_float("WHEEL_RADIUS_M", 0.36)
        self.mass_kg = env_float("YREF_MASS_KG", 2065.03)
        self.g = env_float("YREF_G", 9.81)
        self.lf = env_float("YREF_A_M", 1.169)
        self.lr = env_float("YREF_B_M", 1.801)
        self.iz = env_float("YREF_IZ", 3637.526)
        self.cf = env_float("YREF_CF", 180000.0)*2
        self.cr = env_float("YREF_CR", 120000.0)*2
        self.steer_ratio = env_float("STEER_RATIO", 16.0)
        self.cg_height_m = env_float("CG_HEIGHT_M", 0.55)
        self.yaw_front_share = env_float("YAW_FRONT_SHARE", 0.5)
        self.distribution_mode = os.getenv("TORQUE_DISTRIBUTION_MODE", "algo2").lower()

        self.beta_min = env_float("BETA_MIN_RAD", -0.15)
        self.beta_max = env_float("BETA_MAX_RAD", 0.15)
        self.rho_start = env_float("RHO_START", 0.7)
        self.rho_end = env_float("RHO_END", 1.0)

        self.energy_grid_step = env_float("ENERGY_SHARE_STEP", 0.1)

        self.integral_limit = env_float("INTEGRAL_LIMIT", 2000.0)
        self.state = PIDState()

        # QP cache (algo2): rho별 QP 미리 생성
        self._algo4_qp_count = env_int("ALGO4_QP_COUNT", 21)
        self._algo4_qp_problems: list[cp.Problem] = []
        self._algo4_qp_taus: list[cp.Variable] = []
        self._algo4_qp_params: list[dict[str, cp.Parameter]] = []
        self._algo4_qp_rhos: list[float] = []
        self._algo4_qp_coeff: float | None = None
        self._init_algo4_qp_grid()
        self.algorithms = TorqueDistributionAlgorithms(self, env_int, env_float, cp)

    def _init_algo4_qp_grid(self):
        # 0~1까지 N개 rho로 QP 미리 생성
        self._algo4_qp_problems.clear()
        self._algo4_qp_taus.clear()
        self._algo4_qp_params.clear()
        self._algo4_qp_rhos.clear()
        coeff = (self.track_width_m / 2.0) / max(self.wheel_radius_m, 1e-6)
        count = max(int(self._algo4_qp_count), 2)
        for i in range(count):
            rho = i / (count - 1)
            tau = cp.Variable(4)
            p_total = cp.Parameter(name="total_torque")
            p_req_mz = cp.Parameter(name="req_yaw_moment")
            p_stab_targets = cp.Parameter(4, name="stability_targets")
            p_energy_targets = cp.Parameter(4, name="energy_targets")
            p_w_virtual = cp.Parameter(nonneg=True, name="w_virtual")
            p_w_stability = cp.Parameter(nonneg=True, name="w_stability")
            p_w_energy = cp.Parameter(nonneg=True, name="w_energy")
            p_w_torque = cp.Parameter(nonneg=True, name="w_torque")
            mz = float(coeff) * (tau[1] - tau[0] + tau[3] - tau[2])
            constraints = [
                cp.sum(tau) == p_total,
                tau <= float(self.wheel_torque_limit),
                tau >= -float(self.wheel_torque_limit),
            ]
            j_virtual = p_w_virtual * cp.square(mz - p_req_mz)
            # p_stab_targets carries inverse Fz weights for normalization.
            j_stability = p_w_stability * cp.sum_squares(cp.multiply(tau, p_stab_targets))
            j_energy = p_w_energy * cp.sum_squares(tau - p_energy_targets)
            j_reg = p_w_torque * cp.sum_squares(tau)
            objective = cp.Minimize(j_virtual + (rho * j_stability) + ((1.0 - rho) * j_energy) + j_reg)
            problem = cp.Problem(objective, constraints)
            self._algo4_qp_problems.append(problem)
            self._algo4_qp_taus.append(tau)
            self._algo4_qp_params.append({
                "total_torque": p_total,
                "req_yaw_moment": p_req_mz,
                "stability_targets": p_stab_targets,
                "energy_targets": p_energy_targets,
                "w_virtual": p_w_virtual,
                "w_stability": p_w_stability,
                "w_energy": p_w_energy,
                "w_torque": p_w_torque,
            })
            self._algo4_qp_rhos.append(rho)

    def reset(self) -> None:
        self.state = PIDState()

    def _ensure_algo4_qp(self) -> None:
        if self._algo4_qp_problem is not None:
            return

        coeff = (self.track_width_m) / max(self.wheel_radius_m, 1e-6)
        self._algo4_qp_coeff = float(coeff)

        tau = cp.Variable(4)
        p_total = cp.Parameter(name="total_torque")
        p_req_mz = cp.Parameter(name="req_yaw_moment")
        p_rho = cp.Parameter(name="rho")

        p_stab_targets = cp.Parameter(4, name="stability_targets")
        p_energy_targets = cp.Parameter(4, name="energy_targets")

        p_w_virtual = cp.Parameter(nonneg=True, name="w_virtual")
        p_w_stability = cp.Parameter(nonneg=True, name="w_stability")
        p_w_energy = cp.Parameter(nonneg=True, name="w_energy")
        p_w_torque = cp.Parameter(nonneg=True, name="w_torque")

        mz = float(coeff) * (tau[1] - tau[0] + tau[3] - tau[2])

        constraints = [
            cp.sum(tau) == p_total,
            tau <= float(self.wheel_torque_limit),
            tau >= -float(self.wheel_torque_limit),
        ]

        j_virtual = p_w_virtual * cp.square(mz - p_req_mz)
        # p_stab_targets carries inverse Fz weights for normalization.
        j_stability = p_w_stability * cp.sum_squares(cp.multiply(tau, p_stab_targets))
        j_energy = p_w_energy * cp.sum_squares(tau - p_energy_targets)
        j_reg = p_w_torque * cp.sum_squares(tau)
        objective = cp.Minimize(j_virtual + (p_rho * j_stability) + ((1.0 - p_rho) * j_energy) + j_reg)
        # objective = cp.Minimize((j_stability) + ((1.0 - p_rho) * j_energy) + j_reg)

        self._algo4_qp_problem = cp.Problem(objective, constraints)
        self._algo4_qp_tau = tau
        self._algo4_qp_params = {
            "total_torque": p_total,
            "req_yaw_moment": p_req_mz,
            "rho": p_rho,
            "stability_targets": p_stab_targets,
            "energy_targets": p_energy_targets,
            "w_virtual": p_w_virtual,
            "w_stability": p_w_stability,
            "w_energy": p_w_energy,
            "w_torque": p_w_torque,
        }

    def distribute_torque(
        self,
        total_torque: float,
        req_yaw_moment: float,
        veh_speed: float,
        veh_ax: float = 0.0,
        veh_ay: float = 0.0,
        yaw_rate: float = 0.0,
        ref_yaw_rate: float = 0.0,
        steering_cmd: float = 0.0,
        dt: float = DEFAULT_CONTROL_DT,
        rotv: np.ndarray | None = None,
        motor_map=None,
        fy_front: float = 0.0,
        fy_rear: float = 0.0,
    ) -> np.ndarray:
        rotv_arr = np.zeros(4, dtype=np.float64) if rotv is None else np.asarray(rotv, dtype=np.float64)
        return self.algorithms.distribute_torque(
            total_torque=total_torque,
            req_yaw_moment=req_yaw_moment,
            veh_speed=veh_speed,
            veh_ax=veh_ax,
            veh_ay=veh_ay,
            yaw_rate=yaw_rate,
            rotv=rotv_arr,
            motor_map=motor_map,
            fy_front=fy_front,
            fy_rear=fy_rear,
        )

    def compute_total_torque(self, v_diff: float, dt: float) -> float:
        error = self.vdiff_sign * float(v_diff)
        self.state.integral += error * dt
        self.state.integral = float(np.clip(self.state.integral, -self.integral_limit, self.integral_limit))

        derivative = (error - self.state.last_error) / max(dt, 1e-6)
        self.state.last_error = error

        total_torque = self.kp * error + self.ki * self.state.integral + self.kd * derivative
        total_torque = float(np.clip(total_torque, -self.total_torque_limit, self.total_torque_limit))
        return total_torque


class StanleySteeringController:
    """Stanley 횡제어 컨트롤러 - 경로 추종"""
    
    def __init__(self):
        # Stanley 파라미터
        # self.g1 = env_float("STANLEY_G1", 31)  # 헤딩 에러 가중치
        # self.g2 = env_float("STANLEY_G2", 5)  # 횡방향 에러 가중치
        # self.k = env_float("STANLEY_K", 5.0)    # 크로스트랙 에러 게인
        # self.scale_factor = env_float("STANLEY_SCALE", 3.0)  # 최종 스케일

        self.g1 = env_float("STANLEY_G1", 27)  # 헤딩 에러 가중치
        self.g2 = env_float("STANLEY_G2", 5)  # 횡방향 에러 가중치
        self.k = env_float("STANLEY_K", 5.0)    # 크로스트랙 에러 게인
        self.scale_factor = env_float("STANLEY_SCALE", 1)  # 최종 스케일
        self.wheelbase = env_float("STANLEY_WHEELBASE_M",
                                   env_float("YREF_A_M", 1.169) + env_float("YREF_B_M", 1.801))  # 축거 (m)
        
    def compute(self, veh_x: float, veh_y: float, veh_yaw: float,
                ref_x: float, ref_y: float, ref_yaw: float,
                curvature: float, vel: float) -> Tuple[float, float, float]:
        """
        Stanley 컨트롤러 계산
        
        Args:
            veh_x, veh_y: 차량 위치
            veh_yaw: 차량 헤딩각 (rad)
            ref_x, ref_y: 참조 경로 위치
            ref_yaw: 참조 헤딩각 (rad)
            vel: 차량 속도 (m/s)
        
        Returns:
            stanley_cmd: 조향 명령 (-1 to 1)
            heading_err: 헤딩 에러 (rad)
            cross_track_err: 횡방향 에러 (m)
        """

        # print(f"Stanley Input - Veh: ({veh_x:.2f}, {veh_y:.2f}, yaw={veh_yaw:.2f} rad), "
        # f"Ref: ({ref_x:.2f}, {ref_y:.2f}, yaw={ref_yaw:.2f} rad), vel={vel:.2f} m/s")
        
        # 횡방향 에러 계산 (참조 경로에 수직인 거리)
        cross_track_err = (ref_y - veh_y) * np.cos(ref_yaw) - (ref_x - veh_x) * np.sin(ref_yaw)

        L_f = 1.14 # C.G.에서 전륜축까지의 거리 (차량 제원 확인 필요)
        fx = veh_x + L_f * np.cos(veh_yaw)
        fy = veh_y + L_f * np.sin(veh_yaw)

        # 이후 모든 계산(cross_track_err 등)에 veh_x, y 대신 fx, fy 사용
        cross_track_err = (ref_y - fy) * np.cos(ref_yaw) - (ref_x - fx) * np.sin(ref_yaw)
        
        # 헤딩 에러 계산
        heading_err = ref_yaw - veh_yaw
        
        # 헤딩 에러를 [-π/2, π/2] 범위로 제한
        heading_err = np.clip(heading_err, -np.pi / 2, np.pi / 2)
        
        # 요청 수식:
        # delta = g3 * ( g1 * e_h * (20 / (v_x + 20)) + g2 * (k * e_c / (v_x + 3)) )
        delta_ff = np.arctan(self.wheelbase * curvature)*0  # 곡률 피드포워드
        vel_term_heading = 20.0 / (vel + 1.0)
        vel_term_cte = (self.k * cross_track_err) / (vel/10 + 2.2)**2
        stanley = delta_ff + self.scale_factor * (
            self.g1 * heading_err * vel_term_heading
            + self.g2 * vel_term_cte
        )
        # if os.getenv("TORQUE_DISTRIBUTION_MODE", "algo2").lower() == "algo2":
        #     stanley = delta_ff/2 + self.scale_factor * (
        #     self.g1 * heading_err * vel_term_heading
        #     + self.g2 * vel_term_cte)
        # print(f"Stanley Output - delta_ff={delta_ff:.4f} rad, heading_err={heading_err:.4f} rad, ")
        
        return stanley, heading_err, cross_track_err


class Scenario:
    """CSV 시나리오 파일을 로드하고 시간 기준 선형보간을 통해 참조값 제공"""
    
    def __init__(self):
        self.data = None
        self.time_col = "Time"
        self.x_col = "X"
        self.y_col = "Y"
        self.speed_col = "Speed"
        self.yaw_col = "Yaw"
        self.time_arr = None
        self.x_arr = None
        self.y_arr = None
        self.speed_arr = None
        self.yaw_arr = None
        self.curvature_arr = None
        self.last_idx = 0
        self.last_path_idx = 0

    def _estimate_curvature_from_path(self) -> np.ndarray:
        """X, Y 좌표에서 곡률을 수치 미분으로 추정 (κ = dψ/ds)"""
        yaw = np.unwrap(self.yaw_arr)
        dx = np.diff(self.x_arr)
        dy = np.diff(self.y_arr)
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.where(ds < 1e-6, 1e-6, ds)  # zero-ds 방지
        dyaw = np.diff(yaw)
        kappa = np.append(dyaw / ds, dyaw[-1] / ds[-1])  # 마지막 요소 복사로 길이 맞춤
        return kappa

    def reset(self) -> None:
        self.last_idx = 0
        self.last_path_idx = 0
    
    def load_csv(self, csv_path: Path) -> bool:
        """CSV 파일 로드 및 기본 검증"""
        if not csv_path or not csv_path.exists():
            print(f"[WARN] Scenario CSV file not found: {csv_path}")
            return False
        
        try:
            self.data = pd.read_csv(csv_path)
            required_cols = [self.time_col, self.x_col, self.y_col, self.speed_col, self.yaw_col]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                print(f"[ERROR] CSV missing required columns: {missing_cols}")
                return False

            # Keep arrays in contiguous numpy buffers for fast interpolation.
            self.time_arr = self.data[self.time_col].to_numpy(dtype=np.float64, copy=True)
            self.x_arr = self.data[self.x_col].to_numpy(dtype=np.float64, copy=True)
            self.y_arr = self.data[self.y_col].to_numpy(dtype=np.float64, copy=True)
            self.speed_arr = self.data[self.speed_col].to_numpy(dtype=np.float64, copy=True)
            self.yaw_arr = self.data[self.yaw_col].to_numpy(dtype=np.float64, copy=True)
            # 곡률 컬럼 탐색 (없으면 경로에서 추정)
            curvature_col = next(
                (c for c in ("Curvature", "curvature", "Kappa", "kappa", "Curv", "curv")
                 if c in self.data.columns), None
            )
            if curvature_col:
                self.curvature_arr = self.data[curvature_col].to_numpy(dtype=np.float64, copy=True)
                print(f"[INFO] Scenario curvature column loaded: '{curvature_col}'")
            else:
                self.curvature_arr = self._estimate_curvature_from_path()
                print("[INFO] Scenario curvature estimated from path (no column found)")
            self.last_idx = 0
            
            print(f"[INFO] Scenario loaded: {len(self.data)} samples from {csv_path}")
            print(f"[INFO] Time range: {self.data[self.time_col].min():.3f} ~ {self.data[self.time_col].max():.3f} s")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load scenario CSV: {e}")
            return False
    
    def get_reference_state(self, sim_time: float,
                            veh_x: float = None, veh_y: float = None) -> Optional[dict]:
        """
        속도: 시간 기반 보간 / x,y,yaw,curvature: 시간 기준 주변 점 중 최근접
        veh_x, veh_y 미제공 시 모두 시간 기반으로 fallback.
        """
        if self.time_arr is None or len(self.time_arr) == 0:
            return None

        if sim_time < self.time_arr[0] or sim_time > self.time_arr[-1]:
            return None

        n = len(self.time_arr)

        # --- Speed: 시간 기반 증분 보간 ---
        i_t = self.last_idx
        while i_t + 1 < n and self.time_arr[i_t + 1] < sim_time:
            i_t += 1
        if i_t + 1 >= n:
            i_t = n - 2
        self.last_idx = i_t

        t0 = self.time_arr[i_t]
        t1 = self.time_arr[i_t + 1]
        alpha_t = 0.0 if t1 <= t0 else (sim_time - t0) / (t1 - t0)
        speed_interp = float(
            self.speed_arr[i_t] + alpha_t * (self.speed_arr[i_t + 1] - self.speed_arr[i_t])
        ) / 3.6

        # --- Path state: 시간 기준 주변 점에서 최근접 선택 또는 시간 기반 fallback ---
        if veh_x is not None and veh_y is not None:
            WIN = env_int("REF_NEARBY_WINDOW", 200)
            i0 = max(0, i_t - WIN)
            i1 = min(n - 1, i_t + WIN)
            dx = self.x_arr[i0:i1 + 1] - veh_x
            dy = self.y_arr[i0:i1 + 1] - veh_y
            i_near = int(np.argmin(dx * dx + dy * dy)) + i0
            self.last_path_idx = i_near
            i = min(i_near, n - 2)
            # 세그먼트 투영으로 alpha 계산
            seg_dx = self.x_arr[i + 1] - self.x_arr[i]
            seg_dy = self.y_arr[i + 1] - self.y_arr[i]
            seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len2 > 1e-12:
                alpha = float(np.clip(
                    ((veh_x - self.x_arr[i]) * seg_dx + (veh_y - self.y_arr[i]) * seg_dy) / seg_len2,
                    0.0, 1.0
                ))
            else:
                alpha = 0.0
        else:
            i = i_t
            alpha = alpha_t

        x_interp = float(self.x_arr[i] + alpha * (self.x_arr[i + 1] - self.x_arr[i]))
        y_interp = float(self.y_arr[i] + alpha * (self.y_arr[i + 1] - self.y_arr[i]))
        yaw_interp = float(self.yaw_arr[i] + alpha * (self.yaw_arr[i + 1] - self.yaw_arr[i]))

        curvature_interp = 0.0
        if self.curvature_arr is not None:
            curvature_interp = float(
                self.curvature_arr[i] + alpha * (self.curvature_arr[i + 1] - self.curvature_arr[i])
            )
        return {
            'time': sim_time,
            'x': x_interp,
            'y': y_interp,
            'speed': speed_interp,
            'yaw': yaw_interp,
            'curvature': curvature_interp,
        }


class MotorMap:
    """Motor map helper for torque saturation and efficiency lookup."""

    def __init__(self, map_path: Path):
        self.map_path = map_path
        self.optim_map_path = Path(DEFAULT_GLOBAL_OPTIM_MAP_PATH)
        # Fixed lookup scale: map torque axis is treated as side total torque.
        self.optim_torque_lookup_scale = 1.0
        self.loaded = False
        self.spd_rpm = None
        self.trq_max_nm = None
        self.eff_spd_rpm = None
        self.eff_trq_nm = None
        self.eff_map_pct = None
        self.optim_loaded = False
        self.optim_front_share = None
        self.optim_torque_nm = None
        self.optim_rpm_front = None
        self.optim_rpm_rear = None
        self.optim_front_trq_map = None
        self.optim_demand_trq_map = None
        self.optim_rpm_vector = None
        self.optim_trq_vector = None


    def load(self) -> bool:
        if not self.map_path.exists():
            print(f"[WARN] Motor map not found: {self.map_path}")
            return False

        try:
            import scipy.io as sio
            m = sio.loadmat(self.map_path)
        except Exception as e:
            print(f"[WARN] Failed to load motor map: {e}")
            return False

        try:
            self.spd_rpm = np.asarray(m["spd_rpm"], dtype=np.float64).reshape(-1)
            self.trq_max_nm = np.asarray(m["trq_max_Nm"], dtype=np.float64).reshape(-1)
            self.eff_spd_rpm = np.asarray(m["eff_spd_rpm"], dtype=np.float64).reshape(-1)
            self.eff_trq_nm = np.asarray(m["trq_Nm"], dtype=np.float64).reshape(-1)

            eff_raw = np.asarray(m.get("calc_eff", m.get("original_eff")), dtype=np.float64)
            if eff_raw.ndim != 2:
                raise ValueError("Efficiency map must be 2D")
            if eff_raw.shape[0] != self.eff_trq_nm.size or eff_raw.shape[1] != self.eff_spd_rpm.size:
                raise ValueError(
                    f"Efficiency shape mismatch: map={eff_raw.shape}, "
                    f"trq={self.eff_trq_nm.size}, spd={self.eff_spd_rpm.size}"
                )

            self.eff_map_pct = eff_raw
            self.loaded = True
            print(
                f"[INFO] Motor map loaded: {self.map_path} "
                f"(spd={self.spd_rpm.size}, trq={self.eff_trq_nm.size})"
            )
            self._load_global_optim_map()
            return True
        except Exception as e:
            print(f"[WARN] Invalid motor map contents: {e}")
            self.loaded = False
            return False

    def _load_global_optim_map(self) -> None:
        import os
        import numpy as np
        import pickle
        import scipy.io as sio
        # 우선순위: pkl → npz → mat
        base = str(self.optim_map_path)
        pkl_path = base.replace('.mat', '.pkl')
        npz_path = base.replace('.mat', '.npz')
        mat_path = base

        loaded = False
        # 1. pkl
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    m = pickle.load(f)
                loaded = True
                print(f"[INFO] Global optim map loaded: {pkl_path}")
            except Exception as e:
                print(f"[WARN] Failed to load global optim map (pkl): {e}")
        # 2. npz
        if not loaded and os.path.exists(npz_path):
            try:
                m = dict(np.load(npz_path))
                loaded = True
                print(f"[INFO] Global optim map loaded: {npz_path}")
            except Exception as e:
                print(f"[WARN] Failed to load global optim map (npz): {e}")
        # 3. mat
        if not loaded and os.path.exists(mat_path):
            try:
                m = sio.loadmat(mat_path)
                loaded = True
                print(f"[INFO] Global optim map loaded: {mat_path}")
            except Exception as e:
                print(f"[WARN] Failed to load global optim map (mat): {e}")
        if not loaded:
            print(f"[WARN] No global optim map found: {self.optim_map_path}")
            return

        # 내부 구조 자동 인식
        def _pick_key(candidates):
            for k in candidates:
                if k in m:
                    return k
            return None

        share_key = _pick_key(["front_share", "front_ratio", "share_front", "opt_front_share"])
        tq_key = _pick_key(["torque_nm", "trq_nm", "torque", "trq", "trq_vector"])
        rpm_f_key = _pick_key(["rpm_front", "rpm_f", "rpm_front_rpm", "speed_front_rpm"])
        rpm_r_key = _pick_key(["rpm_rear", "rpm_r", "rpm_rear_rpm", "speed_rear_rpm"])
        front_trq_key = _pick_key(["opt_front_trq_map", "front_trq_map", "opt_front_torque"])
        demand_trq_key = _pick_key(["demand_trq_map", "total_trq_map", "trq_demand_map"])
        rpm_vec_key = _pick_key(["rpm_demand_vector", "rpm_vector", "rpm_grid"])
        trq_vec_key = _pick_key(["trq_demand_vector", "trq_vector", "trq_grid"])

        try:
            # 3D QP 기반 맵 (rpm_f x rpm_r x torque)
            if share_key and tq_key and rpm_f_key and rpm_r_key:
                self.optim_front_share = np.asarray(m[share_key], dtype=np.float64)
                self.optim_torque_nm = np.asarray(m[tq_key], dtype=np.float64).reshape(-1)
                self.optim_rpm_front = np.asarray(m[rpm_f_key], dtype=np.float64).reshape(-1)
                self.optim_rpm_rear = np.asarray(m[rpm_r_key], dtype=np.float64).reshape(-1)
                if self.optim_front_share.ndim != 3:
                    print("[WARN] Global optim map must be 3D (rpm_f x rpm_r x torque)")
                    return
                self.optim_loaded = True
                print("[INFO] Loaded 3D QP-based global optim map.")
                return

            # 2D QP 기반 맵 (rpm x torque)
            if front_trq_key and demand_trq_key and rpm_vec_key and trq_vec_key:
                self.optim_front_trq_map = np.asarray(m[front_trq_key], dtype=np.float64)
                self.optim_demand_trq_map = np.asarray(m[demand_trq_key], dtype=np.float64)
                self.optim_rpm_vector = np.asarray(m[rpm_vec_key], dtype=np.float64).reshape(-1)
                self.optim_trq_vector = np.asarray(m[trq_vec_key], dtype=np.float64).reshape(-1)
                if self.optim_front_trq_map.ndim != 2:
                    print("[WARN] Global optim map must be 2D (rpm x torque)")
                    return
                self.optim_loaded = True
                print("[INFO] Loaded 2D QP-based global optim map.")
                return

            # 2D 효율 기반 맵 (front_share, rpm_vector, trq_vector)
            if share_key and rpm_vec_key and trq_vec_key:
                self.optim_front_share = np.asarray(m[share_key], dtype=np.float64)
                self.optim_rpm_vector = np.asarray(m[rpm_vec_key], dtype=np.float64).reshape(-1)
                self.optim_trq_vector = np.asarray(m[trq_vec_key], dtype=np.float64).reshape(-1)
                if self.optim_front_share.ndim != 2:
                    print("[WARN] Global optim map must be 2D (rpm x torque) for efficiency-based map")
                    return
                self.optim_loaded = True
                print("[INFO] Loaded 2D efficiency-based global optim map.")
                return

            print("[WARN] Global optim map: No recognized key set for known formats (3D QP, 2D QP, 2D efficiency)")
        except Exception as e:
            print(f"[WARN] Invalid global optim map contents: {e}")

    def optimal_front_share(self, rpm_front: float, rpm_rear: float, torque_nm: float) -> float:
        """
        전역 최적 맵 기반 보간
        """

        # 내부 보간기 초기화 (최초 1회)
        if not hasattr(self, '_front_share_interp'):
            from scipy.interpolate import RectBivariateSpline
            if self.optim_front_trq_map is not None and self.optim_demand_trq_map is not None:
                # 2D QP 기반 맵 (front_trq_map, demand_trq_map)
                self._front_share_interp = RectBivariateSpline(
                    self.optim_rpm_vector,
                    self.optim_trq_vector,
                    self.optim_front_trq_map,
                    kx=1,
                    ky=1,
                )
                self._demand_trq_interp = RectBivariateSpline(
                    self.optim_rpm_vector,
                    self.optim_trq_vector,
                    self.optim_demand_trq_map,
                    kx=1,
                    ky=1,
                )
            else:
                # 2D 효율 기반 맵 (front_share)
                self._front_share_interp = RectBivariateSpline(
                    self.optim_rpm_vector,
                    self.optim_trq_vector,
                    self.optim_front_share,
                    kx=1,
                    ky=1,
                )
                self._demand_trq_interp = None

        def _interp_share(rpm_f, rpm_r, tq):
            rpm_avg = 0.5 * (rpm_f + rpm_r)
            trq = self.optim_torque_lookup_scale * tq
            rpm_axis = self.optim_rpm_vector
            trq_axis = self.optim_trq_vector
            rf = float(np.clip(rpm_avg, rpm_axis[0], rpm_axis[-1]))
            # Keep lookup inside the motor's physically feasible torque envelope.
            max_feasible_trq = self.max_torque_nm(rf)
            trq_feasible = float(np.clip(trq, -max_feasible_trq, max_feasible_trq))
            tqc = float(np.clip(trq_feasible, trq_axis[0], trq_axis[-1]))

            if self._demand_trq_interp is None:
                share = float(self._front_share_interp(rf, tqc)[0, 0])
                return float(np.clip(share, 0.0, 1.0))

            front_trq = float(self._front_share_interp(rf, tqc)[0, 0])
            demand_trq = float(self._demand_trq_interp(rf, tqc)[0, 0])
            if abs(demand_trq) < 1e-6:
                return 1.0
            share = (2.0 * front_trq) / (2.0 * demand_trq)
            return float(np.clip(share, 0.0, 1.0))

        # functools.lru_cache 적용 (최대 1024개)
        if not hasattr(self, '_interp_share_cached'):
            try:
                from functools import lru_cache
                self._interp_share_cached = lru_cache(maxsize=1024)(_interp_share)
            except Exception:
                self._interp_share_cached = _interp_share

        return self._interp_share_cached(float(rpm_front), float(rpm_rear), float(torque_nm))

    def max_torque_nm(self, rpm: float) -> float:
        if not self.loaded:
            return float("inf")
        r = abs(float(rpm))
        return float(np.interp(r, self.spd_rpm, self.trq_max_nm, left=self.trq_max_nm[0], right=self.trq_max_nm[-1]))

    def efficiency_pct(self, rpm: float, torque_nm: float) -> float:
        if not self.loaded:
            return 100.0

        t = abs(float(torque_nm))
        if t <= 1e-9:
            return 100.0

        r = abs(float(rpm))
        t_clip = float(np.clip(t, self.eff_trq_nm[0], self.eff_trq_nm[-1]))
        r_clip = float(np.clip(r, self.eff_spd_rpm[0], self.eff_spd_rpm[-1]))

        col_vals = np.array(
            [np.interp(t_clip, self.eff_trq_nm, self.eff_map_pct[:, j]) for j in range(self.eff_spd_rpm.size)],
            dtype=np.float64,
        )
        eff = float(np.interp(r_clip, self.eff_spd_rpm, col_vals))
        return float(np.clip(eff, 1.0, 100.0))


class TelemetryLogger:
    def __init__(self, file_path: Path, flush_every: int = DEFAULT_LOG_FLUSH_EVERY):
        self.file_path = file_path
        self.flush_every = max(flush_every, 1)
        self.buffer = []
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.file_path.open("w", newline="")
        self.writer = csv.writer(self.fp)
        self.writer.writerow(
            [
                "sim_time",
                "veh_x",
                "veh_y",
                "veh_yaw",
                "veh_yaw_rate",
                "veh_speed",
                "veh_ax",
                "veh_ay",
                "rotv_fl",
                "rotv_fr",
                "rotv_rl",
                "rotv_rr",
                "long_slip_loss_fl",
                "long_slip_loss_fr",
                "long_slip_loss_rl",
                "long_slip_loss_rr",
                "lat_slip_loss_fl",
                "lat_slip_loss_fr",
                "lat_slip_loss_rl",
                "lat_slip_loss_rr",
                "aero_drag",
                "roll_resist_fl",
                "roll_resist_fr",
                "roll_resist_rl",
                "roll_resist_rr",
                "req_drive_torque_total_nm",
                "sat_drive_torque_total_nm",
                "req_regen_torque_total_nm",
                "sat_regen_torque_total_nm",
                "batt_drive_power_w",
                "batt_regen_power_w",
                "batt_net_power_w",
                "ref_yaw_rate_base",
                "ref_yaw_rate",
                "torque_fl",
                "torque_fr",
                "torque_rl",
                "torque_rr",
            ]
        )

    def write_row(self, row) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_every:
            self.writer.writerows(self.buffer)
            self.buffer.clear()
            self.fp.flush()

    def close(self) -> None:
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer.clear()
        self.fp.flush()
        self.fp.close()


async def recv_exact(loop: asyncio.AbstractEventLoop, sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = await loop.run_in_executor(None, sock.recv, size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def unpack_observation(raw: bytes, obs_dim: int):
    state_data = struct.unpack("d" * obs_dim, raw)
    veh_x, veh_y, veh_yaw, veh_v = state_data[0:4]
    rotv = state_data[4:8]
    long_slip = state_data[8:12]
    lat_slip = state_data[12:16]
    aero_drag = float(state_data[16])
    roll_resist = state_data[17:21]
    veh_ax = float(state_data[21]) if obs_dim >= 22 else 0.0
    veh_ay = float(state_data[22]) if obs_dim >= 23 else 0.0
    return veh_x, veh_y, veh_yaw, veh_v, veh_ax, veh_ay, rotv, long_slip, lat_slip, aero_drag, roll_resist


async def run_one_episode(
    loop: asyncio.AbstractEventLoop,
    simcontrol,
    variation,
    server_sock: socket.socket,
    controller: PIDTorqueController,
    motor_map: MotorMap,
    scenario: Scenario,
    log_output_dir: Path,
    episode_index: int,
    control_dt: float,
    max_steps: int,
    realtime_factor: float,
    log_every: int,
    scenario_name: str = "",
) -> None:
    client_sock = None
    stanley_controller = StanleySteeringController()
    yaw_moment_controller = RequiredYawMomentController()

    await simcontrol.connect()

    if simcontrol.get_status() != cmapi.SimControlState.configure:
        await simcontrol.stop_sim()
        await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

    simcontrol.set_variation(variation.clone())
    await simcontrol.start_sim()
    await simcontrol.create_simstate_condition(cmapi.ConditionSimState.running).wait()
    simcontrol.set_realtimefactor(realtime_factor)

    controller.reset()
    yaw_moment_controller.reset()
    scenario.reset()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    force_zero_steer = env_bool("FORCE_ZERO_STEER", False)
    open_loop_steer = env_bool("OPEN_LOOP_STEER", False)
    open_loop_amp_deg = env_float("OPEN_LOOP_STEER_AMP_DEG", 80.0)
    open_loop_period_s = env_float("OPEN_LOOP_STEER_PERIOD_S", 2.0)
    open_loop_speed_kph = env_float("OPEN_LOOP_SPEED_KPH", 80.0)
    open_loop_speed_mps = float(open_loop_speed_kph) / 3.6
    mode_tag = controller.distribution_mode
    if force_zero_steer:
        mode_tag = f"{mode_tag}_zsteer"
    if scenario_name:
        log_filename = f"{scenario_name}_{mode_tag}_telemetry_ep{episode_index:03d}.csv"
    else:
        log_filename = f"{mode_tag}_telemetry_ep{episode_index:03d}.csv"
    log_path = log_output_dir / log_filename
    telemetry_logger = TelemetryLogger(log_path)
    obs_dim = env_int("CM_OBS_DIM", DEFAULT_CM_OBS_DIM)

    try:
        client_sock, _ = await loop.run_in_executor(None, server_sock.accept)
        # Receive initial state:
        # 21 doubles + optional ax, ay (23 doubles total)
        initial = await recv_exact(loop, client_sock, obs_dim * 8)
        if len(initial) < obs_dim * 8:
            raise RuntimeError("Failed to receive initial observation.")

        (
            veh_x,
            veh_y,
            veh_yaw,
            veh_v,
            veh_ax,
            veh_ay,
            rotv,
            long_slip,
            lat_slip,
            aero_drag,
            roll_resist,
        ) = unpack_observation(initial, obs_dim)
        step = 0
        prev_yaw = veh_yaw
        yaw_rate_est = 0.0
        yaw_lpf_alpha = float(env_float("YAW_LPF_ALPHA", 0.2))
        yaw_lpf_alpha = float(np.clip(yaw_lpf_alpha, 0.0, 1.0))
        filt_yaw_rate = 0.0
        filt_ay = 0.0
        beta_est = 0.0
        beta_yaw_gain = float(env_float("BETA_YAWRATE_GAIN", 1.0))
        yaw_ref_beta_blend = float(env_float("YAW_REF_BETA_BLEND", 1.0))
        yaw_ref_beta_blend = float(np.clip(yaw_ref_beta_blend, 0.0, 1.0))
        beta_vx_min = float(env_float("BETA_VX_MIN", 0.5))
        torque_lpf_alpha = float(env_float("TORQUE_LPF_ALPHA", 0.2))
        torque_lpf_alpha = float(np.clip(torque_lpf_alpha, 0.0, 1.0))
        filt_wheel_torques = np.zeros(4, dtype=np.float64)

        while True:
            step += 1
            if step > max_steps:
                print(f"[INFO] Reached MAX_STEPS={max_steps}. Ending episode.")
                break
            sim_time = step * control_dt
            
            ref_state = None
            if scenario.data is not None:
                ref_state = scenario.get_reference_state(sim_time, veh_x=veh_x, veh_y=veh_y)

            # Compute longitudinal total torque from PID
            if open_loop_steer:
                target_speed = open_loop_speed_mps
            else:
                target_speed = ref_state["speed"] if ref_state is not None else 10.0
            v_diff = target_speed - veh_v
            total_torque = controller.compute_total_torque(v_diff=v_diff, dt=control_dt)

            # Compute lateral steering from Stanley unless forced to zero.
            steering_cmd = 0.0
            ref_yaw_rate_base = 0.0
            ref_yaw_rate = 0.0
            ref_yaw_rate_dot = 0.0
            if open_loop_steer:
                open_loop_start_s = float(env_float("OPEN_LOOP_STEER_START_S", 30.0))
                open_loop_cycles = int(env_float("OPEN_LOOP_STEER_CYCLES", 1.0))
                open_loop_cycles = max(open_loop_cycles, 0)
                if abs(open_loop_period_s) > 1e-6:
                    omega = (2.0 * np.pi) / open_loop_period_s
                else:
                    omega = 0.0
                steer_amp = float(np.deg2rad(open_loop_amp_deg))
                phase_time = sim_time - open_loop_start_s
                total_window = max(open_loop_period_s, 0.0) * float(open_loop_cycles)
                if 0.0 <= phase_time <= total_window:
                    steering_cmd = steer_amp * float(np.sin(omega * phase_time))
                    ref_yaw_rate_base = yaw_moment_controller.compute_reference_yaw_rate(
                        veh_speed=veh_v,
                        steering_cmd=steering_cmd,
                    )
            elif (not force_zero_steer) and (ref_state is not None):
                steering_cmd, _, _ = stanley_controller.compute(
                    veh_x=veh_x, veh_y=veh_y, veh_yaw=veh_yaw,
                    ref_x=ref_state['x'], ref_y=ref_state['y'], ref_yaw=ref_state['yaw'],
                    curvature=ref_state['curvature'],
                    vel=veh_v
                )
                ref_yaw_rate_base = yaw_moment_controller.compute_reference_yaw_rate(
                    veh_speed=veh_v,
                    steering_cmd=steering_cmd,
                )

            ay_meas = float(veh_ay)
            filt_yaw_rate = (1.0 - yaw_lpf_alpha) * filt_yaw_rate + yaw_lpf_alpha * yaw_rate_est
            filt_ay = (1.0 - yaw_lpf_alpha) * filt_ay + yaw_lpf_alpha * ay_meas

            ref_yaw_rate_beta = 0.0
            if abs(veh_v) >= beta_vx_min:
                beta_dot = (filt_ay / max(abs(veh_v), 1e-6)) - filt_yaw_rate
                beta_est = float(np.clip(beta_est + beta_dot * control_dt, controller.beta_min, controller.beta_max))
                ref_yaw_rate_beta = (filt_ay / max(abs(veh_v), 1e-6)) + beta_yaw_gain * beta_est
            else:
                beta_est = 0.0

            if controller.distribution_mode == "algo2":
                ref_yaw_rate = ref_yaw_rate_base
            else:
                ref_yaw_rate = (yaw_ref_beta_blend * ref_yaw_rate_base) + ((1.0 - yaw_ref_beta_blend) * ref_yaw_rate_beta)
            ref_yaw_rate_dot = (ref_yaw_rate - yaw_moment_controller.state.prev_yawrate_ref) / max(control_dt, 1e-6)
            yaw_moment_controller.state.prev_yawrate_ref = ref_yaw_rate

            # Compute required yaw moment with separate controller function.
            req_yaw_moment_nm = yaw_moment_controller.compute_required_yaw_moment(
                ay_meas=filt_ay,
                veh_yaw_rate=filt_yaw_rate,
                yawrate_ref=ref_yaw_rate,
                yawrate_ref_dot=ref_yaw_rate_dot,
                dt=control_dt,
            )
            if controller.distribution_mode == "algo0":
                req_yaw_moment_nm = 0.0
            fy_front = float(yaw_moment_controller.state.x_hat[1])
            fy_rear = float(yaw_moment_controller.state.x_hat[2])

            # Distribute total torque to 4 wheels using selected yaw-moment-aware algorithm.
            wheel_torques = controller.distribute_torque(
                total_torque=total_torque,
                req_yaw_moment=req_yaw_moment_nm,
                veh_speed=veh_v,
                veh_ax=veh_ax,
                veh_ay=veh_ay,
                yaw_rate=yaw_rate_est,
                ref_yaw_rate=ref_yaw_rate,
                steering_cmd=steering_cmd,
                dt=control_dt,
                rotv=rotv,
                motor_map=motor_map,
                fy_front=fy_front,
                fy_rear=fy_rear,
            )
            algo2_lpf_enable = env_bool("ALGO2_LPF_ENABLE", True)
            algo3_lpf_enable = env_bool("ALGO3_LPF_ENABLE", True)
            if (controller.distribution_mode == "algo2" and algo2_lpf_enable) or (
                controller.distribution_mode == "algo3" and algo3_lpf_enable
            ) or (
                controller.distribution_mode in {"algo0", "algo1"} and env_bool("ALGO1_LPF_ENABLE", True)
            ):
                filt_wheel_torques = (
                    (1.0 - torque_lpf_alpha) * filt_wheel_torques
                    + torque_lpf_alpha * np.asarray(wheel_torques, dtype=np.float64)
                )
                wheel_torques = filt_wheel_torques
            
            # Saturate drive/regen torque by rpm-dependent max torque and compute battery powers.
            req_drive_total_nm = 0.0
            sat_drive_total_nm = 0.0
            req_regen_total_nm = 0.0
            sat_regen_total_nm = 0.0
            batt_drive_power_w = 0.0
            batt_regen_power_w = 0.0

            for w in range(4):
                omega = abs(float(rotv[w]))
                rpm = omega * 60.0 / (2.0 * np.pi)
                tq_cmd = float(wheel_torques[w])

                req_drive = max(tq_cmd, 0.0)
                req_regen = max(-tq_cmd, 0.0)
                tq_max = motor_map.max_torque_nm(rpm)

                sat_drive = min(req_drive, tq_max)
                sat_regen = min(req_regen, tq_max)

                req_drive_total_nm += req_drive
                sat_drive_total_nm += sat_drive
                req_regen_total_nm += req_regen
                sat_regen_total_nm += sat_regen

                if sat_drive > 1e-9:
                    eff_drive = max(motor_map.efficiency_pct(rpm, sat_drive) / 100.0, 1e-3)
                    batt_drive_power_w += (sat_drive * omega) / eff_drive

                if sat_regen > 1e-9:
                    eff_regen = max(motor_map.efficiency_pct(rpm, sat_regen) / 100.0, 1e-3)
                    batt_regen_power_w += (sat_regen * omega) * eff_regen

            batt_net_power_w = batt_drive_power_w - batt_regen_power_w
            
            # Pack action: [torque_FL, FR, RL, RR, steering_angle]
            # User.c expects physical wheel torques (Nm), not normalized command.
            payload = struct.pack(
                "ddddd",
                float(wheel_torques[0]),
                float(wheel_torques[1]),
                float(wheel_torques[2]),
                float(wheel_torques[3]),
                float(steering_cmd),
                # float(5),
            )
            await loop.run_in_executor(None, client_sock.sendall, payload)

            # Receive next state: obs_dim doubles
            raw = await recv_exact(loop, client_sock, obs_dim * 8)
            if len(raw) < obs_dim * 8:
                print("[WARN] Short packet received from CarMaker. Ending episode.")
                break

            (
                veh_x,
                veh_y,
                veh_yaw,
                veh_v,
                veh_ax,
                veh_ay,
                rotv,
                long_slip,
                lat_slip,
                aero_drag,
                roll_resist,
            ) = unpack_observation(raw, obs_dim)
            sim_state = 8.0  # Assume still running if we received data

            # Wrap-around safe yaw rate from consecutive yaw samples.
            yaw_delta = np.arctan2(np.sin(veh_yaw - prev_yaw), np.cos(veh_yaw - prev_yaw))
            yaw_rate = float(yaw_delta / max(control_dt, 1e-6))
            prev_yaw = veh_yaw
            yaw_rate_est = yaw_rate

            telemetry_logger.write_row(
                [
                    float(sim_time),
                    float(veh_x),
                    float(veh_y),
                    float(veh_yaw),
                    yaw_rate,
                    float(veh_v),
                    float(veh_ax),
                    float(veh_ay),
                    float(rotv[0]),
                    float(rotv[1]),
                    float(rotv[2]),
                    float(rotv[3]),
                    float(long_slip[0]),
                    float(long_slip[1]),
                    float(long_slip[2]),
                    float(long_slip[3]),
                    float(lat_slip[0]),
                    float(lat_slip[1]),
                    float(lat_slip[2]),
                    float(lat_slip[3]),
                    aero_drag,
                    float(roll_resist[0]),
                    float(roll_resist[1]),
                    float(roll_resist[2]),
                    float(roll_resist[3]),
                    req_drive_total_nm,
                    sat_drive_total_nm,
                    req_regen_total_nm,
                    sat_regen_total_nm,
                    batt_drive_power_w,
                    batt_regen_power_w,
                    batt_net_power_w,
                    ref_yaw_rate_base,
                    ref_yaw_rate,
                    float(wheel_torques[0]),
                    float(wheel_torques[1]),
                    float(wheel_torques[2]),
                    float(wheel_torques[3]),
                ]
            )

            # if step % max(log_every, 1) == 0:
            #     print(
            #         f"[STEP {step:05d}] pos=({veh_x:7.2f}, {veh_y:7.2f}) yaw={veh_yaw:6.3f} v={veh_v:6.2f} "
            #         f"| r_ref={ref_yaw_rate:6.3f} r_dot_ref={ref_yaw_rate_dot:6.3f} reqMz={req_yaw_moment_nm:7.1f}Nm mode={controller.distribution_mode} "
            #         f"| rotv(FL,FR,RL,RR)=({rotv[0]:6.1f}, {rotv[1]:6.1f}, {rotv[2]:6.1f}, {rotv[3]:6.1f})"
            #     )

        print(f"[EPISODE END] steps={step}, final_pos=({veh_x:.2f}, {veh_y:.2f}), final_v={veh_v:.3f}")
        print(f"[LOG] {log_path}")

    finally:
        telemetry_logger.close()
        if client_sock:
            client_sock.close()

        if simcontrol.get_status() != cmapi.SimControlState.configure:
            await simcontrol.stop_sim()
            await simcontrol.create_simstate_condition(cmapi.ConditionSimState.idle).wait()

        await simcontrol.disconnect()


async def main() -> None:
    run_mode = os.getenv("RUN_MODE", RUN_MODE_GUI).upper()
    project_path = Path(os.getenv("CM_PROJECT_PATH", DEFAULT_PROJECT_PATH))
    testrun_name = os.getenv("TESTRUN_NAME", DEFAULT_TESTRUN_NAME)
    scenario_csv_path = Path(os.getenv("SCENARIO_CSV_PATH", DEFAULT_SCENARIO_CSV_PATH)) if DEFAULT_SCENARIO_CSV_PATH else None

    port = env_int("CM_PORT", DEFAULT_PORT)
    episodes = env_int("EPISODES", DEFAULT_EPISODES)
    control_dt = env_float("CONTROL_DT", DEFAULT_CONTROL_DT)
    cycle_name = infer_cycle_from_scenario_path(scenario_csv_path)
    cycle_default_max_steps = DEFAULT_MAX_STEPS_BY_CYCLE.get(cycle_name, DEFAULT_MAX_STEPS)
    if os.getenv("MAX_STEPS") is not None:
        max_steps = env_int("MAX_STEPS", cycle_default_max_steps)
    else:
        max_steps = cycle_default_max_steps
    realtime_factor = env_float("REALTIME_FACTOR", DEFAULT_REALTIME_FACTOR)
    log_every = env_int("LOG_EVERY", DEFAULT_LOG_EVERY)
    log_output_dir = Path(DEFAULT_LOG_OUTPUT_DIR)

    print(
        "[CONFIG] "
        f"RUN_MODE={run_mode} PROJECT={project_path} TESTRUN={testrun_name} "
        f"PORT={port} EPISODES={episodes} CONTROL_DT={control_dt} "
        f"CYCLE={cycle_name} MAX_STEPS={max_steps}"
    )
    
    # 시나리오 CSV 로드
    scenario = Scenario()
    if scenario_csv_path and scenario_csv_path.exists():
        scenario.load_csv(scenario_csv_path)
    else:
        print("[WARN] Scenario CSV path not configured. Running without scenario tracking.")

    cmapi.Project.load(project_path)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((DEFAULT_HOST, port))
    server_sock.listen(1)

    master = build_master(run_mode, project_path)

    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(master)

    testrun = cmapi.Project.instance().load_testrun_parametrization(project_path / f"Data/TestRun/{testrun_name}")
    variation = cmapi.Variation.create_from_testrun(testrun)

    controller = PIDTorqueController()
    motor_map = MotorMap(Path(DEFAULT_MG_MAP_PATH))
    if not motor_map.load():
        raise RuntimeError("Motor map load failed; cannot run without it.")
    
    # Extract scenario name from path (e.g., "simulation_results_HWFET_ay9p0.csv" -> "HWFET_ay9p0")
    scenario_name = ""
    if scenario_csv_path:
        name = scenario_csv_path.stem  # Remove .csv extension
        if name.startswith("simulation_results_"):
            scenario_name = name.replace("simulation_results_", "")
        else:
            scenario_name = name
    
    loop = asyncio.get_running_loop()

    try:
        for ep in range(1, episodes + 1):
            print(f"\n===== EPISODE {ep}/{episodes} =====")
            await run_one_episode(
                loop=loop,
                simcontrol=simcontrol,
                variation=variation,
                server_sock=server_sock,
                controller=controller,
                motor_map=motor_map,
                scenario=scenario,
                log_output_dir=log_output_dir,
                episode_index=ep,
                control_dt=control_dt,
                max_steps=max_steps,
                realtime_factor=realtime_factor,
                log_every=log_every,
                scenario_name=scenario_name,
            )
    finally:
        server_sock.close()

if __name__ == "__main__":
    cmapi.Task.run_main_task(main())

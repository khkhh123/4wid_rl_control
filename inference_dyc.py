"""
CarMaker GUI + 학습된 PPO 모델 inference 스크립트.

사용법:
    python3 inference_dyc.py [model_path]

환경변수:
    MODEL_PATH          : 모델 경로 (기본값: carmaker_ppo_4wid_dyc_best.zip)
    CM_PROJECT_PATH     : CarMaker 프로젝트 경로
    TESTRUN_NAME        : TestRun 이름 (기본값: test1)
    CM_PORT             : CarMaker 소켓 포트 (기본값: 5555)
    REALTIME_FACTOR     : 실시간 배율 (기본값: 1.0)
    EPISODE_MAX_STEPS   : 에피소드 최대 스텝 (기본값: 50000)
    TELEMETRY_DIR       : CSV 저장 디렉토리 (기본값: telemetry)
"""
import os
import sys
import asyncio
import socket
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi

from train_with_cm_gui import (
    CarMaker4WIDEnv,
    SyncBridge,
    recv_exact,
    carmaker_orchestrator,
    get_carmaker_pid,
    MODEL_BASENAME,
    SCENARIO_DIR,
    SCENARIO_CSV_PATH,
    OPEN_LOOP_STEER,
    OPEN_LOOP_STEER_AMP_DEG,
    OPEN_LOOP_STEER_PERIOD_S,
    OPEN_LOOP_STEER_START_S,
    OPEN_LOOP_STEER_CYCLES,
    OPEN_LOOP_SPEED_MODE,
    OPEN_LOOP_SPEED_KPH,
    OPEN_LOOP_SPEED_MIN_KPH,
    OPEN_LOOP_SPEED_MAX_KPH,
)

PROJECT_DIR = Path(__file__).resolve().parent

# 균등 편차 기준 고정 액션: 좌우 편차 0(균형), 좌/우 전후 바이어스 0(각각 50/50)
# left_total_bias in [-1, 1]:
#   left_total  = demand_trq/2 + left_total_bias*(WHEEL_TORQUE_LIMIT*2)
#   right_total = demand_trq/2 - left_total_bias*(WHEEL_TORQUE_LIMIT*2)
# left_fr_bias/right_fr_bias in [-1, 1]: -1=rear 100%, +1=front 100%, 0=50/50
UNIFORM_ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)

CSV_COLUMNS = [
    "sim_time",
    "veh_x", "veh_y", "veh_yaw", "veh_yaw_rate", "veh_speed", "veh_ax", "veh_ay",
    "rotv_fl", "rotv_fr", "rotv_rl", "rotv_rr",
    "long_slip_loss_fl", "long_slip_loss_fr", "long_slip_loss_rl", "long_slip_loss_rr",
    "lat_slip_loss_fl", "lat_slip_loss_fr", "lat_slip_loss_rl", "lat_slip_loss_rr",
    "aero_drag",
    "roll_resist_fl", "roll_resist_fr", "roll_resist_rl", "roll_resist_rr",
    "req_drive_torque_total_nm", "sat_drive_torque_total_nm",
    "req_regen_torque_total_nm", "sat_regen_torque_total_nm",
    "batt_drive_power_w", "batt_regen_power_w", "batt_net_power_w",
    "ref_yaw_rate_base", "ref_yaw_rate",
    "torque_fl", "torque_fr", "torque_rl", "torque_rr",
]


def build_csv_row(env_async: CarMaker4WIDEnv, sim_time: float) -> dict:
    """env_async의 last_full_state + last_action_info로 CSV 한 행을 구성."""
    s = env_async.last_full_state  # 23-tuple
    ai = env_async.last_action_info

    # state 매핑: [x, y, yaw, speed, rotv×4, long_slip×4, lat_slip×4, aero_drag, roll_resist×4, ax, ay]
    total_torque = ai.get("total_torque", 0.0)
    tfl = ai.get("torque_fl", 0.0)
    tfr = ai.get("torque_fr", 0.0)
    trl = ai.get("torque_rl", 0.0)
    trr = ai.get("torque_rr", 0.0)

    req_drive = max(0.0, total_torque)
    req_regen = abs(min(0.0, total_torque))
    sat_drive = sum(t for t in [tfl, tfr, trl, trr] if t > 0)
    sat_regen = sum(abs(t) for t in [tfl, tfr, trl, trr] if t < 0)

    return {
        "sim_time": sim_time,
        "veh_x": s[0], "veh_y": s[1], "veh_yaw": s[2],
        "veh_yaw_rate": env_async.last_obs[1] if env_async.last_obs is not None else 0.0,
        "veh_speed": s[3],
        "veh_ax": s[21], "veh_ay": s[22],
        "rotv_fl": s[4], "rotv_fr": s[5], "rotv_rl": s[6], "rotv_rr": s[7],
        "long_slip_loss_fl": s[8], "long_slip_loss_fr": s[9],
        "long_slip_loss_rl": s[10], "long_slip_loss_rr": s[11],
        "lat_slip_loss_fl": s[12], "lat_slip_loss_fr": s[13],
        "lat_slip_loss_rl": s[14], "lat_slip_loss_rr": s[15],
        "aero_drag": s[16],
        "roll_resist_fl": s[17], "roll_resist_fr": s[18],
        "roll_resist_rl": s[19], "roll_resist_rr": s[20],
        "req_drive_torque_total_nm": req_drive,
        "sat_drive_torque_total_nm": sat_drive,
        "req_regen_torque_total_nm": req_regen,
        "sat_regen_torque_total_nm": sat_regen,
        "batt_drive_power_w": 0.0,
        "batt_regen_power_w": 0.0,
        "batt_net_power_w": 0.0,
        "ref_yaw_rate_base": ai.get("ref_yaw_rate_base", 0.0),
        "ref_yaw_rate": ai.get("ref_yaw_rate", 0.0),
        "torque_fl": tfl, "torque_fr": tfr, "torque_rl": trl, "torque_rr": trr,
    }


async def main():
    # ── 모델 경로 결정 ──────────────────────────────────────────────
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv(
        "MODEL_PATH", str(PROJECT_DIR / f"{MODEL_BASENAME}_best.zip")
    )
    if not os.path.exists(model_path):
        # best 없으면 일반 모델 시도
        fallback = str(PROJECT_DIR / f"{MODEL_BASENAME}.zip")
        if os.path.exists(fallback):
            print(f"[INFERENCE] best 모델 없음. 대체: {fallback}")
            model_path = fallback
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    algo = os.getenv("ALGO", "ppo").strip().lower()
    if algo not in ("ppo", "sac"):
        raise ValueError(f"ALGO는 'ppo' 또는 'sac' 이어야 합니다. 받은 값: {algo}")

    model_class = PPO if algo == "ppo" else SAC
    print(f"[INFERENCE] ALGO={algo.upper()} | 모델 로드: {model_path}")
    try:
        model = model_class.load(model_path, device="cpu")
    except Exception as e:
        alt_class = SAC if model_class is PPO else PPO
        print(f"[WARN] {model_class.__name__}.load 실패 ({e}). {alt_class.__name__}로 재시도합니다.")
        model = alt_class.load(model_path, device="cpu")

    # ── CarMaker 연결 ───────────────────────────────────────────────
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
    testrun = cmapi.Project.instance().load_testrun_parametrization(
        proj_path / f"Data/TestRun/{testrun_name}"
    )
    variation = cmapi.Variation.create_from_testrun(testrun)

    loop = asyncio.get_running_loop()
    env_async = CarMaker4WIDEnv(loop)
    env_async.max_steps = int(os.getenv("EPISODE_MAX_STEPS", "50000"))
    use_open_loop = bool(env_async.use_open_loop_steer)

    telemetry_dir = Path(os.getenv("TELEMETRY_DIR", str(PROJECT_DIR / "telemetry")))
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(model_path).stem

    control_mode = os.getenv("CONTROL_MODE", "ppo").strip().lower()
    if control_mode not in ("ppo", "uniform"):
        raise ValueError(f"CONTROL_MODE는 'ppo' 또는 'uniform'이어야 합니다. 받은 값: {control_mode}")

    # uniform 모드에서는 모델 태그를 'uniform'으로 고정
    if control_mode == "uniform":
        model_tag = "uniform"

    # ── 시나리오 파일 오버라이드 ────────────────────────────────────
    # CLI: python3 inference_dyc.py [model_path] [csv1] [csv2] ...
    scenario_overrides = sys.argv[2:]
    if use_open_loop:
        print(
            "[INFERENCE] OPEN_LOOP_STEER=1. "
            f"Sine steering amp={OPEN_LOOP_STEER_AMP_DEG:.1f}deg "
            f"period={OPEN_LOOP_STEER_PERIOD_S:.2f}s start={OPEN_LOOP_STEER_START_S:.2f}s cycles={OPEN_LOOP_STEER_CYCLES}."
        )
        print(
            "[INFERENCE] OPEN_LOOP speed profile: "
            f"mode={OPEN_LOOP_SPEED_MODE} const={OPEN_LOOP_SPEED_KPH:.1f}kph "
            f"min={OPEN_LOOP_SPEED_MIN_KPH:.1f}kph max={OPEN_LOOP_SPEED_MAX_KPH:.1f}kph"
        )
        if scenario_overrides or os.getenv("SCENARIO_CSVS"):
            print("[INFERENCE] OPEN_LOOP_STEER=1 이므로 시나리오 override는 무시합니다.")
    else:
        if scenario_overrides:
            env_async.scenario_files = [Path(p) for p in scenario_overrides]
            env_async.scenario_idx = -1
            print(f"[INFERENCE] CLI 시나리오 {len(scenario_overrides)}개: "
                  f"{[Path(p).name for p in scenario_overrides]}")
        elif os.getenv("SCENARIO_CSVS"):
            paths = [p.strip() for p in
                     os.getenv("SCENARIO_CSVS").replace(":", ",").split(",") if p.strip()]
            env_async.scenario_files = [Path(p) for p in paths]
            env_async.scenario_idx = -1
            print(f"[INFERENCE] SCENARIO_CSVS {len(paths)}개: {[Path(p).name for p in paths]}")

    print(
        f"[INFERENCE] CM_PORT={cm_port} | TESTRUN={testrun_name} | PID={pid}"
    )
    print(
        f"[INFERENCE] SCENARIO_DIR={SCENARIO_DIR} | REALTIME_FACTOR={os.getenv('REALTIME_FACTOR', '1.0')}"
    )
    print(f"[INFERENCE] CONTROL_MODE={control_mode} | Telemetry 저장 경로: {telemetry_dir}")

    # 오케스트레이터를 백그라운드 태스크로 실행
    cmapi.Task.run_task_bg(carmaker_orchestrator(env_async, simcontrol, variation, server_sock))

    env_sync = SyncBridge(env_async, loop)

    # ── inference 루프 ──────────────────────────────────────────────
    await loop.run_in_executor(
        None, run_inference, model, env_sync, env_async, telemetry_dir, model_tag, control_mode
    )

    env_async.stop_req.set()
    loop.stop()


def run_inference(model: PPO, env: SyncBridge, env_async: CarMaker4WIDEnv,
                  telemetry_dir: Path, model_tag: str, mode: str = "ppo"):
    """
    mode='ppo'    : PPO 모델로 액션 예측 (deterministic)
    mode='uniform': 고정 균등 편차 액션 [0.0, 0.0, 0.0] 사용 (비교용 baseline)
    """
    episode = 0
    try:
        while True:
            episode += 1
            obs, _ = env.reset()
            episode_reward = 0.0
            step = 0
            control_dt = env_async.control_dt

            # 시나리오 이름 추출 (CSV 파일명용)
            scenario_tag = "openloop_sine" if env_async.use_open_loop_steer else "unknown"
            if (not env_async.use_open_loop_steer) and (env_async.current_scenario_path is not None):
                scenario_tag = Path(env_async.current_scenario_path).stem

            csv_path = telemetry_dir / f"{scenario_tag}_{model_tag}_telemetry_ep{episode:03d}.csv"
            print(f"\n[EPISODE {episode}] 시작 | 모드={mode} | 시나리오={scenario_tag} | CSV={csv_path.name}")

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

                while True:
                    if mode == "uniform":
                        action = UNIFORM_ACTION.copy()
                    else:
                        action, _ = model.predict(obs, deterministic=True)

                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    episode_reward += reward
                    sim_time = step * control_dt

                    if env_async.last_full_state is not None and env_async.last_action_info:
                        row = build_csv_row(env_async, sim_time)
                        writer.writerow(row)

                    if terminated or truncated:
                        print(
                            f"[EPISODE {episode}] 종료 | steps={step} "
                            f"| reward={episode_reward:.2f} | CSV={csv_path}"
                        )
                        break

    except KeyboardInterrupt:
        print("\n[INFERENCE] 중단 요청됨.")
    finally:
        print("[INFERENCE] 종료.")


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())

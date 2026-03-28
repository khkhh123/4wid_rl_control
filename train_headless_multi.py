import os
import sys
import subprocess
from pathlib import Path

from experiment_modes import resolve_num_workers


def build_worker_env(worker_id: int):
    env = os.environ.copy()
    env.setdefault("CM_RUN_MODE", "headless")
    env["ENV_ID"] = str(worker_id)
    env.setdefault("CM_PORT_BASE", "5555")
    env.setdefault("NUM_WORKERS", env.get("NUM_WORKERS", "1"))
    return env


def main():
    num_workers = resolve_num_workers(default_value=max(1, (os.cpu_count() or 2) // 2))
    profile = os.getenv("PPO_PROFILE") or (sys.argv[1] if len(sys.argv) > 1 else "B")

    project_dir = Path(__file__).resolve().parent
    train_script = project_dir / "train_with_cm_gui.py"

    print(f"[MULTI] Launching {num_workers} headless workers with PPO_PROFILE={profile}")
    procs = []

    for worker_id in range(num_workers):
        env = build_worker_env(worker_id)
        cmd = [sys.executable, str(train_script), profile]
        print(
            f"[WORKER-{worker_id}] CM_PORT={int(env.get('CM_PORT_BASE', '5555')) + worker_id}"
            f" | ENV_ID={worker_id}"
        )
        procs.append(subprocess.Popen(cmd, cwd=str(project_dir), env=env))

    exit_codes = []
    try:
        for p in procs:
            exit_codes.append(p.wait())
    except KeyboardInterrupt:
        print("[MULTI] KeyboardInterrupt: terminating workers...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            if p.poll() is None:
                p.wait(timeout=10)

    non_zero = [c for c in exit_codes if c != 0]
    if non_zero:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

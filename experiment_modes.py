import os


ACTION_MODE_RATIO3 = "ratio3"
ACTION_MODE_ACTION4 = "action4"

RUN_MODE_GUI = "gui"
RUN_MODE_HEADLESS = "headless"

PROTOCOL_MODE_ASSESSMENT23 = "assessment23"

REWARD_MODE_EFFORT_ONLY = "effort_only"
REWARD_MODE_VELOCITY_EFFORT = "velocity_effort"


def resolve_action_mode() -> str:
    mode = os.getenv("ACTION_MODE", ACTION_MODE_RATIO3).strip().lower()
    if mode not in {ACTION_MODE_RATIO3, ACTION_MODE_ACTION4}:
        print(f"[WARN] Unknown ACTION_MODE='{mode}'. Fallback to '{ACTION_MODE_RATIO3}'.")
        return ACTION_MODE_RATIO3
    return mode


def default_reward_mode(action_mode: str) -> str:
    if action_mode == ACTION_MODE_ACTION4:
        return REWARD_MODE_VELOCITY_EFFORT
    return REWARD_MODE_EFFORT_ONLY


def resolve_reward_mode(action_mode: str) -> str:
    reward_mode = os.getenv("REWARD_MODE", default_reward_mode(action_mode)).strip().lower()
    if reward_mode not in {REWARD_MODE_EFFORT_ONLY, REWARD_MODE_VELOCITY_EFFORT}:
        fallback = default_reward_mode(action_mode)
        print(f"[WARN] Unknown REWARD_MODE='{reward_mode}'. Fallback to '{fallback}'.")
        return fallback
    return reward_mode


def default_model_basename(action_mode: str) -> str:
    algo = os.getenv("ALGO", "sac").strip().lower()
    if action_mode == ACTION_MODE_ACTION4:
        return f"carmaker_{algo}_4wid_action4_dyc"
    return f"carmaker_{algo}_4wid_dyc"


def resolve_model_basename(action_mode: str) -> str:
    return os.getenv("MODEL_BASENAME", default_model_basename(action_mode)).strip()


def default_tensorboard_dir(action_mode: str) -> str:
    algo = os.getenv("ALGO", "sac").strip().lower()
    if action_mode == ACTION_MODE_ACTION4:
        return f"{algo}_carmaker_action4_tensorboard"
    return f"{algo}_carmaker_lefttorque_ratio3_tensorboard"


def resolve_tensorboard_dir(action_mode: str) -> str:
    return os.getenv("TB_DIRNAME", default_tensorboard_dir(action_mode)).strip()


def resolve_env_id() -> int:
    raw = os.getenv("ENV_ID", "0").strip()
    try:
        env_id = int(raw)
    except ValueError:
        print(f"[WARN] Invalid ENV_ID='{raw}'. Fallback to 0.")
        return 0
    if env_id < 0:
        print(f"[WARN] ENV_ID must be >= 0. Fallback to 0.")
        return 0
    return env_id


def resolve_cm_port() -> int:
    explicit = os.getenv("CM_PORT")
    if explicit is not None and explicit.strip() != "":
        try:
            return int(explicit.strip())
        except ValueError:
            print(f"[WARN] Invalid CM_PORT='{explicit}'. Using CM_PORT_BASE + ENV_ID.")

    base_raw = os.getenv("CM_PORT_BASE", "5555").strip()
    try:
        base = int(base_raw)
    except ValueError:
        print(f"[WARN] Invalid CM_PORT_BASE='{base_raw}'. Fallback to 5555.")
        base = 5555

    return base + resolve_env_id()


def resolve_run_mode() -> str:
    mode = os.getenv("CM_RUN_MODE", RUN_MODE_GUI).strip().lower()
    if mode not in {RUN_MODE_GUI, RUN_MODE_HEADLESS}:
        print(f"[WARN] Unknown CM_RUN_MODE='{mode}'. Fallback to '{RUN_MODE_GUI}'.")
        return RUN_MODE_GUI
    return mode


def resolve_protocol_mode() -> str:
    mode = os.getenv("CM_PROTOCOL_MODE", PROTOCOL_MODE_ASSESSMENT23).strip().lower()
    if mode != PROTOCOL_MODE_ASSESSMENT23:
        print(
            f"[WARN] CM_PROTOCOL_MODE='{mode}' is ignored. "
            f"Protocol is fixed to '{PROTOCOL_MODE_ASSESSMENT23}'."
        )
    return PROTOCOL_MODE_ASSESSMENT23


def resolve_control_dt(default_value: float = 0.05) -> float:
    raw = os.getenv("CONTROL_DT", str(default_value)).strip()
    try:
        value = float(raw)
    except ValueError:
        print(f"[WARN] Invalid CONTROL_DT='{raw}'. Fallback to {default_value}.")
        return default_value
    if value <= 0.0:
        print(f"[WARN] CONTROL_DT must be > 0. Fallback to {default_value}.")
        return default_value
    return value


def resolve_num_workers(default_value: int = 1) -> int:
    raw = os.getenv("NUM_WORKERS", str(default_value)).strip()
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid NUM_WORKERS='{raw}'. Fallback to {default_value}.")
        return default_value
    if value < 1:
        print(f"[WARN] NUM_WORKERS must be >= 1. Fallback to {default_value}.")
        return default_value
    return value

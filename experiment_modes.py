import os


ACTION_MODE_RATIO3 = "ratio3"
ACTION_MODE_ACTION4 = "action4"

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
    if action_mode == ACTION_MODE_ACTION4:
        return "carmaker_ppo_4wid_action4"
    return "carmaker_ppo_4wid_lefttorque_ratio3"


def resolve_model_basename(action_mode: str) -> str:
    return os.getenv("MODEL_BASENAME", default_model_basename(action_mode)).strip()


def default_tensorboard_dir(action_mode: str) -> str:
    if action_mode == ACTION_MODE_ACTION4:
        return "ppo_carmaker_action4_tensorboard"
    return "ppo_carmaker_lefttorque_ratio3_tensorboard"


def resolve_tensorboard_dir(action_mode: str) -> str:
    return os.getenv("TB_DIRNAME", default_tensorboard_dir(action_mode)).strip()

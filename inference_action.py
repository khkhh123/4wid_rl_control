import os

from inference import main, cmapi


if __name__ == "__main__":
    os.environ.setdefault("ACTION_MODE", "action4")
    os.environ.setdefault("REWARD_MODE", "velocity_effort")
    os.environ.setdefault("MODEL_BASENAME", "carmaker_ppo_4wid_action4")
    cmapi.Task.run_main_task(main())

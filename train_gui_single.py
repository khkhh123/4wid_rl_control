import os

from train_with_cm_gui import main, cmapi


if __name__ == "__main__":
    os.environ.setdefault("CM_RUN_MODE", "gui")
    os.environ.setdefault("ENV_ID", "0")
    os.environ.setdefault("NUM_WORKERS", "1")
    cmapi.Task.run_main_task(main())

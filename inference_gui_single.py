import os

from inference import main, cmapi


if __name__ == "__main__":
    os.environ.setdefault("CM_RUN_MODE", "gui")
    os.environ.setdefault("ENV_ID", "0")
    cmapi.Task.run_main_task(main())

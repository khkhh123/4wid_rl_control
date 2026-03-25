import sys
import asyncio
import threading
import struct
import numpy as np
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

# CarMaker API 경로
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi
from cmapi import Project, Variation
from carmaker_env import CarMaker4WIDEnv

# 1. 비동기 루프를 별도 스레드에서 실행하기 위한 헬퍼
class AsyncThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

# 2. SB3와 연동할 수 있는 동기식 래퍼
class SyncWrapper(gym.Env):
    def __init__(self, a_env, loop):
        self.a_env = a_env
        self.loop = loop
        self.action_space = a_env.action_space
        self.observation_space = a_env.observation_space

    def reset(self, seed=None, options=None):
        # 다른 스레드에 있는 루프에 reset 작업을 던지고 결과를 기다림
        future = asyncio.run_coroutine_threadsafe(self.a_env.reset(seed=seed, options=options), self.loop)
        return future.result()

    def step(self, action):
        # 다른 스레드에 있는 루프에 step 작업을 던지고 결과를 기다림
        future = asyncio.run_coroutine_threadsafe(self.a_env.step(action), self.loop)
        return future.result()

    def close(self):
        future = asyncio.run_coroutine_threadsafe(self.a_env.close(), self.loop)
        return future.result()

def main():
    # A. 비동기 루프 스레드 시작
    async_thread = AsyncThread()
    async_thread.start()
    loop = async_thread.loop

    # B. 프로젝트 및 환경 설정 (동기식으로 실행 가능하도록 루프에 위임)
    project_path = Path("/home/khkhh/CM_Projects/test1")
    Project.load(project_path)
    
    app = cmapi.CarMaker()
    app.set_executable_path(project_path / "src/CarMaker.linux64")

    testrun_path = project_path / "Data/TestRun/testrun_test1"
    testrun = Project.instance().load_testrun_parametrization(testrun_path)
    variation_kh = Variation.create_from_testrun(testrun)

    # C. 환경 생성 (비동기 루프 스레드 안에서 생성되도록 위임)
    def create_env():
        return CarMaker4WIDEnv(app, variation_kh)
    
    env_async = loop.call_soon_threadsafe(create_env) 
    # 실제로는 동기 래퍼가 필요하므로 아래처럼 감쌉니다.
    env_async = CarMaker4WIDEnv(app, variation_kh)
    env = SyncWrapper(env_async, loop)

    # D. PPO 모델 정의 (CPU 추천)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    print("학습 시작! (Thread-safe 구조)")
    try:
        model.learn(total_timesteps=5000000)
        model.save("carmaker_ppo_4wid")
        print("학습 완료 및 저장 성공")
    except KeyboardInterrupt:
        print("학습 중단 - 저장 중...")
        model.save("carmaker_ppo_interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    # cmapi.Task.run_main_task 대신 일반 main 실행
    main()
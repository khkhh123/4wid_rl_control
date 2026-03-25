import sys
import asyncio
import threading
import struct
import numpy as np
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
import psutil

# CarMaker API 경로
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi
from cmapi import Project, Variation
from cmapi import SimControlInteractive
from cmapi import ApoServer, Application
from carmaker_env import CarMaker4WIDEnv

def get_carmaker_pid():
    # CarMaker GUI의 프로세스 이름 키워드
    target_name = "CarMaker.linux64" 
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 프로세스 이름이나 실행 경로에 키워드가 포함되어 있는지 확인
            if target_name in proc.info['name'] or any(target_name in arg for arg in proc.info['cmdline'] or []):
                print(f"Found CarMaker GUI! PID: {proc.info['pid']}")
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
            
    print("CarMaker GUI is not running.")
    return None

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
    
    # app = cmapi.CarMaker()
    # app.set_executable_path(project_path / "src/CarMaker.linux64")

    PID = get_carmaker_pid()
    # Create a handle to the apo server started with the CarMaker application.
    sinfo = cmapi.ApoServerInfo(pid=PID, description="Idle")
    master = ApoServer()
    master.set_sinfo(sinfo)
    master.set_host("localhost")

    testrun_path = project_path / "Data/TestRun/testrun_test1"
    testrun = Project.instance().load_testrun_parametrization(testrun_path)
    variation_kh = Variation.create_from_testrun(testrun)

    # C. 환경 생성 (비동기 루프 스레드 안에서 생성되도록 위임)
    def create_env():
        return CarMaker4WIDEnv(master, variation_kh)
    
    env_async = loop.call_soon_threadsafe(create_env) 
    # 실제로는 동기 래퍼가 필요하므로 아래처럼 감쌉니다.
    env_async = CarMaker4WIDEnv(master, variation_kh)
    env = SyncWrapper(env_async, loop)

    # D. 저장된 모델 불러오기
    # 파일명이 'carmaker_ppo_4wid.zip'이라고 가정합니다.
    # model_path = "carmaker_ppo_speedtracking"
    model_path = "carmaker_ppo_interrupted1"
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env, device="cpu")

    print("인퍼런스 시뮬레이션 시작!")

    try:
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 학습된 신경망이 현재 관측값(obs)을 보고 액션을 결정
            # deterministic=True: 가장 확률이 높은 최적의 액션만 선택
            action, _states = model.predict(obs, deterministic=True)
            
            # 환경에 액션 적용
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
            
        print(f"시뮬레이션 종료. 총 보상: {total_reward}")

    except KeyboardInterrupt:
        print("사용자에 의해 중단됨")
    finally:
        env.close()

if __name__ == "__main__":
    # cmapi.Task.run_main_task 대신 일반 main 실행
    main()
import sys, asyncio, threading, psutil
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

# CarMaker API 설정
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi
from carmaker_env_simple import CarMaker4WIDEnv

def get_carmaker_pid():
    target = "CarMaker.linux64"
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if target in proc.info['name'] or any(target in arg for arg in (proc.info['cmdline'] or [])):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    return None

# SB3 학습을 별도 스레드에서 실행 (비동기 루프를 방해하지 않기 위해)
def run_learning(env):
    print("학습 시작! (Thread-safe 구조)")
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    try:
        model.learn(total_timesteps=5000000)
        model.save("carmaker_ppo_4wid")
    finally:
        env.close()

class CarMakerSyncBridge(gym.Env):
    """비동기 Env를 SB3용 동기 함수로 변환하는 브릿지"""
    def __init__(self, async_env, loop):
        self.a_env = async_env
        self.loop = loop
        self.action_space = async_env.action_space
        self.observation_space = async_env.observation_space

    def _run_sync(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def reset(self, **kwargs): return self._run_sync(self.a_env.reset(**kwargs))
    def step(self, action):    return self._run_sync(self.a_env.step(action))
    def close(self):           return self._run_sync(self.a_env.close())

async def main():
    # 1. 초기 설정 및 객체 생성
    project_path = Path("/home/khkhh/CM_Projects/test1")
    cmapi.Project.load(project_path)
    
    master = cmapi.ApoServer()
    master.set_sinfo(cmapi.ApoServerInfo(pid=get_carmaker_pid(), description="Idle"))
    master.set_host("localhost")

    testrun = cmapi.Project.instance().load_testrun_parametrization(project_path / "Data/TestRun/testrun_test1")
    variation = cmapi.Variation.create_from_testrun(testrun)

    # 2. 비동기 환경 생성 및 브릿지 연결
    loop = asyncio.get_running_loop()
    async_env = CarMaker4WIDEnv(master, variation, loop) # loop 주입
    sync_env = CarMakerSyncBridge(async_env, loop)
    sync_env = TimeLimit(sync_env, max_episode_steps=10000)
    
    # 3. 학습 스레드 분리 및 대기
    # SB3는 메인 스레드에서 돌리고 싶어하므로, 현재 루프를 백그라운드에 유지한 채 학습 실행
    learn_thread = threading.Thread(target=run_learning, args=(sync_env,), daemon=True)
    learn_thread.start()

    # 4. cmapi 루프가 죽지 않도록 무한 대기 (학습이 끝날 때까지)
    while learn_thread.is_alive():
        await asyncio.sleep(1)

if __name__ == "__main__":
    # cmapi의 공식 권장 실행 방식 (모든 비동기 자원 자동 관리)
    cmapi.Task.run_main_task(main())
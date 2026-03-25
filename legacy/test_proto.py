import asyncio
from pathlib import Path
from carmaker_env import CarMaker4WIDEnv
import sys

# CarMaker API 경로
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")
import cmapi
from cmapi import Project, Variation

async def main():
    # 1. 프로젝트 로드 및 마스터 앱 선언
    project_path = Path("/home/khkhh/CM_Projects/test1")
    Project.load(project_path)
    
    app = cmapi.CarMaker()
    app.set_executable_path(project_path / "src/CarMaker.linux64")

    testrun_path = project_path / "Data/TestRun/testrun_test1"
    testrun = Project.instance().load_testrun_parametrization(testrun_path)
    variation_kh = Variation.create_from_testrun(testrun)

    env = CarMaker4WIDEnv(app, variation_kh)

    print("RL 테스트 시작 (비동기 await 방식)")
    for ep in range(5):
        # [수정] await 추가
        obs, info = await env.reset() 
        done = False
        while not done:
            action = env.action_space.sample()
            # print(action)
            # [수정] await 추가
            obs, reward, terminated, truncated, info = await env.step(action)
            done = terminated or truncated
        print(f"에피소드 {ep+1} 완료")

    # [수정] await 추가
    await env.close()

if __name__ == "__main__":
    cmapi.Task.run_main_task(main())
import sys
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")

from pathlib import Path
import cmapi
from cmapi import Runtime, Project, Variation

import time
import multiprocessing as mp
from functools import partial
import asyncio
import traceback

# * cmapi debuggin log
# cmapi.logger.setLevel("DEBUG")

# * 단위 시뮬레이션 수행 (app과 simcontrol은 공유하는데, 병렬 실행 시 app이 공유되는 게 문제될 것으로 예상)
async def run_sim(app, variation):
    
    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(app)
    await simcontrol.start_and_connect()
    simcontrol.set_variation(variation.clone())
    await simcontrol.start_sim()

    simtime = 0
    simstate = 0
    car_v_max = 0
    while (simstate != 2):
        simtime_raw = await simcontrol.simio.dva_read_async("Time")
        simtime = simtime_raw[0]
        simstate_raw = await simcontrol.simio.dva_read_async("SC.State")
        simstate = simstate_raw[0]

        car_v_temp = await simcontrol.simio.dva_read_async("Car.vx")
        trq_FL = await simcontrol.simio.dva_read_async("PT.WFL.Trq_Drive")
        gas = await simcontrol.simio.dva_read_async("DM.Gas")

        # print("FL trq: ", trq_FL[0])
        # print("velocity: ", car_v_temp[0])
        # print(simstate)
        # print("gas: ", gas[0])
        # print("time: ", simtime)
        car_v_max = max(car_v_max, car_v_temp[0])

    await simcontrol.stop_sim()
    # await simcontrol.stop_and_disconnect()
    print("successfully stopped simcontrol")
    # await app.SessionLog.clear()
    return car_v_max

async def trace_pending_tasks():
    print("\n" + "="*50)
    print("🔍 미완료 태스크 정밀 추적 시작")
    print("="*50)
    
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    for i, task in enumerate(pending):
        print(f"\n[{i+1}] 태스크 정보: {task}")
        # 태스크가 멈춰 있는 지점의 스택 프레임을 가져옵니다.
        frames = task.get_stack()
        if frames:
            print("📍 멈춰 있는 지점 (Stack Trace):")
            for frame in frames:
                # 파일명, 줄 번호, 함수명 등을 상세히 출력
                traceback.print_stack(frame)
        else:
            print("❗ 스택 정보를 가져올 수 없습니다 (내부 C/Cython 영역)")
    
    print("\n" + "="*50)

async def kill_all_tasks():
    # 1. 현재 실행 중인 모든 태스크 가져오기
    current_task = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current_task]
    
    if not tasks:
        return

    print(f"--- {len(tasks)}개의 잔여 태스크 강제 종료 시작 ---")
    
    # 2. 모든 태스크에 취소 신호 전송
    for task in tasks:
        task.cancel()
    
    # 3. 태스크들이 취소 신호를 처리할 시간을 줌 (Gathering)
    # return_exceptions=True를 해야 취소될 때 발생하는 에러로 인해 중단되지 않음
    await asyncio.gather(*tasks, return_exceptions=True)
    print("--- 모든 잔여 태스크 정리 완료 ---")

async def main():

    project_path = Path("/home/khkhh/CM_Projects/test1")
    Project.load(project_path)

    testrun_path = Path("/home/khkhh/CM_Projects/test1/Data/TestRun/testrun_test1")
    testrun = Project.instance().load_testrun_parametrization(testrun_path)     # Load Testrun example
    variation_kh = Variation.create_from_testrun(testrun)                    # Create a variation from testrun

    app = cmapi.Application.create(cmapi.AppType.CarMaker)
    app.set_executable_path(Path("/home/khkhh/CM_Projects/test1/src/CarMaker.linux64"))
    # await app.start()

    start = time.time()
    results = await asyncio.gather(
        run_sim(app, variation_kh),
        run_sim(app, variation_kh),
        run_sim(app, variation_kh),
        run_sim(app, variation_kh),
        run_sim(app, variation_kh),
        run_sim(app, variation_kh),
        run_sim(app, variation_kh),
        run_sim(app, variation_kh)
    )
    end = time.time()
    print(f"실행 시간: {end - start:.5f} 초")

    print(results)    

    cmapi.logger.info("All simulations finished")
    

    await kill_all_tasks()
    # await trace_pending_tasks()
    print("\n--- 남은 태스크 확인 ---")
    pending = asyncio.all_tasks()
    for task in pending:
        print(f"태스크: {task.get_name()}, 상태: {task._state}, 코루틴: {task.get_coro()}")
    print("------------------------\n")
    await app.stop()

if __name__ == "__main__":
    cmapi.Task.run_main_task(main())                                                # Create asynchronous scope
    

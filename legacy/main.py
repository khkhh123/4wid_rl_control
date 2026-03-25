import sys
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")

from pathlib import Path
import cmapi
from cmapi import Runtime, Project, Variation

import time
import multiprocessing as mp

# * cmapi debuggin log
# cmapi.logger.setLevel("DEBUG")

async def main():
    start_time = time.perf_counter()

    app = cmapi.Application.create(cmapi.AppType.CarMaker)
    app.set_executable_path(Path("/home/khkhh/CM_Projects/test1/src/CarMaker.linux64"))

    project_path = Path("/home/khkhh/CM_Projects/test1")
    Project.load(project_path)                                                  # Set Project directory

    await app.start()

    simcontrol = cmapi.SimControlInteractive()

    testrun_path = Path("/home/khkhh/CM_Projects/test1/Data/TestRun/testrun_test1")
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"실행 시간1: {execution_time:.6f} 초")

    for i in range(10):
        
        testrun = Project.instance().load_testrun_parametrization(testrun_path)     # Load Testrun example

        variation = Variation.create_from_testrun(testrun)                    # Create a variation from testrun

        simcontrol.set_variation(variation)

        await simcontrol.set_master(app)    
        await simcontrol.start_and_connect()
        await simcontrol.start_sim()
        start_time = time.perf_counter()
        simtime = 0
        simstate = 0
        # while (simcontrol.get_status() == "SimControlState.configure" or simcontrol.get_status() == "SimControlState.simulating"):
        while (simstate != 2):

            simtime_raw = await simcontrol.simio.dva_read_async("Time")
            simtime = simtime_raw[0]
            simstate_raw = await simcontrol.simio.dva_read_async("SC.State")
            simstate = simstate_raw[0]
            # print(simstate)

            car_v = await simcontrol.simio.dva_read_async("Car.vx")
            trq_FL = await simcontrol.simio.dva_read_async("PT.WFL.Trq_Drive")
            gas = await simcontrol.simio.dva_read_async("DM.Gas")

            # print("FL trq: ", trq_FL[0])
            # print("velocity: ", car_v)
            # print("gas: ", gas[0])
            # print("time: ", simtime)

        await simcontrol.stop_sim()
        simcontrol.clear_storage_buffer()
        # await app.stop()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"실행 시간2: {execution_time:.6f} 초")
    
    start_time = time.perf_counter()
    await simcontrol.stop_and_disconnect()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"실행 시간2: {execution_time:.6f} 초")

    cmapi.logger.info("All simulations finished")

cmapi.Task.run_main_task(main())                                                # Create asynchronous scope

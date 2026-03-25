import sys
sys.path.append("/opt/ipg/carmaker/linux64-14.0.1/Python/python3.10")

from pathlib import Path
import cmapi
from cmapi import Runtime, Project, Variation

import time
import asyncio

# * cmapi debuggin log
# cmapi.logger.setLevel("DEBUG")

async def run_sim(simcontrol, variation):
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
        car_v_max = max(car_v_max, car_v_temp[0])

    await simcontrol.stop_sim()

    # await simcontrol.stop_and_disconnect()
    print("successfully stopped simcontrol")
    return car_v_max

async def main():

    project_path = Path("/home/khkhh/CM_Projects/test1")
    Project.load(project_path)

    testrun_path = Path("/home/khkhh/CM_Projects/test1/Data/TestRun/testrun_test1")
    testrun = Project.instance().load_testrun_parametrization(testrun_path)     # Load Testrun example
    variation_kh = Variation.create_from_testrun(testrun)                    # Create a variation from testrun

    app = cmapi.CarMaker()
    app.set_executable_path(Path("/home/khkhh/CM_Projects/test1/src/CarMaker_875Nm.linux64"))
    simcontrol = cmapi.SimControlInteractive()
    await simcontrol.set_master(app)
    await simcontrol.start_and_connect()
    results1 = await run_sim(simcontrol, variation_kh)
    results2 = await run_sim(simcontrol, variation_kh)
    results3 = await run_sim(simcontrol, variation_kh)
    await simcontrol.disconnect()

    # results = {results1, results2, results3, results4, results5, results6}
    print(results1)
    print(results2)
    print(results3)

    cmapi.logger.info("All simulations finished")

if __name__ == "__main__":
    cmapi.Task.run_main_task(main())                                                # Create asynchronous scope
    

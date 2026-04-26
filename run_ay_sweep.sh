#!/bin/bash
cd /home/khkhh/Projects/4wid_rl_control

declare -A SCRIPTS
SCRIPTS["2p0"]="custom_controller_cm_ay2.py"
SCRIPTS["3p0"]="custom_controller_cm_ay3.py"
SCRIPTS["4p0"]="custom_controller_cm_ay4.py"
SCRIPTS["5p0"]="custom_controller_cm_ay5.py"

ALGOS="algo0 algo1 algo2 algo3"

for AY in 2p0 3p0 4p0 5p0; do
    SCENARIO="scenarios/simulation_results_WLTP_ay${AY}.csv"
    SCRIPT=${SCRIPTS[$AY]}

    for ALGO in $ALGOS; do
        echo "[SWEEP] ${SCRIPT} ${ALGO} ay${AY}..."
        TESTRUN_NAME=WLTP \
        SCENARIO_CSV_PATH=$SCENARIO \
        TORQUE_DISTRIBUTION_MODE=$ALGO \
        python3 $SCRIPT > logs/sweep_${ALGO}_ay${AY}.log 2>&1
        sleep 3
    done

    echo "[SWEEP] RL ay${AY}..."
    ALGO=sac TESTRUN_NAME=WLTP \
    SCENARIO_CSV_PATH=$SCENARIO \
    OPEN_LOOP_STEER=0 CURRICULUM_STAGE=2 NUM_EPISODES=1 \
    python3 inference_dyc.py carmaker_sac_4wid_dyc_C_cl_s2_latest.zip \
    > logs/sweep_rl_ay${AY}.log 2>&1
    sleep 3
done

echo "[SWEEP] All done."

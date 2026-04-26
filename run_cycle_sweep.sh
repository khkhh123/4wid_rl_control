#!/bin/bash
cd /home/khkhh/Projects/4wid_rl_control

declare -A SCRIPTS
SCRIPTS["2p0"]="custom_controller_cm_ay2.py"
SCRIPTS["3p0"]="custom_controller_cm_ay3.py"
SCRIPTS["4p0"]="custom_controller_cm_ay4.py"
SCRIPTS["5p0"]="custom_controller_cm_ay5.py"

ALGOS="algo0 algo1 algo2 algo3"

for CYCLE in FTP HWFET; do
    [ "$CYCLE" = "FTP" ] && CYCLE_FILE="FTP75" || CYCLE_FILE="HWFET"

    for AY in 1p0 2p0 3p0 4p0 5p0; do
        SCENARIO="scenarios/simulation_results_${CYCLE_FILE}_ay${AY}.csv"
        [ "$AY" = "1p0" ] && CTRL_SCRIPT="custom_controller_cm.py" || CTRL_SCRIPT=${SCRIPTS[$AY]}

        for ALGO in $ALGOS; do
            echo "[SWEEP] ${CYCLE} ${ALGO} ay${AY}..."
            TESTRUN_NAME=$CYCLE \
            SCENARIO_CSV_PATH=$SCENARIO \
            TORQUE_DISTRIBUTION_MODE=$ALGO \
            python3 $CTRL_SCRIPT > logs/sweep_${CYCLE}_${ALGO}_ay${AY}.log 2>&1
            sleep 3
        done

        echo "[SWEEP] RL ${CYCLE} ay${AY}..."
        ALGO=sac TESTRUN_NAME=$CYCLE \
        SCENARIO_CSV_PATH=$SCENARIO \
        OPEN_LOOP_STEER=0 CURRICULUM_STAGE=2 NUM_EPISODES=1 \
        python3 inference_dyc.py carmaker_sac_4wid_dyc_C_cl_s2_latest.zip \
        > logs/sweep_rl_${CYCLE}_ay${AY}.log 2>&1
        sleep 3
    done
done

echo "[SWEEP] All done."

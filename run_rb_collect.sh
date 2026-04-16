#!/usr/bin/env bash
# ============================================================
# rb 재수집 스크립트: algo2 + algo3 × 3 시나리오 × 5 ay레벨 = 30회
# 출력: output/rb/
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_BASE="/home/khkhh/Projects/scenario_gen_multi/output/ay_sweep"
LOG_DIR="${SCRIPT_DIR}/output/rb"
CONTROLLER="${SCRIPT_DIR}/custom_controller_cm.py"

mkdir -p "${LOG_DIR}"

# ── TESTRUN_NAME: 시나리오별 CarMaker 테스트런 ────────────────
declare -A TESTRUN
TESTRUN["WLTP"]="WLTP"
TESTRUN["HWFET"]="HWFET"
TESTRUN["FTP75"]="FTP"

# ── YAWM_SMC_K: ay레벨별 튜닝값 ──────────────────────────────
declare -A SMC_K_DEFAULT
SMC_K_DEFAULT["ay1p0"]="10.0"
SMC_K_DEFAULT["ay2p0"]="4.0"
SMC_K_DEFAULT["ay3p0"]="2.7"
SMC_K_DEFAULT["ay4p0"]="2.5"
# ay5p0는 시나리오별 별도 처리

declare -A SMC_K_AY5
SMC_K_AY5["WLTP"]="3.0"
SMC_K_AY5["HWFET"]="3.7"
SMC_K_AY5["FTP75"]="3.7"

# ── 실행 함수 ─────────────────────────────────────────────────
run_one() {
    local scenario_name="$1"   # e.g. WLTP
    local ay="$2"              # e.g. ay3p0
    local algo="$3"            # algo2 or algo3
    local csv_path="$4"
    local smc_k="$5"
    local testrun="$6"

    local tag="${scenario_name}_${ay}_${algo}"
    local existing
    existing=$(ls "${LOG_DIR}/${tag}"_telemetry_ep*.csv 2>/dev/null | head -1 || true)
    if [[ -n "${existing}" ]]; then
        echo "[SKIP] ${tag} 이미 존재: $(basename "${existing}")"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "[RUN] ${tag}  SMC_K=${smc_k}  TESTRUN=${testrun}"
    echo "========================================"

    if RUN_MODE=HEADLESS \
       TESTRUN_NAME="${testrun}" \
       SCENARIO_CSV_PATH="${csv_path}" \
       OPEN_LOOP_STEER=0 \
       TORQUE_DISTRIBUTION_MODE="${algo}" \
       YAWM_SMC_K="${smc_k}" \
       LOG_OUTPUT_DIR="${LOG_DIR}" \
       python3 "${CONTROLLER}"; then
        echo "[OK] ${tag} 완료"
    else
        echo "[ERROR] ${tag} 실패 (종료코드 $?) — 다음으로 계속"
    fi
}

# ── 메인 루프 ─────────────────────────────────────────────────
SCENARIOS=("WLTP" "HWFET" "FTP75")
AY_LEVELS=("ay1p0" "ay2p0" "ay3p0" "ay4p0" "ay5p0")
ALGOS=("algo2" "algo3")

total=0
for scenario in "${SCENARIOS[@]}"; do
for ay in "${AY_LEVELS[@]}"; do
for algo in "${ALGOS[@]}"; do

    csv="${SCENARIO_BASE}/simulation_results_${scenario}_${ay}.csv"
    if [[ ! -f "${csv}" ]]; then
        echo "[WARN] CSV 없음: ${csv} — skip"
        continue
    fi

    if [[ "${ay}" == "ay5p0" ]]; then
        smc_k="${SMC_K_AY5["${scenario}"]}"
    else
        smc_k="${SMC_K_DEFAULT["${ay}"]}"
    fi

    run_one "${scenario}" "${ay}" "${algo}" "${csv}" "${smc_k}" "${TESTRUN["${scenario}"]}"
    ((total++)) || true

done
done
done

echo ""
echo "========================================"
echo "완료: 총 ${total}회 실행"
echo "출력 디렉토리: ${LOG_DIR}"
echo "========================================"

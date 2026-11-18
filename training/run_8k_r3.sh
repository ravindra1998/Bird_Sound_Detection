#!/bin/bash
# Round 3 (user 2026-11-12): extra weight on the false alarms - cat + frog x6, train negatives that r8k_1 OR r8k_2 alarm on x5
cd /c/Users/Ravindra/Desktop/Bird_sound_detection_PCEN/bird_voice_8khz || exit 1
PY=/c/Users/Ravindra/anaconda3/envs/tf210/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
LOG=logs/run_8k_final.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { local n=$1; shift; log "START $n"
    if "$@" > "logs/$n.log" 2>&1; then log "DONE  $n"; else log "FAILED $n (logs/$n.log)"; exit 1; fi; }
step train_r8k_3 env HARD_FROM=r8k_1,r8k_2 HARD_W=5 CAT_BOOST=cat=6,frog=6 $PY train_8k_final.py r8k_3
step score_r8k_3 $PY score_board_8k.py r8k_1 r8k_2 r8k_3
log "ROUND 3 DONE"

#!/bin/bash
# Round 4 (user 2026-11-12): round-3 settings + extra weight on weak bird species (no 2,500 cap for them)
cd /c/Users/Ravindra/Desktop/Bird_sound_detection_PCEN/bird_voice_8khz || exit 1
PY=/c/Users/Ravindra/anaconda3/envs/tf210/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
LOG=logs/run_8k_final.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { local n=$1; shift; log "START $n"
    if "$@" > "logs/$n.log" 2>&1; then log "DONE  $n"; else log "FAILED $n (logs/$n.log)"; exit 1; fi; }
log "round 4 waiting for round 3"
until grep -qE "ROUND 3 DONE|FAILED (train|score)_r8k_3" "$LOG"; do sleep 60; done
grep -q "ROUND 3 DONE" "$LOG" || { log "round 4 not started: round 3 failed"; exit 1; }
step train_r8k_4 env HARD_FROM=r8k_1,r8k_2 HARD_W=5 CAT_BOOST=cat=6,frog=6 \
    "BIRD_BOOST=board_bird=3,Kingfisher=2,Peacock=1.5,Babbler=1.5,Duck=1.5,Sparrow=1.5,Cuckoo=1.5" \
    $PY train_8k_final.py r8k_4
step score_r8k_4 $PY score_board_8k.py r8k_1 r8k_2 r8k_3 r8k_4
log "ROUND 4 DONE"

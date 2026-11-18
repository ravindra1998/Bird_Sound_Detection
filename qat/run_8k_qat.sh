#!/bin/bash
# QAT (user 2026-11-12): after round 4, QAT fine-tune rounds 3 and 4 (5 epochs, lr 1e-4, their own weights), score all
cd /c/Users/Ravindra/Desktop/Bird_sound_detection_PCEN/bird_voice_8khz || exit 1
PY=/c/Users/Ravindra/anaconda3/envs/tf210/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
LOG=logs/run_8k_final.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { local n=$1; shift; log "START $n"
    if "$@" > "logs/$n.log" 2>&1; then log "DONE  $n"; else log "FAILED $n (logs/$n.log)"; return 1; fi; }
log "QAT waiting for rounds 3 and 4"
until grep -qE "ROUND 4 DONE|round 4 not started|FAILED (train|score)_r8k_[34]" "$LOG"; do sleep 60; done
RUNS=""
[ -f runs/r8k_3/model.h5 ] && step qat_r8k_3q $PY qat_8k.py r8k_3 r8k_3q && RUNS="$RUNS r8k_3 r8k_3q"
[ -f runs/r8k_4/model.h5 ] && step qat_r8k_4q $PY qat_8k.py r8k_4 r8k_4q && RUNS="$RUNS r8k_4 r8k_4q"
[ -n "$RUNS" ] && step score_qat $PY score_board_8k.py r8k_2 $RUNS
log "QAT DONE"

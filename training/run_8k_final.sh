#!/bin/bash
# LOCKED 8 kHz task (2026-11-11), unattended overnight run. Log: bird_voice_8khz/logs/run_8k_final.log
#   wait  for the 8 kHz BirdNET check (class_check/audit8k_train.txt + audit8k_unseen.txt)
#   clean clean_8k.py      move non-bird / empty / no-bird-at-8k clips, delete their feature rows, Wren decision
#   r8k_1 round 1          category weights x3
#   r8k_2 round 2          + round-1 false alarms on train negatives x3
#   score score_board_8k.py (test_birds / test_zero, 1-window then 2-of-3) for both rounds
cd /c/Users/Ravindra/Desktop/Bird_sound_detection_PCEN/bird_voice_8khz || exit 1
PY=/c/Users/Ravindra/anaconda3/envs/tf210/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
LOG=logs/run_8k_final.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { local n=$1; shift; log "START $n"
    if "$@" > "logs/$n.log" 2>&1; then log "DONE  $n"; else log "FAILED $n (logs/$n.log)"; exit 1; fi; }

log "waiting for the 8 kHz BirdNET check"
while [ ! -f class_check/audit8k_train.txt ] || [ ! -f class_check/audit8k_unseen.txt ]; do sleep 60; done
step clean_8k      $PY clean_8k.py
step train_r8k_1   $PY train_8k_final.py r8k_1
step score_r8k_1   $PY score_board_8k.py r8k_1
step train_r8k_2   env HARD_FROM=r8k_1 $PY train_8k_final.py r8k_2
step score_r8k_2   $PY score_board_8k.py r8k_1 r8k_2
log "ALL DONE"

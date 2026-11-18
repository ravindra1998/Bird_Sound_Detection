#!/bin/bash
# Round 5 - LOCKED by user 2026-11-13 (final round, unattended). Scorecard 486 calls (6 no-bird-at-8k calls excluded).
#   hard negatives: train negatives r8k_1 | r8k_2 | r8k_4q alarm on -> x5
#   category x: cat 8, frog 8, monkey 8, traffic 6, siren 6, dog 5, children_playing 5, donkey 5, elephant 5 (rest of x3 list kept)
#   bird boost: round 4 (board_bird 3, Kingfisher 2, Peacock/Babbler/Duck/Sparrow/Cuckoo 1.5)
#   then QAT (5 epochs, lr 1e-4) -> r8k_5q, score r8k_4 r8k_4q r8k_5 r8k_5q
cd /c/Users/Ravindra/Desktop/Bird_sound_detection_PCEN/bird_voice_8khz || exit 1
PY=/c/Users/Ravindra/anaconda3/envs/tf210/python.exe
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
LOG=logs/run_8k_final.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
step() { local n=$1; shift; log "START $n"
    if "$@" > "logs/$n.log" 2>&1; then log "DONE  $n"; else log "FAILED $n (logs/$n.log)"; exit 1; fi; }
step train_r8k_5 env HARD_FROM=r8k_1,r8k_2,r8k_4q HARD_W=5 \
    CAT_BOOST=cat=8,frog=8,monkey=8,traffic=6,siren=6,dog=5,children_playing=5,donkey=5,elephant=5 \
    "BIRD_BOOST=board_bird=3,Kingfisher=2,Peacock=1.5,Babbler=1.5,Duck=1.5,Sparrow=1.5,Cuckoo=1.5" \
    $PY train_8k_final.py r8k_5
step qat_r8k_5q   $PY qat_8k.py r8k_5 r8k_5q
step score_r8k_5  $PY score_board_8k.py r8k_4 r8k_4q r8k_5 r8k_5q
log "ROUND 5 DONE"

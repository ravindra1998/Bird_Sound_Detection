# Bird Sound Detection on GPX10: 8 kHz Model Report

13 Nov 2026

## Summary

The released 8 kHz model is **r8k_4q** (`AUD_BirdSound_8k_INT8_v1.0`): 147k parameters, 159 KB int8, about 1.32 s per window on the GPX10. On the board it catches **486 of 486** played bird calls with the 1-window rule and **480 of 486** with 2-of-3, gives 0 false alarms per hour in the office test, and reaches **98.8 % accuracy** on 45,000 held-out clips.

The 8 kHz phase ran from 11 to 13 Nov 2026, after the GPX10 was changed to support only 8 kHz audio. All data was converted to 8 kHz, cleaned again with BirdNET at 8 kHz, and trained in five rounds (r8k_1 to r8k_5) plus QAT. The earlier 16 kHz work is in the 16 kHz report.

The target is not fully met: every call is caught, but the model still gives **20 false alarms** on the 1-hour played non-bird test recording with 2-of-3, mostly **cat (11)** and **frog (7)**. At 8 kHz, cat meows and some animal calls look too similar to birds for sample weighting alone to fix.

| Requirement | Target | r8k_4q (released) |
| --- | --- | --- |
| Bird calls caught, 1-window | 100 % | 486 / 486 (100 %) |
| Bird calls caught, 2-of-3 | 100 % | 480 / 486 (98.8 %) |
| Alarms on 7 zero sounds, 2-of-3 | 0 | 20 (cat 11, frog 7, cricket 1, thunder 1) |
| Office false alarms | at most 1 to 2 per hour | 0 per hour |
| Time per window on GPX10 | under 1.5 s | about 1.32 s (estimate), heap OK |

## Goal, target and constraints

The locked target (set 8 Nov 2026, kept for both phases) is to catch every bird call while staying silent on common nature and indoor sounds, on the real GPX10 board.

| # | Requirement | How it is measured |
| --- | --- | --- |
| 1 | Catch every bird call (100 %) | test_birds playback recording on the GPX10 (486 calls of 15 species) |
| 2 | 0 false alarms on frog, cricket, cicada, cat, rain, wind, thunder | test_zero playback recording on the GPX10 (about 1 h) |
| 3 | At most 1 to 2 false alarms per hour in normal office and outdoor sound | office test session 20261110_110606 (45.8 min) and held-out recordings |
| 4 | Runs on the GPX10 in under 1.5 s per window, heap at most 284 KB | device_cost.py estimate and the TFLite check |

**Fixed settings:**

- Threshold 0.5 for every model (no per-model threshold).
- Alarm rule: 1-window first (bird recall has priority), then 2-of-3 as the fallback.
- Output class order: index 0 = bird (wake-word), index 1 = not bird.
- The test recordings (test_birds, test_zero, office test) never go into training.

**Device constraints (GPX10, 50 MHz):**

- Full int8 TFLite, tf.keras Sequential model.
- Supported ops only: Conv2D, DepthwiseConv2D, AveragePooling2D, Dense (no GlobalAveragePooling2D).
- Heap (activation arena plus model) at most 284 KB.
- Parameter budget raised over time: under 50k at first, then 100k, then 150k (10 Nov).
- Depthwise layers are slow on this chip (about 36 µs per output element), so plain Conv2D is used except in late wide layers.

**Microphone path:** I2S microphone, raw int16, streamed as text over UART at 921,600 baud on COM3. At 16 kHz the UART drops about 11 % of samples evenly, so the live dashboard stretches the audio by x1.1285 to restore true speed.

## Data

The final 8 kHz training set has **284,458 clips of 2 s** (62,860 bird, 221,598 not bird), with 40,653 for validation and 45,000 for test. All clips are mono, 2.00 s, and since 11 Nov 2026 stored at 8 kHz.

**Folders (project root):**

| Folder | Contents |
| --- | --- |
| training_dataset/ | wake-word/<species> and Not-wake-word/<category> clips used for training, including 30,841 GPX10 board clips (gpx_run15__, gpx_pb14__) |
| Bird_Balanced_Dataset/ | train / val / test split of the bird clips (source of features/train.npz) |
| unseen_test/ | 44,473 held-out test clips (unseen_test.csv), never trained on |
| playback/ | GPX10 recordings of the scheduled playback: test_birds, test_zero, train recordings, with alignment CSVs |
| run15_data/ | board sessions recorded for run15 (live dashboard), windows and manifest |
| features/, bird_voice_8khz/features_final/ | 16 kHz and 8 kHz feature files (npz) |
| removed/ | every clip taken out by a cleaning step, with a moved_list.csv |

**Bird species:** 24 species were used over the project; the released 8 kHz model is trained on 22 of them plus GPX10 board bird recordings (board_bird, 993 clips). Northern Mockingbird and Bewick's Wren were trained in the 16 kHz models only; the 8 kHz model is not trained on them.

| # | Species | Includes | 16 kHz models | 8 kHz release (r8k_4q) | 8 kHz training clips |
| --- | --- | --- | --- | --- | --- |
| 1 | American Robin |  | trained | trained | 4,103 |
| 2 | Babbler | Puff-throated, Pomatorhinus and Jungle babblers | trained | trained | 4,005 |
| 3 | Bewick's Wren |  | trained | not trained | 0 |
| 4 | Blue Jay |  | trained | trained | 2,304 |
| 5 | Common Hoopoe |  | not trained | trained | 1,879 |
| 6 | Crow |  | trained | trained | 2,094 |
| 7 | Cuckoo | all cuckoos, including koels | trained | trained | 2,977 |
| 8 | Duck | whole group (mainly Mallard) | trained | trained | 4,030 |
| 9 | Eurasian Blackbird |  | not trained | trained | 909 |
| 10 | Eurasian Golden Oriole |  | not trained | trained | 964 |
| 11 | Eurasian Jay |  | not trained | trained | 5,695 |
| 12 | Kingfisher | whole group, including White-throated | trained | trained | 2,877 |
| 13 | Light-vented Bulbul |  | not trained | trained | 1,137 |
| 14 | Myna | all mynas and Rosy Starling | trained | trained | 1,608 |
| 15 | Nightingale |  | trained | trained | 4,294 |
| 16 | Northern Cardinal |  | trained | trained | 3,248 |
| 17 | Northern Lapwing |  | trained | trained | 3,587 |
| 18 | **Northern Mockingbird** |  | trained | not trained | 0 |
| 19 | Parrot |  | trained | trained | 2,979 |
| 20 | Peacock | Indian Peafowl | trained | trained | 2,991 |
| 21 | Red-wattled Lapwing | and other lapwings | trained | trained | 1,884 |
| 22 | Song Thrush |  | not trained | trained | 4,019 |
| 23 | Sparrow | including Song Sparrow | trained | trained | 2,742 |
| 24 | Thrush Nightingale |  | not trained | trained | 1,541 |

Species with more than 2,500 clips are weighted down to about 2,500 in training. The 16 kHz models therefore covered 17 species (including Northern Mockingbird and Bewick's Wren); the 8 kHz release covers 22.

**Species changes:**

- 29 Oct: 17 target species fixed, including Bewick's Wren. Group-level classes used (all cuckoos, all mynas, all kingfishers).
- 11 Nov: Northern Mockingbird dropped. Seven species added from Xeno-canto and local folders: Song Thrush, Eurasian Blackbird, Eurasian Golden Oriole, Light-vented Bulbul, Thrush Nightingale, Common Hoopoe, Eurasian Jay.
- 11 Nov (8 kHz): Bewick's Wren dropped, because only 31 % of its clips keep a bird at 8 kHz.

**Not-bird sounds (about 45 categories):** silence, office ambience and office recordings, outdoor ambience, wind, rain, thunder, cricket, cicada, insects, frog, cat, dog, puppy, monkey, donkey, elephant, cow, sheep, chicken, lion, pig, conversation, laughing, children playing, podcast, music, songs, street music, traffic, engine idling, siren, horn, drilling, jackhammer, air conditioner, glass breaking, door, clapping, gun shot, whistle, fire crackers, and more.

**Cleaning steps (every removed clip is kept in removed/):**

1. 29 Oct: BirdNET check of all bird clips; clips with no bird or the wrong species dropped.
2. 11 Nov: found that the 5 worst false-alarm species (Myna, Peacock, Mockingbird, Crow, Red-wattled Lapwing) were padded with augmented copies. All 2,731 augmented copies were removed and replaced with real recordings (Xeno-canto, iNaturalist, PeaDetect).
3. 11 Nov: final BirdNET audit at 16 kHz removed 461 empty or non-bird clips.
4. 11 Nov (8 kHz): BirdNET re-run on the 8 kHz audio moved **8,331 clips** (empty, non-bird, or bird lost at 8 kHz) and deleted 7,179 feature rows.
5. 12 Nov: 6 test_birds calls with no bird sound below 4 kHz (5 Kingfisher, 1 Northern Cardinal) excluded from the 8 kHz scorecard after listening (playback/test_birds_excluded_8k.csv).

**Board data:** the speaker-to-board path changes sound a lot (about -20 to -28 dB under 250 Hz, +10 to +18 dB at 2 to 3.5 kHz), so real GPX10 recordings were collected: scheduled bird and zero-sound playback (run14), and live sessions of office, cat, frog, monkey, whistle, puppy, glass, crow and peacock (run15). Users listened to the alarms and labelled the spans.

## Starting point from the 16 kHz phase

The 8 kHz work reused the data, test recordings, scorecard and model design from the 16 kHz phase (full details in the 16 kHz report).

- **Best 16 kHz models:** run14aq (1-window 521/524 calls, 7 zero-sound alarms with 2-of-3) and run15q.
- **Model design carried over:** mid150 (Conv 16-32-80-128 plus 1x1 192), int8 QAT, PCEN front end, 2 s window every 1 s.
- **Lesson carried over:** false alarms came mostly from cat and frog, and from augmented bird copies (removed on 11 Nov).

A first 8 kHz test on 6 Nov (n_fft 128 / hop 64, 251 frames) lost to 16 kHz: about 4 false alarms per hour against 0.4, at the same recall, and with no saving in device time.

## Move to 8 kHz

On 11 Nov 2026 the device was changed to support **only 8 kHz audio**, so 8 kHz became the production path with the same locked target.

**Front end (`bird_preprocess_8k.py`):** the same STFT size in samples as 16 kHz, so a 2 s window gives 126 frames instead of 251 and the model needs about half the compute.

| Setting | 16 kHz (run15q) | 8 kHz (r8k models) |
| --- | --- | --- |
| Window | 32,000 samples = 2.00 s | 16,000 samples = 2.00 s |
| n_fft / hop | 256 / 128 (16 ms / 8 ms) | 256 / 128 (32 ms / 16 ms) |
| Mel bands | 40 over 0 to 8 kHz | 40 over 0 to 4 kHz |
| Frames | 251 | 126 |
| PCEN | time constant 0.4 s | same (25 frames) |
| Model input | (1, 1, 251, 40) int8 | (1, 1, 126, 40) int8 |
| int8 input scale / zero point | 0.0194853 / -128 | same |

**Steps taken:**

1. All audio in training_dataset, unseen_test and Bird_Balanced_Dataset was converted **in place** to 8 kHz, 2.00 s, mono (519,425 files). The 16 kHz record is kept only in features/*.npz.
2. Features were extracted again from the 8 kHz audio into bird_voice_8khz/features_final/ (39 sets). Board windows were matched exactly to the 16 kHz windows.
3. BirdNET was re-run on the 8 kHz audio. Clips that were empty, non-bird, or lost their bird at 8 kHz were moved out: **8,331 clips**, 7,179 feature rows.
4. **Bewick's Wren was dropped:** only 31 % of its 5,441 clips keep a bird (BirdNET at least 0.3) at 8 kHz, because about 70 % of its energy is above 4 kHz.
5. Board scoring uses the 16 kHz GPX10 test recordings, converted to 8 kHz (low-pass and decimate by 2), because the board stream stayed at 16 kHz.

## Training rounds

Five rounds were trained; QAT on round 4 gave the released model. Every round uses the same model (mid150 with a first time stride of 2: 146,962 parameters, 162 KB int8) and the same recipe: 60 epochs, cosine learning-rate decay, about 100 s per epoch on CPU (tf210 env, TF 2.10).

**Sample weights used in every round:**

- **Category weights x3** (locked list): cicada, cricket, insects, cat, frog, conversation, monkey, rain, office ambience, office data, traffic, puppy dog, whistle.
- **Species cap:** species with more than 2,500 clips are weighted down to about 2,500 (Eurasian Jay, Nightingale, American Robin, Duck, Song Thrush, Babbler and others).
- **Balance:** total bird weight is rescaled to equal total not-bird weight.
- **Hard negatives (round 2 on):** training not-bird clips that an earlier model scores at 0.5 or more get extra weight. Finding them takes about 72 min per model.

| Round | Model | What changed | Script |
| --- | --- | --- | --- |
| 1 | r8k_1 | category weights x3, species cap | run_8k_final.sh |
| 2 | r8k_2 | + r8k_1 false alarms on training clips x3 | run_8k_final.sh |
| 3 | r8k_3 | cat and frog x6; r8k_1 and r8k_2 false alarms x5 | run_8k_r3.sh |
| 4 | r8k_4 | round 3 + weak bird species: board_bird x3, Kingfisher x2, Peacock, Babbler, Duck, Sparrow, Cuckoo x1.5 | run_8k_r4.sh |
| QAT | r8k_3q, **r8k_4q** | 5 epochs of quantization-aware training at learning rate 1e-4, same weights as the source round | run_8k_qat.sh, qat_8k.py |
| 5 | r8k_5, r8k_5q | cat, frog, monkey x8; traffic, siren x6; dog, children playing, donkey, elephant x5; r8k_4q false alarms x5 (2,541 clips); then QAT | run_8k_r5.sh |

**Why QAT:** plain int8 conversion of r8k_4 lost about 1 point of bird recall (float 97.1 % to int8 96.1 %). QAT trains with simulated int8, so the int8 model keeps the accuracy. It raised recall to 98.3 %, but also shifted all scores up slightly, which raised false alarms.

**Why round 5 failed:** heavier weights on nine sound types made the model more cautious overall (15 fewer calls with 2-of-3) without teaching it to separate cat from bird; cat false alarms rose from 8.1 % to 10.9 % of test cat clips. Heavier weighting has stopped helping.

## How models are scored

Every model is scored as int8 at threshold 0.5 on the same four tests (score_board_8k.py and the run report).

1. **Board scorecard, birds:** test_birds GPX10 recording (59.9 min), 486 scheduled calls of 15 species. A call counts as caught when an alarm fires between 0.3 s after its onset and 2.5 s after its end.
2. **Board scorecard, zero sounds:** test_zero GPX10 recording (59.3 min) of frog, cricket, cicada, cat, rain, wind and thunder. Every alarm is counted per sound; 4 spots that contain real birds are excluded (test_zero_excluded.csv). Alarms in the silent gaps between calls are counted separately (gap FA).
3. **Held-out clips:** 45,000 test clips of 2 s (9,281 bird, 35,719 not bird), per species and per not-bird category.
4. **Recordings:** 13.9 h of held-out not-bird recordings (false alarms per hour), office ambience and office data clips, and GPX10 board bird windows (142).

**Alarm rules:** windows are 2 s long, one every 1 s.

- **1-window:** any window at 0.5 or more raises an alarm. Fastest response, most false alarms.
- **2-of-3:** at least 2 of the last 3 windows at 0.5 or more. It can never give more alarms than 1-window, removes most single-window false alarms, adds about 1 s of delay, and can miss very short calls.

The board recordings were made at 16 kHz and are converted to 8 kHz the same way the dashboard does it, because the GPX10 stream stayed at 16 kHz during this phase.

## Results

**r8k_4q is the best overall:** it is the only model that catches every call with 1-window and the most with 2-of-3. r8k_4 gives the fewest false alarms (8) but misses 31 calls with 2-of-3.

**Board scorecard** (486 calls, threshold 0.5; "zero" = alarms on the 7 zero sounds):

| Model | 1-window calls | 1-window zero | 2-of-3 calls | 2-of-3 zero | Cat | Frog | Gap FA | 1-window misses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r8k_1 | 482 | 60 | 471 | 25 | 20 | 3 | 4 | Kingfisher 2, Sparrow 1, Cuckoo 1 |
| r8k_2 | 484 | 62 | 477 | 23 | 19 | 4 | 3 | Sparrow 1, Kingfisher 1 |
| r8k_3 | 484 | 38 | 469 | 15 | 11 | 3 | 3 | Sparrow 1, Cuckoo 1 |
| r8k_3q | **486** | 52 | 478 | 20 | 15 | 4 | 3 | none |
| r8k_4 | 477 | 30 | 455 | **8** | **6** | 1 | 5 | Kingfisher 4, Cuckoo 2, 3 others |
| **r8k_4q** | **486** | 55 | **480** | 20 | 11 | 7 | 3 | **none** |
| r8k_5 | 479 | 49 | 465 | 27 | 25 | 2 | 3 | Kingfisher 3, 4 others |
| r8k_5q | 480 | 48 | 465 | 24 | 21 | 2 | 4 | Kingfisher 3, Cuckoo 2, Blue Jay 1 |

Cat and frog columns are 2-of-3. Cricket, cicada, rain, wind and thunder gave 0 or 1 alarm each with 2-of-3 for every model.

**Held-out clips and recordings:**

| Model | Bird recall | False-positive rate | GPX10 bird windows | FA/h 1-window | FA/h 2-of-3 | Cat | Monkey | Traffic | Children playing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r8k_1 | 97.4 % | 0.79 % | 0.56 | 10.3 | 2.8 | 8.4 % | 9.0 % | 0.8 % | 2.3 % |
| r8k_2 | 98.1 % | 0.96 % | **0.66** | 13.5 | 4.2 | 8.2 % | 11.3 % | 0.4 % | 5.7 % |
| r8k_3 | 97.5 % | 0.66 % | 0.55 | 9.1 | 3.2 | 5.1 % | 12.0 % | 0.6 % | 4.6 % |
| r8k_3q | 98.0 % | 0.90 % | 0.59 | 12.0 | 4.2 | 7.5 % | 13.5 % | 0.7 % | 5.1 % |
| r8k_4 | 96.1 % | **0.55 %** | 0.54 | **7.5** | **2.4** | **4.3 %** | 11.3 % | 0.4 % | 7.4 % |
| **r8k_4q** | **98.3 %** | 1.12 % | 0.65 | 15.5 | 5.3 | 8.1 % | 13.5 % | 1.1 % | 11.4 % |
| r8k_5 | 96.6 % | 1.05 % | 0.55 | 12.9 | 5.5 | 12.4 % | 10.5 % | 0.6 % | 6.9 % |
| r8k_5q | 97.4 % | 1.10 % | 0.63 | 14.7 | 6.1 | 10.9 % | 12.8 % | 0.7 % | 8.0 % |

The category columns are the share of test clips of that sound that alarm. Office false alarms were 0 per hour for every model (office ambience 0.02 % to 0.46 % of clips, office data 0 % to 0.06 %).

**r8k_4q in detail (45,000 held-out clips):** accuracy 98.8 %, balanced accuracy 98.6 %.

| Class | Precision | Recall | F1 | Clips |
| --- | --- | --- | --- | --- |
| bird | 95.8 % | 98.3 % | 97.0 % | 9,281 |
| not_bird | 99.6 % | 98.9 % | 99.2 % | 35,719 |
| Macro average | 97.7 % | 98.6 % | 98.1 % | 45,000 |

| True \ predicted | bird | not_bird |
| --- | --- | --- |
| bird | 9,124 | 157 |
| not_bird | 399 | 35,320 |

All eight models are 163 KB int8, pass the heap check and use only AVERAGE_POOL_2D, CONV_2D, FULLY_CONNECTED, RESHAPE and SOFTMAX.

## Live testing on the GPX10

The 8 kHz models are tested live with `live_dashboard_8k.py`, a browser dashboard at http://127.0.0.1:8766 that reads the board over COM3 (921,600 baud) and records every session to `live_sessions/`.

| Launcher (double-click) | Model | Rule |
| --- | --- | --- |
| start_live_dashboard_8k_r8k_4q.bat | r8k_4q (released) | 2-of-3 |
| start_live_dashboard_8k_r8k_4q_1win.bat | r8k_4q (released) | 1-window |

The 16 kHz dashboard (`live_dashboard.py`, port 8765) is separate. Only one dashboard can use COM3 at a time.

**Board stream:**

- **16 kHz firmware (normal):** the board sends about 14,150 samples per second because the UART drops samples evenly; the dashboard stretches the audio by 2257/2000 back to true speed, then converts each 2 s window to 8 kHz.
- **8 kHz firmware:** about 8,010 samples per second, clean. Start the dashboard with `--board_sr 8000`. On 12 Nov the 8 kHz firmware was only loaded to RAM, so a board reset brought back the 16 kHz image; it must be written to flash for 8 kHz tests.

**Checks done:** live session 20261112_153807 (r8k_3) ran at real-time speed, the room level was about -68 dBFS, and the dashboard scores matched the offline 8 kHz path exactly.

## Release package

r8k_4q is released in the Model Zoo as **`AUD_BirdSound_8k_INT8_v1.0`** (ambient-sw-ai-models, branch ai_models, folder Model_Zoo/Models/), in the same layout as the other models.

| File | Contents |
| --- | --- |
| model/AUD_BirdSound_8k_INT8_v1.0.tflite | the int8 model, 159.2 KB (identical to r8k_4q_int8.tflite in this folder) |
| model_card.yaml | model card, schema 3 (use case, input, 10 pre-processing steps, classes, output, accuracy, per class, confusion matrix) |
| preprocessing/preprocess.py | reference front end in numpy only, with `--check` |
| preprocessing/README.md | front-end guide, samples, device notes |
| preprocessing/raw/ | 3 GPX10 test windows, 8 kHz, 16-bit WAV |
| preprocessing/processed/ | the same windows as int8 model input, 126 x 40 CSV |
| release/AUD_BirdSound_8k_INT8_v1.0_ReleaseNote.docx | public release note, 8 pages |
| release/AUD_BirdSound_8k_INT8_v1.0_ReleaseNote_Internal.docx | internal release note, 12 pages, with figures |

**Reference samples (all three print MATCH with `--check` and give the right answer):**

| Sample | Source | Model output (int8) | p(bird) |
| --- | --- | --- | --- |
| sample_01_bird_nightingale | Nightingale song (XC840431) | [127, -127] | 0.996 |
| sample_02_bird_cuckoo_koel | Asian Koel (XC309568) | [127, -128] | 0.996 |
| sample_03_rain | rain (FSD / ESC-50) | [-126, 126] | 0.008 |

The numpy front end gives the same int8 values as the training front end (librosa), except one value of sample 02 that differs by 1 step; the model output is the same.

**Pre-processing on the device:**

1. Capture 16-bit PCM at 8 kHz (or low-pass and decimate a 16 kHz stream by 2).
2. Divide by 32768 and keep the newest 16,000 samples (2 s).
3. If RMS is above 0.005 and the peak above 0.0001, scale the window so its peak is 0.95.
4. Add 128 zeros at both ends; 126 frames of 256 samples every 128, periodic Hann window.
5. 256-point FFT, power (129 bins).
6. 40 Slaney mel bands, 0 to 4 kHz, unit area.
7. Multiply by 2^31.
8. PCEN per band: M[t] = 0.0392 E[t] + 0.9608 M[t-1], starting from 1; output sqrt(2) (sqrt(1 + E / (M + 1e-6)^0.98 / 2) - 1).
9. Quantise: round(x / 0.0194853) - 128, clamped to -128..127.

**Output:** [1, 2] int8, scale 1/256, zero point -128. Index 0 is bird; a window is bird when output index 0 is 0 or more as a signed int8 (score 0.5 or more). Run every 1 s and raise the alarm on 2 of the last 3 windows.

## Known limits and next steps

The main gap is false alarms on animal sounds that look like birds at 8 kHz. More weighting did not fix it (round 5), so the next step needs a different kind of change.

**Known limits of r8k_4q:**

- **Cat:** 11 alarms on the test recording with 2-of-3; 8.1 % of test cat clips alarm.
- **Frog, monkey, children playing:** frog 7 alarms on the test recording; monkey 13.5 % and children playing 11.4 % of test clips.
- **High-pitched birds:** birds that call only above 4 kHz (some Kingfisher calls, Bewick's Wren) cannot be heard at 8 kHz.
- **Device time** is an estimate (about 1.32 s); it has not been measured on the board yet.
- **Board data at 8 kHz:** all board scoring used 16 kHz recordings converted to 8 kHz; no native 8 kHz board recording exists yet.

**Next steps:**

- [ ] Live-test r8k_4q on the GPX10 and score the sessions with both rules
- [ ] Write the 8 kHz firmware to flash and record test_birds, test_zero and office again at 8 kHz on the board
- [ ] Measure latency and power of r8k_4q on the GPX10 for the release note
- [ ] Check a threshold of 0.55 to 0.7 with 2-of-3 on r8k_4q (cat and frog alarms against calls lost); the threshold is locked at 0.5 until decided
- [ ] For cat and monkey: collect more real board recordings of these sounds, or try a model or feature change aimed at them
- [ ] Commit the AUD_BirdSound_8k_INT8_v1.0 folder to the Model Zoo

## Appendix: this folder

| Folder | Contents |
| --- | --- |
| model/ | r8k_4q_int8.tflite (released int8 model), r8k_4_float.h5 (float Keras model of round 4, the QAT starting point), r8k_4q_qat_weights.h5 (QAT weights), report and QAT comparison |
| preprocessing/ | bird_preprocess_8k.py (training front end, librosa), preprocess_numpy_reference.py (numpy reference with --check), raw/ and processed/ samples |
| live_dashboard/ | live_dashboard_8k.py with its helpers (live_dashboard.py, bird_preprocess.py, bird_preprocess_8k.py) and two launchers |
| training/ | train_8k_final.py, train_bird_model.py (helpers), bird_preprocess_8k.py, score_board_8k.py and the round scripts run_8k_final.sh, run_8k_r3.sh, run_8k_r4.sh, run_8k_r5.sh |
| qat/ | qat_8k.py, qat_finetune.py, train_8k_final.py, train_bird_model.py, run_8k_qat.sh |
| document/ | this report |

The training and QAT scripts read their data from the project (`bird_voice_8khz/features_final/`, `runs/`); run them from `Bird_sound_detection_PCEN/bird_voice_8khz/` in the tf210 conda env (`CUDA_VISIBLE_DEVICES=-1`).

**Reproduce the released model:**

1. `bash run_8k_final.sh` (clean-up, r8k_1, r8k_2)
2. `bash run_8k_r3.sh`, then `bash run_8k_r4.sh`
3. `python qat_8k.py r8k_4 r8k_4q`
4. `python score_board_8k.py r8k_4q`

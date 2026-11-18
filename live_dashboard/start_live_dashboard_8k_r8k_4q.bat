@echo off
rem Live test dashboard for the 8 kHz r8k_4q (2-of-3) int8 bird model. Opens http://127.0.0.1:8766 in the browser.
rem Pick "GPX10 board (serial)" and COM3, keep Record ticked, press Start. Close this window to stop.
rem Sessions are saved in live_sessions\ next to this file. Only one dashboard can use the COM port at a time.
cd /d "%~dp0"
set CUDA_VISIBLE_DEVICES=-1
set PYTHONIOENCODING=utf-8
start "" /min cmd /c "timeout /t 10 >nul & start http://127.0.0.1:8766"
"C:\Users\Ravindra\anaconda3\envs\tf210\python.exe" live_dashboard_8k.py --model ..\model\r8k_4q_int8.tflite --threshold 0.5 --port 8766 --baud 921600
pause

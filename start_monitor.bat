@echo off
REM ============================================================
REM  CORE Automated Anomaly Monitor, Launcher
REM  Double-click this file to start the monitor.
REM  A log window will appear showing live detection status.
REM ============================================================

title CORE Anomaly Monitor
color 0A

echo.
echo  ==========================================
echo   CORE AUTOMATED ANOMALY MONITOR
echo  ==========================================
echo.
echo  Starting monitor... (do not close this window)
echo.

cd /d "%~dp0"

REM Use the Anaconda Python that has all the required packages
"C:\Users\user\anaconda3\python.exe" monitor.py

echo.
echo  Monitor stopped. Press any key to exit.
pause >nul

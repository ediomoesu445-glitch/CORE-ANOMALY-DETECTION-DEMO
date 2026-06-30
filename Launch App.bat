@echo off
title NMDPRA Anomaly Detection System
echo Starting NMDPRA Anomaly Detection System...
echo.
echo The app will open in your browser at http://localhost:8501
echo Keep this window open while using the app.
echo Close this window to stop the app.
echo.
"C:\Users\user\anaconda3\python.exe" -m streamlit run "%~dp0app.py" --server.headless false
pause

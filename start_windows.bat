@echo off
cd /d "%~dp0"
python -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 (
  echo.
  echo If packages are missing, run: python -m pip install -r requirements.txt
  pause
)

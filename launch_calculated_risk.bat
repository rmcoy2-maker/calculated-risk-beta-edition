@echo off
setlocal

REM =========================================================
REM Calculated Risk / Edge Finder launcher
REM Stable local launcher for Streamlit using 00_Home.py
REM =========================================================

set "ROOT1=C:\Projects\project files\edge-finder"
set "ROOT2=C:\Projects\edge-finder"

if exist "%ROOT1%\serving_ui_recovered\app\00_Home.py" (
  cd /d "%ROOT1%"
) else if exist "%ROOT2%\serving_ui_recovered\app\00_Home.py" (
  cd /d "%ROOT2%"
) else (
  echo [ERROR] Could not find project root.
  echo Checked:
  echo   %ROOT1%
  echo   %ROOT2%
  pause
  exit /b 1
)

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

set "ENTRY=serving_ui_recovered\app\00_Home.py"
if not exist "%ENTRY%" (
  echo [ERROR] Entrypoint not found: %CD%\%ENTRY%
  pause
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found on PATH.
  pause
  exit /b 1
)

for /f "delims=" %%I in ('where python') do (
  set "PY_EXE=%%I"
  goto :gotpython
)
:gotpython

for /f "usebackq delims=" %%P in (`python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"`) do set "PORT=%%P"

if "%PORT%"=="" (
  echo [WARN] Could not detect a free port. Falling back to 8501.
  set "PORT=8501"
)

set "ADDR=http://127.0.0.1:%PORT%"

echo.
echo ========================================================
echo   Launching Calculated Risk / Edge Finder
echo   Root   : %CD%
echo   Entry  : %CD%\%ENTRY%
echo   Python : %PY_EXE%
echo   Port   : %PORT%
echo   URL    : %ADDR%
echo ========================================================
echo.

start "" "%ADDR%"
python -m streamlit run "%ENTRY%" --server.port %PORT% --server.address 127.0.0.1

set "EC=%ERRORLEVEL%"
echo.
if not "%EC%"=="0" (
  echo [WARN] Streamlit exited with code %EC%
)

pause
endlocal

@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   SEO Spider - Local Install
echo   Built by Raul @ MaleBasics Corp
echo ============================================
echo.

:: Set paths
set "SCRIPT_DIR=%~dp0"
set "PYTHON_DIR=%SCRIPT_DIR%python_portable"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PIP_EXE=%PYTHON_DIR%\Scripts\pip.exe"
set "STREAMLIT_EXE=%PYTHON_DIR%\Scripts\streamlit.exe"

:: Check if portable Python already exists
if exist "%PYTHON_EXE%" (
    echo [OK] Portable Python found.
    goto :install_deps
)

:: Check if system Python exists
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] System Python found.
    set "PYTHON_EXE=python"
    set "PIP_EXE=pip"
    set "STREAMLIT_EXE=streamlit"
    goto :install_deps
)

:: No Python found — download portable version
echo [INFO] Python not found. Downloading portable Python 3.12...
echo        This is a one-time download (~25 MB). Please wait...
echo.

:: Create temp directory
if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"

:: Download Python embeddable package
set "PY_URL=https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip"
set "PY_ZIP=%SCRIPT_DIR%python_portable.zip"

:: Try PowerShell download
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_ZIP%' }" 2>nul
if not exist "%PY_ZIP%" (
    echo ERROR: Failed to download Python. Please check your internet connection.
    echo Alternatively, install Python manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Downloaded Python. Extracting...
powershell -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del "%PY_ZIP%"

:: Enable pip in embeddable Python by uncommenting import site in python312._pth
set "PTH_FILE=%PYTHON_DIR%\python312._pth"
if exist "%PTH_FILE%" (
    powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site','import site' | Set-Content '%PTH_FILE%'"
)

:: Download and install pip
echo [INFO] Installing pip...
set "GET_PIP=%PYTHON_DIR%\get-pip.py"
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GET_PIP%' }"
"%PYTHON_EXE%" "%GET_PIP%" --no-warn-script-location >nul 2>&1
del "%GET_PIP%"

if not exist "%PIP_EXE%" (
    echo ERROR: pip installation failed.
    pause
    exit /b 1
)
echo [OK] Portable Python 3.12 is ready!
echo.

:install_deps
echo [1/2] Installing dependencies (first run may take a minute)...
"%PIP_EXE%" install -r "%SCRIPT_DIR%requirements.txt" --no-warn-script-location -q
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.
echo.

echo [2/2] Starting SEO Spider...
echo.
echo ============================================
echo   Open your browser to:
echo   http://localhost:8501
echo ============================================
echo   Press Ctrl+C to stop the server.
echo ============================================
echo.
"%STREAMLIT_EXE%" run "%SCRIPT_DIR%app.py" --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause

@echo off
echo ============================================
echo   SEO Spider - Local Install
echo   Built by Raul @ MaleBasics Corp
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Download it from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/2] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting SEO Spider...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo.
streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause

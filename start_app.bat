@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo Starting Smart Attendance System...
echo Version: arcface-guided-register-v5-2026-04-26
echo.

REM --- Check and install optional packages (insightface, onnxruntime) ---
echo Checking optional AI packages (insightface, onnxruntime)...
python -c "import insightface" >nul 2>&1
if errorlevel 1 (
    echo insightface missing. Installing...
    pip install insightface
) else (
    echo insightface OK.
)

python -c "import onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo onnxruntime missing. Installing...
    pip install onnxruntime
) else (
    echo onnxruntime OK.
)

REM Re-verify after install
python -c "import insightface, onnxruntime" >nul 2>&1
if errorlevel 1 (
    python -c "import onnxruntime" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo WARNING: ArcFace packages install nahi ho paye. App fallback mode mein chalegi.
        echo Baad mein manually run karein: pip install insightface onnxruntime
        echo.
    ) else (
        echo.
        echo NOTE: insightface install nahi hua Python 3.13 pe, par onnxruntime OK hai.
        echo App ArcFace ONNX mode mein chalegi - background se independent verification hoga.
        echo.
    )
) else (
    echo ArcFace packages verified OK.
)
echo.

set "APP_PORT=5000"
netstat -ano | findstr ":5000" | findstr "LISTENING" >nul
if not errorlevel 1 (
    set "APP_PORT=5001"
    echo Port 5000 is already running by an old server.
    echo Starting the fixed app on port 5001 instead.
    echo.
)
echo Checking app runtime...
powershell -ExecutionPolicy Bypass -NoExit -Command "$env:PYTHONPATH = '%cd%\.vendor312;' + $env:PYTHONPATH; Set-Location -LiteralPath '%cd%'; python bootstrap_runtime.py; if ($LASTEXITCODE -ne 0) { Write-Host ''; Write-Host 'App start nahi hui kyunki required packages ready nahi hain.'; Write-Host 'Bootstrap message ko follow karke dependency issue fix karein, phir start_app.bat dubara run karein.'; exit $LASTEXITCODE }; $env:PORT='%APP_PORT%'; Start-Job -ScriptBlock { param($port) for ($i = 0; $i -lt 60; $i++) { Start-Sleep -Seconds 1; $listening = netstat -ano | Select-String (':{0} ' -f $port) | Select-String 'LISTENING'; if ($listening) { Start-Process ('http://127.0.0.1:' + $port); break } } } -ArgumentList '%APP_PORT%' | Out-Null; Write-Host 'Runtime ready. App start ho rahi hai...'; python app.py"

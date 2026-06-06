@echo off
:: Multi-Strategy Dry-Run: RegimeAdaptive + MeanRevConfluence + TrendComposite
:: 3 processes, 4 pairs each (BTC/ETH/SOL/SPX), separate DBs and API ports
:: Telegram only on RegimeAdaptive (primary)

cd /d D:\Private\freqtrade
set PYTHON=C:\Users\khieuvm\AppData\Local\Programs\Python\Python312\python.exe

if "%1"=="stop" (
    echo Stopping all bots...
    taskkill /F /IM python.exe 2>nul
    echo Done.
    exit /b
)

if "%1"=="status" (
    echo === Bot Processes ===
    tasklist /FI "IMAGENAME eq python.exe" 2>nul | find "python"
    if %errorlevel% neq 0 echo [STOPPED] No bots running
    exit /b
)

echo === Starting 3 Strategy Bots ===
echo.

echo [1/3] Regime_Adaptive (ComboH, port 8081, Telegram ON)...
start /min "Regime_Adaptive" %PYTHON% ft_run.py trade --config config_regime.json --logfile logs/regime.log --db-url sqlite:///regime_dry.sqlite

timeout /t 5 /nobreak >nul

echo [2/3] MeanRev_Confluence (ComboG, port 8082)...
start /min "MeanRev_Confluence" %PYTHON% ft_run.py trade --config config_meanrev.json --logfile logs/meanrev.log --db-url sqlite:///meanrev_dry.sqlite

timeout /t 5 /nobreak >nul

echo [3/3] Trend_Composite (ComboM, port 8083)...
start /min "Trend_Composite" %PYTHON% ft_run.py trade --config config_trend.json --logfile logs/trend.log --db-url sqlite:///trend_dry.sqlite

echo.
echo All 3 bots started!
echo   Regime_Adaptive:    http://127.0.0.1:8081  (Telegram ON)
echo   MeanRev_Confluence: http://127.0.0.1:8082
echo   Trend_Composite:    http://127.0.0.1:8083
echo.
echo Logs: logs/regime.log, logs/meanrev.log, logs/trend.log
echo Use "run_all.bat stop" to stop all.

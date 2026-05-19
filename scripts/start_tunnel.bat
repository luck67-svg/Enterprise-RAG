@echo off
REM =====================================================================
REM  start_tunnel.bat - One-click startup: Docker + Remote Ollama/Reranker
REM  Prerequisite: run deploy_reranker_remote.sh once to deploy ~/reranker_service.py
REM =====================================================================

REM --- Remote connection config ---
set REMOTE_USER=veridian
set REMOTE_HOST=192.168.144.129
set REMOTE_OLLAMA_PORT=11434
set REMOTE_RERANKER_PORT=8001
set LOCAL_OLLAMA_PORT=11434
set LOCAL_RERANKER_PORT=8001
set REMOTE_OLLAMA_BIN=/mnt/mydisk/home/veridian/ollama/bin/ollama
set REMOTE_VENV_PY=/mnt/mydisk/home/veridian/LJY/.venv/bin/python
set REMOTE_MODEL_PATH=/mnt/mydisk/home/veridian/LJY/bge-reranker-v2-m3

set SCRIPT_DIR=%~dp0
set PROJ_DIR=%SCRIPT_DIR%..
set SSH_KEY=%USERPROFILE%\.ssh\veridian
set SSH_OPTS=-i "%SSH_KEY%" -o BatchMode=yes -o StrictHostKeyChecking=no

REM --- Read model names from .env (lines starting with # are skipped) ---
set OLLAMA_MODEL=qwen3.5:35b
set EMBEDDING_MODEL=bge-m3
if exist "%PROJ_DIR%\.env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%PROJ_DIR%\.env") do (
        if /i "%%A"=="OLLAMA_MODEL"    set OLLAMA_MODEL=%%B
        if /i "%%A"=="EMBEDDING_MODEL" set EMBEDDING_MODEL=%%B
    )
)

if not exist "%PROJ_DIR%\logs" mkdir "%PROJ_DIR%\logs"

REM =====================================================================
REM  [1/5] Docker (Qdrant + Open WebUI)
REM =====================================================================
echo.
echo [1/5] Starting Docker services...
docker compose -f "%SCRIPT_DIR%docker-compose.yml" up -d
if errorlevel 1 echo       [WARN] docker compose failed or already running
echo       Qdrant:      http://localhost:6333
echo       Open WebUI:  http://localhost:3000

REM =====================================================================
REM  [2/5] Remote Ollama
REM =====================================================================
echo.
echo [2/5] Checking remote Ollama (%REMOTE_HOST%:%REMOTE_OLLAMA_PORT%)...
ssh %SSH_OPTS% %REMOTE_USER%@%REMOTE_HOST% "curl -sf http://localhost:%REMOTE_OLLAMA_PORT%/api/tags" >nul 2>&1
if errorlevel 1 (
    echo       Not running - starting Ollama...
    ssh %SSH_OPTS% %REMOTE_USER%@%REMOTE_HOST% "nohup %REMOTE_OLLAMA_BIN% serve </dev/null >>/tmp/ollama.log 2>&1 & disown && echo ok"
    timeout /t 5 /nobreak >nul
    ssh %SSH_OPTS% %REMOTE_USER%@%REMOTE_HOST% "curl -sf http://localhost:%REMOTE_OLLAMA_PORT%/api/tags" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Ollama failed to start. Check remote: tail /tmp/ollama.log
        pause & exit /b 1
    )
    echo       Ollama started OK
) else (
    echo       Ollama already running
)
echo       Tunnel: localhost:%LOCAL_OLLAMA_PORT% -^> remote:%REMOTE_OLLAMA_PORT%
netstat -ano | findstr ":%LOCAL_OLLAMA_PORT% " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start /b cmd /c "ssh %SSH_OPTS% -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -NL %LOCAL_OLLAMA_PORT%:localhost:%REMOTE_OLLAMA_PORT% %REMOTE_USER%@%REMOTE_HOST% 2>nul"
) else (
    echo       Ollama tunnel already active on port %LOCAL_OLLAMA_PORT%
)

REM =====================================================================
REM  [3/5] Remote Reranker  (model: bge-reranker-v2-m3)
REM =====================================================================
echo.
echo [3/5] Checking remote Reranker (%REMOTE_HOST%:%REMOTE_RERANKER_PORT%)...
ssh %SSH_OPTS% %REMOTE_USER%@%REMOTE_HOST% "curl -sf http://localhost:%REMOTE_RERANKER_PORT%/health" >nul 2>&1
if errorlevel 1 (
    echo       Not running - starting with venv python...
    ssh %SSH_OPTS% %REMOTE_USER%@%REMOTE_HOST% "cd ~ && RERANKER_MODEL_PATH=%REMOTE_MODEL_PATH% nohup %REMOTE_VENV_PY% -m uvicorn reranker_service:app --host 0.0.0.0 --port %REMOTE_RERANKER_PORT% </dev/null >>~/reranker.log 2>&1 & disown && echo ok"
    timeout /t 10 /nobreak >nul
) else (
    echo       Reranker already running
)
echo       Tunnel: localhost:%LOCAL_RERANKER_PORT% -^> remote:%REMOTE_RERANKER_PORT%
netstat -ano | findstr ":%LOCAL_RERANKER_PORT% " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start /b cmd /c "ssh %SSH_OPTS% -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -NL %LOCAL_RERANKER_PORT%:localhost:%REMOTE_RERANKER_PORT% %REMOTE_USER%@%REMOTE_HOST% 2>nul"
) else (
    echo       Reranker tunnel already active on port %LOCAL_RERANKER_PORT%
)

REM =====================================================================
REM  [4/5] Verify tunnels + warm up models
REM =====================================================================
echo.
echo [4/5] Verifying tunnels and warming up models...
timeout /t 5 /nobreak >nul

curl -sf http://localhost:%LOCAL_OLLAMA_PORT%/api/tags >nul 2>&1
if errorlevel 1 (
    echo       [WARN] Ollama tunnel not ready - recreating...
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%LOCAL_OLLAMA_PORT% " ^| findstr "LISTENING"') do taskkill /f /pid %%P >nul 2>&1
    start /b cmd /c "ssh %SSH_OPTS% -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -NL %LOCAL_OLLAMA_PORT%:localhost:%REMOTE_OLLAMA_PORT% %REMOTE_USER%@%REMOTE_HOST% 2>nul"
    timeout /t 4 /nobreak >nul
    curl -sf http://localhost:%LOCAL_OLLAMA_PORT%/api/tags >nul 2>&1
    if errorlevel 1 (echo       [WARN] Ollama tunnel still not ready) else (echo       Ollama tunnel OK (recovered))
) else (
    echo       Ollama tunnel OK
)

curl -sf http://localhost:%LOCAL_RERANKER_PORT%/health >nul 2>&1
if errorlevel 1 (
    echo       [WARN] Reranker tunnel not ready - recreating...
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%LOCAL_RERANKER_PORT% " ^| findstr "LISTENING"') do taskkill /f /pid %%P >nul 2>&1
    start /b cmd /c "ssh %SSH_OPTS% -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -NL %LOCAL_RERANKER_PORT%:localhost:%REMOTE_RERANKER_PORT% %REMOTE_USER%@%REMOTE_HOST% 2>nul"
    timeout /t 4 /nobreak >nul
    curl -sf http://localhost:%LOCAL_RERANKER_PORT%/health >nul 2>&1
    if errorlevel 1 (echo       [WARN] Reranker tunnel still not ready) else (echo       Reranker tunnel OK (recovered))
) else (
    echo       Reranker tunnel OK
)

REM Background warmup: pre-load models into memory so first API call is fast
start /b powershell -NoProfile -NonInteractive -WindowStyle Hidden -Command "Start-Sleep 5; try { Invoke-RestMethod -Method POST -Uri 'http://localhost:%LOCAL_OLLAMA_PORT%/api/generate' -Body ('{""model"":""%EMBEDDING_MODEL%"",""prompt"":"""",""stream"":false,""keep_alive"":""30m""}') -ContentType 'application/json' | Out-Null; Write-Host '[warm] %EMBEDDING_MODEL% ready' } catch { Write-Host '[warm] %EMBEDDING_MODEL% unreachable' }"
start /b powershell -NoProfile -NonInteractive -WindowStyle Hidden -Command "Start-Sleep 5; try { Invoke-RestMethod -Method POST -Uri 'http://localhost:%LOCAL_OLLAMA_PORT%/api/generate' -Body ('{""model"":""%OLLAMA_MODEL%"",""prompt"":"""",""stream"":false,""keep_alive"":""30m""}') -ContentType 'application/json' | Out-Null; Write-Host '[warm] %OLLAMA_MODEL% ready' } catch { Write-Host '[warm] %OLLAMA_MODEL% unreachable' }"
echo       Models warming up in background: %EMBEDDING_MODEL%, %OLLAMA_MODEL%

REM =====================================================================
REM  [5/5] FastAPI
REM =====================================================================
echo.
echo [5/5] Starting FastAPI...
echo       API:  http://localhost:8000
echo       Docs: http://localhost:8000/docs
cd /d "%PROJ_DIR%"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

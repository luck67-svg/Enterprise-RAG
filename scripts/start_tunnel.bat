@echo off
REM 项目一键启动脚本：Docker 容器 + Ollama + FastAPI
REM Usage:
REM   scripts\start_tunnel.bat           默认使用远程 Ollama
REM   scripts\start_tunnel.bat --local   使用本地 Ollama

if "%REMOTE_SSH_USER%"=="" set REMOTE_SSH_USER=veridian
if "%REMOTE_SSH_HOST%"=="" set REMOTE_SSH_HOST=192.168.144.129
if "%REMOTE_OLLAMA_PORT%"=="" set REMOTE_OLLAMA_PORT=11434
if "%LOCAL_TUNNEL_PORT%"=="" set LOCAL_TUNNEL_PORT=11434

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set SSH_OPTS=-i "%USERPROFILE%\.ssh\veridian" -o BatchMode=yes
set MODE=%1

if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

REM ---------- 1. 启动 Docker 容器（Qdrant + Open WebUI） ----------
echo === 启动 Docker 容器...
docker compose -f "%SCRIPT_DIR%docker-compose.yml" up -d
echo     Qdrant:     http://localhost:6333
echo     Open WebUI: http://localhost:3000

REM ---------- 2. 启动 Ollama ----------
if "%MODE%"=="--local" goto :local_ollama

REM ----- 远程模式 -----
echo === [远程模式] 检查远程 Ollama 状态...
ssh %SSH_OPTS% %REMOTE_SSH_USER%@%REMOTE_SSH_HOST% "curl -sf http://localhost:%REMOTE_OLLAMA_PORT%/api/tags >/dev/null 2>&1 && echo Ollama已在运行 || (echo 正在启动Ollama... && nohup /mnt/mydisk/home/veridian/ollama/bin/ollama serve > /tmp/ollama.log 2>&1 & sleep 3 && echo Ollama启动完成)"

if errorlevel 1 (
    echo [ERROR] SSH 连接失败
    pause
    exit /b 1
)

echo === 建立 SSH 隧道: localhost:%LOCAL_TUNNEL_PORT% -^> %REMOTE_SSH_HOST%:%REMOTE_OLLAMA_PORT%
start /b ssh %SSH_OPTS% -NL %LOCAL_TUNNEL_PORT%:localhost:%REMOTE_OLLAMA_PORT% %REMOTE_SSH_USER%@%REMOTE_SSH_HOST%
echo     隧道已在后台运行

echo === 同步远程 Ollama 日志 -^> logs\ollama.log
start /b ssh %SSH_OPTS% %REMOTE_SSH_USER%@%REMOTE_SSH_HOST% "tail -f /tmp/ollama.log" > "%PROJECT_DIR%\logs\ollama.log" 2>&1

echo === 检查远程 Reranker 服务状态...
ssh %SSH_OPTS% %REMOTE_SSH_USER%@%REMOTE_SSH_HOST% "curl -sf http://localhost:8001/health >/dev/null 2>&1 && echo Reranker已在运行 || (echo 正在启动Reranker... && cd ~ && nohup uvicorn reranker_service:app --host 0.0.0.0 --port 8001 > ~/reranker.log 2>&1 & for i in 1 2 3 4 5 6 7 8 9 10; do sleep 2 && curl -sf http://localhost:8001/health >/dev/null 2>&1 && echo Reranker启动完成 && break || echo 等待Reranker... ; done)"

echo === 建立 Reranker 隧道: localhost:8001 -^> %REMOTE_SSH_HOST%:8001
start /b ssh %SSH_OPTS% -NL 8001:localhost:8001 %REMOTE_SSH_USER%@%REMOTE_SSH_HOST%
echo     Reranker 隧道已在后台运行

goto :start_fastapi

:local_ollama
REM ----- 本地模式 -----
echo === [本地模式] 检查本地 Ollama 状态...
curl -sf http://localhost:%LOCAL_TUNNEL_PORT%/api/tags >nul 2>&1
if not errorlevel 1 (
    echo     Ollama 已在运行
) else (
    echo     正在启动本地 Ollama...
    start /b ollama serve > "%PROJECT_DIR%\logs\ollama.log" 2>&1
    timeout /t 3 /nobreak >nul
    curl -sf http://localhost:%LOCAL_TUNNEL_PORT%/api/tags >nul 2>&1
    if errorlevel 1 (
        echo     Ollama 启动失败，请检查 logs\ollama.log
        pause
        exit /b 1
    )
    echo     Ollama 启动成功
)

:start_fastapi
REM ---------- 3. 启动 FastAPI 服务 ----------
echo === 启动 FastAPI 服务...
echo     API:  http://localhost:8000
echo     Docs: http://localhost:8000/docs
echo     Ollama 日志: logs\ollama.log
echo     FastAPI 日志: logs\fastapi.log
cd /d "%PROJECT_DIR%"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --use-colors 2>&1 | powershell -Command "$input | ForEach-Object { $_; ($_ -replace '\x1b\[[0-9;]*m','') | Out-File -Append -Encoding utf8 logs\fastapi.log }"

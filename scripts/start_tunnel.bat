@echo off
REM 项目一键启动脚本：Docker 容器 + 远程 Ollama + SSH 隧道 + FastAPI
REM Usage: scripts\start_tunnel.bat

if "%REMOTE_SSH_USER%"=="" set REMOTE_SSH_USER=veridian
if "%REMOTE_SSH_HOST%"=="" set REMOTE_SSH_HOST=192.168.144.129
if "%REMOTE_OLLAMA_PORT%"=="" set REMOTE_OLLAMA_PORT=11434
if "%LOCAL_TUNNEL_PORT%"=="" set LOCAL_TUNNEL_PORT=11434

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set SSH_OPTS=-i "%USERPROFILE%\.ssh\veridian" -o BatchMode=yes

REM ---------- 1. 启动 Docker 容器（Qdrant + Open WebUI） ----------
echo === 启动 Docker 容器...
docker compose -f "%SCRIPT_DIR%docker-compose.yml" up -d
echo     Qdrant:     http://localhost:6333
echo     Open WebUI: http://localhost:3000

REM ---------- 2. 在远程服务器上启动 Ollama ----------
echo === 检查远程 Ollama 状态...
ssh %SSH_OPTS% %REMOTE_SSH_USER%@%REMOTE_SSH_HOST% "curl -sf http://localhost:%REMOTE_OLLAMA_PORT%/api/tags >/dev/null 2>&1 && echo Ollama已在运行 || (echo 正在启动Ollama... && nohup /mnt/mydisk/home/veridian/ollama/bin/ollama serve > /tmp/ollama.log 2>&1 & sleep 3 && echo Ollama启动完成)"

if errorlevel 1 (
    echo [ERROR] SSH 连接失败
    pause
    exit /b 1
)

REM ---------- 3. 建立 SSH 端口转发（后台） ----------
echo === 建立 SSH 隧道: localhost:%LOCAL_TUNNEL_PORT% -^> %REMOTE_SSH_HOST%:%REMOTE_OLLAMA_PORT%
start /b ssh %SSH_OPTS% -NL %LOCAL_TUNNEL_PORT%:localhost:%REMOTE_OLLAMA_PORT% %REMOTE_SSH_USER%@%REMOTE_SSH_HOST%
echo     隧道已在后台运行

REM ---------- 4. 同步远程 Ollama 日志到本地（后台） ----------
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"
echo === 同步远程 Ollama 日志 -^> logs\ollama.log
start /b ssh %SSH_OPTS% %REMOTE_SSH_USER%@%REMOTE_SSH_HOST% "tail -f /tmp/ollama.log" > "%PROJECT_DIR%\logs\ollama.log" 2>&1

REM ---------- 5. 启动 FastAPI 服务 ----------
echo === 启动 FastAPI 服务...
echo     API:  http://localhost:8000
echo     Docs: http://localhost:8000/docs
echo     Ollama 日志: logs\ollama.log
echo     FastAPI 日志: logs\fastapi.log
cd /d "%PROJECT_DIR%"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --use-colors 2>&1 | powershell -Command "$input | ForEach-Object { $_; ($_ -replace '\x1b\[[0-9;]*m','') | Out-File -Append -Encoding utf8 logs\fastapi.log }"

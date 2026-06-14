@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo ================================================
echo 智能审计 Agent 平台启动脚本
echo ================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+ 并加入 PATH。
    pause
    exit /b 1
)

set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    if exist "venv\Scripts\activate.bat" (
        set "VENV_DIR=venv"
    ) else (
        echo [信息] 创建虚拟环境 .venv ...
        python -m venv .venv
        if errorlevel 1 (
            echo [错误] 虚拟环境创建失败。
            pause
            exit /b 1
        )
    )
)

echo [信息] 使用虚拟环境: %VENV_DIR%
call "%VENV_DIR%\Scripts\activate.bat"

python -c "import fastapi, uvicorn, langchain_openai, sklearn" >nul 2>&1
if errorlevel 1 (
    echo [信息] 安装或补齐依赖 ...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络后重试。
        pause
        exit /b 1
    )
)

if not exist "config.env" (
    echo [信息] 创建 config.env ...
    copy config.env.example config.env >nul
)

set "WEB_HOST=127.0.0.1"
if "%WEB_PORT%"=="" set "WEB_PORT=8000"

echo ================================================
echo 正在启动服务，请保持此窗口打开。
echo 访问地址: http://127.0.0.1:%WEB_PORT%
echo 停止服务: 在此窗口按 Ctrl+C
echo ================================================

start "" "http://127.0.0.1:%WEB_PORT%"
python start.py

pause

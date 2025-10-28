@echo off
chcp 65001 >nul
echo ================================================
echo 智能审计决策系统启动脚本
echo ================================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [信息] Python环境检查通过

REM 检查虚拟环境
if not exist "venv" (
    echo [信息] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
)

REM 激活虚拟环境
echo [信息] 激活虚拟环境...
call venv\Scripts\activate.bat

REM 检查关键依赖
echo [信息] 检查依赖包...
pip list | findstr "fastapi" >nul
if errorlevel 1 (
    echo [信息] 安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [警告] 依赖包安装失败，请手动安装: pip install -r requirements.txt
    )
)

REM 检查配置文件
if not exist "config.env" (
    echo [信息] 创建配置文件...
    if exist "config.env.example" (
        copy config.env.example config.env
    ) else (
        echo MYSQL_HOST=localhost > config.env
        echo MYSQL_PORT=3306 >> config.env
        echo MYSQL_USER=root >> config.env
        echo MYSQL_PASSWORD=123456 >> config.env
        echo MYSQL_DATABASE=audit_system >> config.env
        echo NEO4J_URI=bolt://localhost:7687 >> config.env
        echo NEO4J_USER=neo4j >> config.env
        echo NEO4J_PASSWORD=12345678 >> config.env
        echo QWEN_API_KEY=sk-484fb339d2274307b3aa3fd6400964ae >> config.env
    )
    echo [信息] 请编辑config.env文件，配置数据库和API信息
    pause
)

REM 启动系统
echo ================================================
echo 正在启动智能审计决策系统...
echo 访问地址: http://localhost:8000
echo 按 Ctrl+C 可停止系统
echo ================================================
python start.py

pause

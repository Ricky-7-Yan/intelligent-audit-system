#!/bin/bash

echo "================================================"
echo "智能审计决策系统启动脚本"
echo "================================================"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查Python版本
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "[错误] Python版本过低，需要3.8+，当前版本: $python_version"
    exit 1
fi

echo "[信息] Python环境检查通过"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "[信息] 创建虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[错误] 虚拟环境创建失败"
        exit 1
    fi
fi

# 激活虚拟环境
echo "[信息] 激活虚拟环境..."
source venv/bin/activate

# 检查关键依赖
echo "[信息] 检查依赖包..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "[信息] 安装依赖包..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[警告] 依赖包安装失败，请手动安装: pip install -r requirements.txt"
    fi
fi

# 检查配置文件
if [ ! -f "config.env" ]; then
    echo "[信息] 创建配置文件..."
    if [ -f "config.env.example" ]; then
        cp config.env.example config.env
    else
        cat > config.env << EOF
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DATABASE=audit_system
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
QWEN_API_KEY=sk-484fb339d2274307b3aa3fd6400964ae
EOF
    fi
    echo "[信息] 请编辑config.env文件，配置数据库和API信息"
    read -p "按回车键继续..."
fi

# 启动系统
echo "================================================"
echo "正在启动智能审计决策系统..."
echo "访问地址: http://localhost:8000"
echo "按 Ctrl+C 可停止系统"
echo "================================================"
python3 start.py

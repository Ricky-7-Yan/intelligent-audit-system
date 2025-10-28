# 🚀 PyCharm项目配置指南

> 详细的PyCharm开发环境配置教程，助你快速上手智能审计决策系统

---

## 📖 目录

- [环境配置](#-环境配置)
- [数据库配置](#-数据库配置)
- [运行项目](#-运行项目)
- [调试配置](#-调试配置)
- [常见问题](#-常见问题)

---

## 🎯 环境配置

### 1️⃣ 创建PyCharm项目

1. 📂 打开PyCharm
2. 选择 **File** → **Open**
3. 选择项目目录 `intelligent-audit-system`
4. ✅ 选择Python解释器（建议Python 3.8+）

### 2️⃣ 配置Python解释器

1. 🔧 打开 `File` → `Settings` → `Project: intelligent-audit-system` → `Python Interpreter`
2. ➕ 点击齿轮图标 → `Add`
3. 🐍 选择 `Virtualenv Environment` → `New environment`
4. 📍 设置虚拟环境位置：`项目根目录/venv`
5. ✅ 选择Python版本（3.8+）

### 3️⃣ 安装项目依赖

在PyCharm终端中执行：

```bash
# 激活虚拟环境
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装spaCy英文模型
python -m spacy download en_core_web_sm
```

### 4️⃣ 配置环境变量

创建 `config.env` 文件：

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DATABASE=audit_system

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678

QWEN_API_KEY=sk-484fb339d2274307b3aa3fd6400964ae
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

WEB_HOST=0.0.0.0
WEB_PORT=8000
```

### 5️⃣ 配置运行配置

#### 🎬 主应用运行配置

1. 点击 `Run` → `Edit Configurations`
2. ➕ 点击 `+` → `Python`
3. 🔧 配置如下：
   - **Name**: `智能审计系统`
   - **Script path**: `web/main.py`
   - **Working directory**: `项目根目录`
   - **Python interpreter**: `项目虚拟环境解释器`

#### 🗄️ 数据库初始化配置

创建新配置：
- **Name**: `📊 数据库初始化`
- **Script path**: `database/init_db.py`
- **Working directory**: `项目根目录`

#### 🕸️ Neo4j初始化配置

创建新配置：
- **Name**: `🕸️ Neo4j初始化`
- **Script path**: `knowledge_graph/neo4j_init.py`
- **Working directory**: `项目根目录`

### 6️⃣ 配置代码检查

1. 🔍 打开 `File` → `Settings` → `Editor` → `Inspections`
2. ✅ 启用以下检查：
   - Python
   - PEP 8
   - Type checking
   - Security

### 7️⃣ 配置代码格式化

1. 📝 打开 `File` → `Settings` → `Editor` → `Code Style` → `Python`
2. ➡️ 设置缩进为4个空格
3. ✅ 启用 `Use tab character` 选项

---

## 🗄️ 数据库配置

### 📊 MySQL配置

1. 📥 下载并安装MySQL 8.0+
2. ▶️ 启动MySQL服务
3. 📝 创建数据库：

```sql
CREATE DATABASE audit_system 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;
```

4. 🚀 运行数据库初始化：
```bash
python database/init_db.py
```

### 🕸️ Neo4j配置

1. 📥 下载并安装Neo4j Desktop
2. ➕ 创建新项目
3. ▶️ 启动数据库
4. 🔐 设置密码：`12345678`
5. 🚀 运行初始化：
```bash
python knowledge_graph/neo4j_init.py
```

---

## 🏃 运行项目

### 1️⃣ 初始化数据库

```bash
# 在PyCharm终端中执行
python database/init_db.py
python knowledge_graph/neo4j_init.py
```

### 2️⃣ 启动Web服务

**方式一：点击运行按钮** ▶️

**方式二：终端命令**
```bash
python web/main.py
```

### 3️⃣ 访问系统

🌐 打开浏览器访问: **http://localhost:8000**

---

## 🐛 调试配置

### 1️⃣ 设置断点

🟡 在代码行号左侧点击设置断点

### 2️⃣ 调试运行

1. 右键点击 `web/main.py`
2. 选择 `Debug 'main'`
3. 🐛 或点击调试按钮（绿色虫子图标）

### 3️⃣ 调试控制台

📊 使用调试控制台查看变量值和执行表达式

---

## ⌨️ 常用快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + Shift + F10` | ▶️ 运行当前文件 |
| `Shift + F10` | ▶️ 运行 |
| `Shift + F9` | 🐛 调试 |
| `Ctrl + F` | 🔍 查找 |
| `Ctrl + R` | 🔄 替换 |
| `Ctrl + /` | 💬 注释/取消注释 |

---

## ⚡ 性能优化

### 1️⃣ 内存设置

打开 `Help` → `Edit Custom VM Options`，添加：

```
-Xmx4g
-Xms2g
-XX:MaxMetaspaceSize=512m
```

### 2️⃣ 推荐插件

建议安装以下插件：
- 🐍 Python
- 🗄️ Database Tools and SQL
- 🔀 Git Integration
- 📝 Markdown
- 🌐 REST Client

---

## ❓ 常见问题

### ❓ 导入错误

**问题**: 模块导入失败  
**✅ 解决**: 
- 检查Python解释器配置
- 确保虚拟环境已激活
- 重新安装依赖包

### ❓ 数据库连接失败

**问题**: 无法连接MySQL或Neo4j  
**✅ 解决**:
- 检查服务是否启动
- 验证连接配置
- 检查防火墙设置

### ❓ 模型下载失败

**问题**: HuggingFace模型下载超时  
**✅ 解决**:
- 使用镜像站: `export HF_ENDPOINT=https://hf-mirror.com`
- 或手动下载模型到 `./models/` 目录

---

## 💡 开发建议

### 📝 代码规范

- 使用PEP 8代码风格
- 添加类型注解
- 编写清晰的注释

### 🧪 测试

- 为每个模块编写单元测试
- 使用pytest进行测试
- 保持测试覆盖率

### 🔀 Git工作流

```bash
# 创建新分支
git checkout -b feature/new-feature

# 提交更改
git add .
git commit -m "Add new feature"

# 推送更改
git push origin feature/new-feature
```

---

<div align="center">

**📚 更多文档**

[返回首页](README.md) • [项目总结](Project_Summary.md) • [模型下载](下载模型说明.md)

Made with ❤️ by Intelligent Audit Team

</div>

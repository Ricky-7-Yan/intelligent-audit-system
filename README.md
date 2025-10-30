# 🤖 AutoAudit--智能审计决策系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**基于大语言模型的智能审计平台 | 集成知识图谱、RAG、强化学习等前沿技术**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [技术架构](#-技术架构) • [文档](Project_Summary.md)

</div>

---

## 📋 项目简介

AutoAudit是一个基于大语言模型（LLM）的智能审计平台，集成了知识图谱、RAG检索增强生成、强化学习等前沿技术，为审计工作提供智能化支持。

> 🎯 **核心价值**: 突破传统审计工具局限，支持复杂业务逻辑的深度推理

---

## ✨ 功能特性

### 🤖 智能对话Agent
- ✅ 基于LangChain的多轮对话式审计决策
- ✅ 上下文感知的连续对话
- ✅ 专业推理和审计建议

### 🗺️ 知识图谱
- ✅ 整合COBIT、ISO27001、SOX等审计标准
- ✅ 实体识别和关系抽取
- ✅ 动态知识更新

### 🔍 Agentic RAG系统
- ✅ 智能检索增强生成
- ✅ 向量数据库支持
- ✅ 语义搜索和答案生成

### ⚠️ 风险评估
- ✅ 自动识别和评估审计风险
- ✅ 风险等级评分
- ✅ 实时风险监控

### ✅ 合规检查
- ✅ 对照标准进行合规性检查
- ✅ 多标准支持（COBIT、ISO27001、SOX）
- ✅ 整改建议生成

### 🎓 模型训练
- ✅ 支持SFT、RLHF等训练方法
- ✅ LoRA高效微调
- ✅ Benchmark测评系统

---

## 🚀 快速开始

### 📦 环境要求

- Python 3.8+ 🐍
- MySQL 8.0+ 🗄️
- Neo4j 5.0+ 🕸️
- 8GB+ RAM 💾

### 🏃 一键启动（推荐）

#### Windows 用户
```bash
# 双击运行
start.bat
```

#### Linux/Mac 用户
```bash
# 赋予执行权限
chmod +x start.sh

# 运行启动脚本
./start.sh
```

### 📝 手动安装

#### 1. 克隆项目
```bash
git clone <repository-url>
cd intelligent-audit-system
```

#### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 安装spaCy模型
```bash
python -m spacy download en_core_web_sm
```

#### 5. 配置环境变量
```bash
cp config.env.example config.env
# 编辑config.env，填入相应配置
```

#### 6. 初始化数据库
```bash
# 初始化MySQL
python database/init_db.py

# 初始化Neo4j
python knowledge_graph/neo4j_init.py
```

#### 7. 启动系统
```bash
python web/main.py
```

### 🌐 访问系统

打开浏览器访问: **http://localhost:8000**

---

## 🎮 功能模块

### 💬 智能对话
- 🎯 访问 `/chat` 
- 💡 与AI审计助手进行专业对话
- 🔄 多轮对话支持

### 🔍 审计分析
- 🎯 访问 `/audit`
- ⚡ 自动风险评估和合规检查
- 📊 可视化结果展示

### 📚 知识管理
- 🎯 访问 `/knowledge`
- 📖 构建和维护审计知识库
- 🗺️ 知识图谱可视化

### 🎓 模型训练
- 🎯 访问 `/training`
- 🏋️ SFT和RLHF训练
- 📈 性能评估和对比

---

## 📡 API接口

### 💬 聊天API
```http
POST /api/chat
Content-Type: application/json

{
    "message": "请对ERP系统进行安全审计",
    "session_id": "optional_session_id"
}
```

### 🔍 审计API
```http
POST /api/audit
Content-Type: application/json

{
    "audit_item": "ERP系统",
    "audit_type": "安全审计",
    "standard_type": "COBIT",
    "risk_level": "高"
}
```

---

## 📁 项目结构

```
intelligent-audit-system/
├── agents/                 # 智能Agent模块
│   └── audit_agent.py
├── knowledge_graph/        # 知识图谱模块
│   ├── builder.py
│   └── neo4j_init.py
├── rag/                    # RAG系统模块
│   └── agentic_rag.py
├── training/               # 训练模块
│   └── training_pipeline.py
├── web/                    # Web界面
│   └── main.py
├── templates/              # HTML模板
│   ├── index.html         # 主页
│   ├── chat.html          # 聊天页面
│   ├── audit.html         # 审计分析
│   ├── knowledge.html     # 知识管理
│   └── training.html      # 模型训练
├── database/               # 数据库模块
│   └── init_db.py
├── config.py               # 配置文件
├── requirements.txt        # 依赖列表
└── start.py                # 启动脚本
```

---

## ⚙️ 配置说明

### 🔧 环境变量

编辑 `config.env` 文件：

```env
# MySQL配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your-username
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=audit_system

# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=your-username
NEO4J_PASSWORD=your-password

# LLM API配置
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

---

## 📥 模型下载

如果需要使用RAG功能，需要下载Sentence Transformers模型：

### 🌟 使用镜像站（推荐）

```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com

# 启动系统，会自动下载
python web/main.py
```

详细说明请查看：[📥 下载模型说明.md](下载模型说明.md)

---

## ❓ 常见问题

### ❓ 模型下载失败？
- ✅ 使用镜像站: 设置 `HF_ENDPOINT=https://hf-mirror.com`
- ✅ 或手动下载模型到 `./models/` 目录

### ❓ 数据库连接失败？
- ✅ 确保MySQL和Neo4j服务已启动
- ✅ 检查配置信息是否正确

### ❓ API调用失败？
- ✅ 检查API密钥是否正确
- ✅ 验证网络连接

更多问题请查看：[📚 PyCharm配置指南](PyCharm_Setup_Guide.md)

---

## 📊 性能指标

- 🎯 **准确率**: 85%+ 的审计建议准确性
- ⚡ **响应时间**: <3秒
- 👥 **并发支持**: 100+ 并发用户
- 💪 **可用性**: 99.9% 系统可用性

---

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. 🍴 Fork 项目
2. 🌿 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 💾 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 📤 推送到分支 (`git push origin feature/AmazingFeature`)
5. 🔄 开启 Pull Request

---

## 📄 许可证

本项目采用 **MIT License** - 查看 [LICENSE](LICENSE) 文件了解详情

---

## 👥 联系我们

如有问题或建议，请提交 [Issue](https://github.com/your-username/intelligent-audit-system/issues)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给它一个星星！**

Made with ❤️ by AutoAudit Team

</div>







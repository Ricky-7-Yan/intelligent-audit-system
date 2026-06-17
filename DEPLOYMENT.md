# 审脉 AuditPilot 部署说明

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.env.example config.env
python start.py
```

默认访问地址：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

## 配置项

生产环境建议在 `config.env` 或进程环境变量中配置：

```env
WEB_HOST=0.0.0.0
WEB_PORT=8000
CORS_ORIGINS=https://your-domain.example

LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

RAG_DISABLE_EMBEDDINGS=0
RAG_LIGHT_MODE=0
```

`config.env` 被 `.gitignore` 忽略，不要提交真实密钥。

## Docker 部署

```powershell
docker build -t auditpilot:latest .
docker run -d --name auditpilot -p 8000:8000 --env-file config.env auditpilot:latest
```

如需持久化运行数据：

```powershell
docker run -d --name auditpilot `
  -p 8000:8000 `
  --env-file config.env `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\logs:/app/logs `
  auditpilot:latest
```

## 公网访问

正式生产建议使用云服务器、容器平台或 PaaS，并通过 Nginx/Caddy/云负载均衡开启 HTTPS：

```text
Internet -> HTTPS Reverse Proxy -> 127.0.0.1:8000
```

没有云账号时，可用 Cloudflare Tunnel 或 ngrok 做临时公网演示：

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

临时隧道适合演示和验收，不等同于长期生产部署。正式上线需要绑定域名、配置 TLS、日志留存、密钥托管、访问控制和备份策略。

## 云平台迁移

项目已经包含通用生产部署文件：

- `Dockerfile`：容器镜像构建入口，支持云平台注入的 `PORT`。
- `.dockerignore`：排除本地密钥、运行数据、模型、日志和 node_modules。
- `render.yaml`：Render Blueprint。
- `railway.json`：Railway Docker 部署配置。
- `fly.toml`：Fly.io 部署配置。
- `Procfile`：Heroku/部分 PaaS 的 Web 进程入口。

### Render

1. 把仓库推送到 GitHub/GitLab。
2. 在 Render 选择 New Blueprint 或 Web Service。
3. 选择本仓库，Render 会读取 `render.yaml`。
4. 在环境变量中配置：
   - `DEEPSEEK_API_KEY`
   - `CORS_ORIGINS=https://你的域名或 Render 域名`
5. 部署完成后访问 Render 分配的 HTTPS 地址。

### Railway

```powershell
railway login
railway init
railway variables set DEEPSEEK_API_KEY=你的密钥
railway variables set CORS_ORIGINS=https://你的域名
railway up
```

Railway 会读取 `railway.json` 并使用 Dockerfile 构建。

### Fly.io

```powershell
flyctl auth login
flyctl launch --copy-config
flyctl secrets set DEEPSEEK_API_KEY=你的密钥 CORS_ORIGINS=https://你的域名
flyctl deploy
```

如应用名 `auditpilot` 被占用，修改 `fly.toml` 中的 `app`。

### 云服务器 / VPS

```bash
git clone <your-repo-url>
cd intelligent-audit-system
cp config.env.example config.env
# 编辑 config.env，填入 DEEPSEEK_API_KEY 和 CORS_ORIGINS
docker compose up -d --build
```

建议使用 Caddy 自动 HTTPS：

```caddyfile
auditpilot.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

## 上线前检查

```powershell
python -m py_compile web\main.py services\*.py
node --check static\app.js
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

确认公网地址：

```powershell
Invoke-RestMethod https://你的公网地址/api/health
```

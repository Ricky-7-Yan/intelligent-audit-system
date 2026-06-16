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

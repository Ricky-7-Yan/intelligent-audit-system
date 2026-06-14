# 部署说明

## 本机运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.env.example config.env
python start.py
```

访问 `http://localhost:8000`。

## 生产建议

- 在 `config.env` 中配置真实 `QWEN_API_KEY`；为空时系统会使用确定性降级回答。
- MySQL 和 Neo4j 是可选增强项；不配置密码时系统不会强连。
- `data/`、`logs/`、`models/` 属于运行时数据，应挂载持久化磁盘，不建议提交到 git。
- 对公网部署时建议使用 Nginx/Caddy 反向代理到 `127.0.0.1:8000`，并启用 HTTPS。

## 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

返回 `status: healthy` 即表示 Web 服务和核心降级链路可用。

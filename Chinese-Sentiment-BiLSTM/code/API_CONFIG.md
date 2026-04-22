# DeepSeek API 配置说明

## ✅ 当前配置

您的 DeepSeek API 配置已成功设置：

```bash
DEEPSEEK_API_KEY=sk-368cd8790f254db7bdb1fe6b8d6682da
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 📁 配置文件位置

所有配置文件都位于：`Chinese-Sentiment-BiLSTM\code\`

### 1. 主要配置文件

- **deepseek_api.py** - API 服务主程序
  - API Key 配置在第 30 行
  - Base URL 配置在第 31 行

### 2. 环境变量文件

- **.env_deepseek** - 环境变量配置模板
- **set_api_key.bat** - Windows 一键设置脚本

## 🚀 服务状态

### 后端服务
- **状态**: ✅ 运行中
- **地址**: http://localhost:5000
- **接口**:
  - `POST /api/analyze` - 单条文本分析
  - `POST /api/batch_analyze` - 批量文件分析
  - `GET /api/health` - 健康检查

### 前端页面
- **状态**: ✅ 可访问
- **地址**: http://localhost:8000/Chinese-Sentiment-BiLSTM/code/advanced_app.html
- **功能**: DeepSeek 云端智能舆情分析

## 🔧 如何修改配置

### 方法一：使用环境变量（推荐）

**Windows PowerShell**:
```powershell
$env:DEEPSEEK_API_KEY="sk-368cd8790f254db7bdb1fe6b8d6682da"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

**Linux/Mac**:
```bash
export DEEPSEEK_API_KEY="sk-368cd8790f254db7bdb1fe6b8d6682da"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

### 方法二：直接修改代码

编辑 `deepseek_api.py` 第 30-31 行：

```python
# 第 30 行：修改 API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "您的新 API Key")

# 第 31 行：修改 Base URL（通常不需要）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

### 方法三：使用设置脚本

**Windows**:
```bash
cd Chinese-Sentiment-BiLSTM\code
set_api_key.bat
```

## 🧪 测试配置

### 测试 1: 检查服务是否运行

```bash
curl http://localhost:5000/api/health
```

预期响应：
```json
{
  "status": "ok",
  "service": "DeepSeek 情感分析服务"
}
```

### 测试 2: 测试单条分析

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"这是我今年看过的最好的电影，特效炸裂，强烈推荐！\"}"
```

### 测试 3: 使用 Python 测试脚本

```bash
cd Chinese-Sentiment-BiLSTM\code
python test_deepseek_api.py
```

## 📝 常见问题

### Q1: API 调用失败怎么办？

**检查步骤**：
1. 确认 API Key 是否正确
2. 检查网络连接是否正常
3. 查看终端错误日志
4. 确认服务是否运行（访问 http://localhost:5000/api/health）

### Q2: 如何更换 API Key？

**方法**：
1. 修改 `deepseek_api.py` 第 30 行
2. 或设置环境变量 `DEEPSEEK_API_KEY`
3. 重启服务使配置生效

### Q3: 服务无法启动？

**解决方案**：
1. 检查端口 5000 是否被占用
2. 确认依赖包已安装：`pip install -r requirements_deepseek.txt`
3. 查看详细错误日志

### Q4: 如何查看当前配置？

**方法**：
```python
# 在 Python 中运行
import os
print("API Key:", os.getenv("DEEPSEEK_API_KEY", "未设置"))
print("Base URL:", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
```

## 🔗 相关资源

- **DeepSeek 官网**: https://platform.deepseek.com/
- **API 文档**: https://platform.deepseek.com/api-docs/
- **使用文档**: DEEPSEEK_USAGE.md
- **快速入门**: QUICKSTART.md

## 📊 配置信息总结

| 配置项 | 值 | 说明 |
|--------|-----|------|
| API Key | sk-368cd8790f254db7bdb1fe6b8d6682da | DeepSeek API 认证密钥 |
| Base URL | https://api.deepseek.com | DeepSeek API 服务端地址 |
| API 端点 | /chat/completions | 对话补全接口 |
| 模型名称 | deepseek-chat | 使用的模型 |
| 本地服务端口 | 5000 | Flask 服务端口 |
| 前端端口 | 8000 | HTTP 服务器端口 |

---

**最后更新**: 2026-04-15  
**版本**: 1.0.0

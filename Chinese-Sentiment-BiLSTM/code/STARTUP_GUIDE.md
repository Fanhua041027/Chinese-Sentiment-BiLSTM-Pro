# 🚀 情感分析服务启动指南

## ⚠️ 重要提示

**本地模型服务启动失败** - 正在加载模型文件（约需 10-15 秒）

## 🔧 问题诊断

### 当前状态
- ✅ **DeepSeek API 服务**（端口 5000）- 运行正常
- ⏳ **本地模型服务**（端口 5001）- 正在加载模型
- ✅ **Web 服务器**（端口 8000）- 运行正常

### 本地模型服务启动慢的原因

本地模型需要加载以下文件：
1. **词表文件** (`dataset/processed/vocab.pkl`) - 387KB
2. **模型权重** (`checkpoints/best_model.pth`) - 248MB

模型文件较大，首次加载需要 10-15 秒时间。

## 📝 启动步骤

### 方法一：使用启动脚本（推荐）

1. 双击运行：
   ```
   start_services.bat
   ```

2. 等待所有服务启动完成（约 15 秒）

3. 访问页面：
   ```
   http://localhost:8000/advanced_app.html
   ```

### 方法二：手动启动

打开三个独立的命令行窗口，分别运行：

**窗口 1 - DeepSeek API 服务：**
```bash
cd c:\Users\29838\Desktop\2026-4-3\Chinese-Sentiment-BiLSTM\code
python deepseek_api.py
```

**窗口 2 - 本地模型服务：**
```bash
cd c:\Users\29838\Desktop\2026-4-3\Chinese-Sentiment-BiLSTM\code
python local_model_api.py
```
*等待看到"✅ 本地模型加载成功！"消息（约 10-15 秒）*

**窗口 3 - Web 服务器：**
```bash
cd c:\Users\29838\Desktop\2026-4-3\Chinese-Sentiment-BiLSTM\code
python -m http.server 8000
```

## ✅ 验证服务状态

### 方式一：使用测试页面
在浏览器中打开：
```
c:\Users\29838\Desktop\2026-4-3\Chinese-Sentiment-BiLSTM\code\test_services.html
```
点击"重新检查服务状态"按钮

### 方式二：使用测试脚本
```bash
python test_services.py
```

期望输出：
```
✅ DeepSeek API 服务运行正常！
✅ 本地模型服务运行正常！
```

### 方式三：检查端口
```bash
netstat -ano | findstr ":5000 :5001"
```

应该看到：
- `0.0.0.0:5000` - LISTENING
- `0.0.0.0:5001` - LISTENING

## 🎯 使用情感分析功能

1. 访问：http://localhost:8000/advanced_app.html
2. 点击"**情感分析**"标签
3. 选择分析模式：
   - 🚀 **本地模型**（Bi-LSTM）- 快速、离线
   - 🤖 **DeepSeek AI** - 功能丰富（关键词、摘要）
4. 输入文本并点击分析

## ⚠️ 常见问题

### Q1: 本地模型服务一直无法启动？

**解决方案：**
1. 检查模型文件是否存在：
   ```bash
   dir dataset\processed\vocab.pkl
   dir checkpoints\best_model.pth
   ```

2. 如果文件不存在，需要先训练模型

3. 查看错误日志，确认是否有其他问题

### Q2: 提示"API 调用失败"？

**解决方案：**
1. 确保两个后端服务都在运行
2. 检查端口是否被占用：
   ```bash
   netstat -ano | findstr ":5000 :5001"
   ```

3. 重启服务：
   - 关闭所有 Python 进程
   - 重新运行 `start_services.bat`

### Q3: 本地模型分析速度慢？

**说明：**
- 首次加载模型需要 10-15 秒
- 后续分析应该很快（毫秒级）
- 如果持续很慢，检查 CPU/GPU 使用情况

## 🔍 调试技巧

### 查看本地模型服务日志

启动本地模型服务时，应该看到：
```
============================================================
本地 Bi-LSTM 模型情感分析服务
============================================================
正在加载本地模型...
✅ 本地模型加载成功！设备：CPU
API 地址：http://localhost:5001
单条分析接口：POST /api/local/analyze
健康检查接口：GET /api/local/health
============================================================
```

### 测试本地模型 API

```bash
curl http://localhost:5001/api/local/health
```

期望响应：
```json
{"service": "本地模型情感分析服务", "status": "ok"}
```

## 📞 需要帮助？

如果以上方法都无法解决问题，请检查：
1. Python 版本（建议 3.8+）
2. 依赖包是否安装完整
3. 系统资源是否充足（内存至少 4GB）
4. 防火墙是否阻止端口访问

---

**最后更新**: 2026-04-15  
**版本**: 1.0.0

# 🚀 DeepSeek 云端智能舆情分析 - 快速入门

## ✅ 已完成配置

1. ✅ **API Key 已设置**：`sk-368cd8790f254db7bdb1fe6b8d6682da`
2. ✅ **后端服务已启动**：http://localhost:5000
3. ✅ **前端页面已更新**：http://localhost:8000/Chinese-Sentiment-BiLSTM/code/advanced_app.html

## 📝 使用步骤

### 方法一：使用网页界面（推荐）

1. **打开网页**：
   - 访问：http://localhost:8000/Chinese-Sentiment-BiLSTM/code/advanced_app.html
   - 点击导航栏的"情感分析"标签

2. **单条文本分析**：
   - 在"单条文本情感分析"框中输入评论文本
   - 点击"调用 DeepSeek API 分析"按钮
   - 等待 1-2 秒，查看分析结果（情感倾向、分数、关键词、摘要）

3. **批量文件分析**：
   - 准备 CSV 或 Excel 文件（包含 `text` 列）
   - 在"批量文件分析"区域上传文件
   - 点击"开始批量分析"
   - 查看分析结果表格

### 方法二：使用 Python 脚本

```python
# 测试示例
cd Chinese-Sentiment-BiLSTM\code
python test_deepseek_api.py
```

### 方法三：直接调用 API

```python
import requests

# 单条分析
response = requests.post('http://localhost:5000/api/analyze', json={
    "text": "这是我今年看过的最好的电影，特效炸裂，强烈推荐！"
})
result = response.json()
print(result)

# 输出示例：
# {
#     "success": true,
#     "data": {
#         "sentiment": "正面",
#         "score": 92,
#         "keywords": ["最好", "特效", "推荐"],
#         "summary": "用户对电影给予高度评价"
#     }
# }
```

## 🎯 功能特点

### 1. 单条文本分析
- **输入**：任意中文文本（至少 5 个字符）
- **输出**：
  - 情感倾向：正面/负面/中性
  - 情感分数：0-100
  - 关键词：3 个关键词
  - 智能摘要：一句话总结

### 2. 批量文件分析
- **支持格式**：CSV、Excel
- **文件格式**：
  ```csv
  text
  这是我今年看过的最好的电影
  太失望了，剧情烂俗
  还行吧，中规中矩
  ```
- **输出**：包含原文、情感、分数、关键词、摘要的表格

### 3. 实时 API 服务
- **地址**：http://localhost:5000
- **接口**：
  - `POST /api/analyze` - 单条分析
  - `POST /api/batch_analyze` - 批量分析
  - `GET /api/health` - 健康检查

## 💡 示例测试

### 测试文本 1（正面）
```
这是我今年看过的最好的电影，特效炸裂，演员演技在线，剧情紧凑无尿点，强烈推荐！
```

### 测试文本 2（负面）
```
太失望了，剧情烂俗，演员演技尴尬，特效五毛，浪费票钱，建议大家避雷。
```

### 测试文本 3（中性）
```
还行吧，中规中矩，没有什么亮点但也不算太差，打发时间还可以。
```

## 🔧 服务管理

### 启动服务
```bash
cd Chinese-Sentiment-BiLSTM\code
python deepseek_api.py
```

### 检查服务状态
```bash
curl http://localhost:5000/api/health
```

### 停止服务
- 在终端中按 `Ctrl+C`

## 📊 输出格式说明

### 情感倾向
- **正面**：积极、赞扬、推荐的情感
- **负面**：消极、批评、吐槽的情感
- **中性**：客观、中立、无明显倾向的情感

### 情感分数
- **0-30**：情感较弱
- **31-70**：中等情感
- **71-100**：强烈情感

### 关键词
- 自动提取文本中最重要的 3 个词语
- 用于快速了解文本核心内容

### 智能摘要
- 一句话总结文本主旨
- 不超过 30 字

## ⚠️ 注意事项

1. **API 调用需要网络**：确保网络连接正常
2. **批量处理速度**：每条文本约需 1-2 秒
3. **文本长度限制**：建议不超过 500 字
4. **错误处理**：如遇失败，检查 API Key 和网络

## 🎉 开始使用

现在就可以打开网页开始使用 DeepSeek 云端智能舆情分析系统了！

**网页地址**：http://localhost:8000/Chinese-Sentiment-BiLSTM/code/advanced_app.html

---

**提示**：如需进一步了解详细使用方法，请查看 `DEEPSEEK_USAGE.md` 文档

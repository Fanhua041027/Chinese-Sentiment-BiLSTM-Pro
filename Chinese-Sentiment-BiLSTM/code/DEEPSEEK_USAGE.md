# DeepSeek 云端智能舆情分析系统

## 🌟 系统架构升级

**本地数据采集 + DeepSeek 云端智能分析**

### 核心优势

1. **无需本地模型**：不再需要加载几十万的参数权重，通过 DeepSeek API 直接调用云端智能
2. **功能更丰富**：输出包含情感倾向、分数、关键词和摘要，远超传统模型
3. **准确率更高**：基于 DeepSeek 强大的通用理解能力，无需训练即可达到优秀效果
4. **部署简单**：只需 API Key，Python 脚本即可运行

## 📦 安装步骤

### 1. 安装依赖

```bash
cd Chinese-Sentiment-BiLSTM\code
pip install -r requirements_deepseek.txt
```

### 2. 获取 DeepSeek API Key

1. 访问 https://platform.deepseek.com/
2. 注册/登录账号
3. 在 API Keys 页面创建新的 API Key
4. 设置环境变量：
   ```bash
   # Windows
   setx DEEPSEEK_API_KEY "your-api-key-here"
   
   # Linux/Mac
   export DEEPSEEK_API_KEY="your-api-key-here"
   ```

## 🚀 使用方法

### 方法一：启动 API 服务（推荐）

1. 启动后端服务：
   ```bash
   python deepseek_api.py
   ```

2. 打开浏览器访问：
   ```
   http://localhost:8000/Chinese-Sentiment-BiLSTM/code/advanced_app.html
   ```

3. 在"情感分析"标签页使用以下功能：
   - **单条文本分析**：输入文本，点击"调用 DeepSeek API 分析"
   - **批量文件分析**：上传 CSV/Excel 文件，进行批量分析

### 方法二：直接调用 API

```python
from deepseek_api import analyze_with_deepseek

text = "这是我今年看过的最好的电影，特效炸裂，强烈推荐！"
result = analyze_with_deepseek(text)

if result:
    print(f"情感倾向：{result['sentiment']}")
    print(f"情感分数：{result['score']}/100")
    print(f"关键词：{', '.join(result['keywords'])}")
    print(f"摘要：{result['summary']}")
```

### 方法三：测试示例

```bash
python test_deepseek_api.py
```

## 📊 API 接口说明

### 单条文本分析接口

**请求：**
```http
POST http://localhost:5000/api/analyze
Content-Type: application/json

{
    "text": "待分析的文本"
}
```

**响应：**
```json
{
    "success": true,
    "data": {
        "sentiment": "正面",
        "score": 85,
        "keywords": ["最好", "特效", "推荐"],
        "summary": "用户对电影给予高度评价"
    }
}
```

### 批量分析接口

**请求：**
```http
POST http://localhost:5000/api/batch_analyze
Content-Type: multipart/form-data

file: [CSV 或 Excel 文件]
text_column: text (可选，默认为'text')
```

**响应：**
```json
{
    "success": true,
    "data": [
        {
            "original_text": "原文内容",
            "sentiment": "正面",
            "score": 85,
            "keywords": "关键词 1, 关键词 2, 关键词 3",
            "summary": "摘要内容"
        }
    ],
    "total": 100,
    "processed": 98
}
```

## 🔧 Prompt 模板设计

系统使用以下 Prompt 模板来保证输出格式稳定：

```python
SYSTEM_PROMPT = """你是一个专业的舆情分析专家。请分析用户输入的文本情感。
请仅返回一个标准的 JSON 格式，不要包含 markdown 标记或其他废话。
JSON 字段包含：
- "sentiment": 情感倾向（正面/负面/中性）
- "score": 情感强烈程度（0-100）
- "keywords": 提取的 3 个关键词（数组格式）
- "summary": 一句话总结（不超过 30 字）"""
```

## 📈 批量处理 CSV/Excel 示例

### CSV 文件格式

```csv
text
这是我今年看过的最好的电影，特效炸裂，强烈推荐！
太失望了，剧情烂俗，演员演技尴尬，浪费票钱。
还行吧，中规中矩，没有什么亮点但也不算太差。
```

### Excel 文件格式

创建一个 Excel 文件，包含 `text` 列，每行是一条待分析的文本。

## 💡 进阶技巧

### 1. 实时舆情监控

结合爬虫或 API 获取实时数据，使用 DeepSeek 进行实时分析：

```python
import time

def get_realtime_data():
    # 替换为你的爬虫代码或 API 调用
    return ["评论 1", "评论 2", "评论 3"]

while True:
    comments = get_realtime_data()
    for comment in comments:
        result = analyze_with_deepseek(comment)
        if result:
            print(f"{comment}: {result['sentiment']}")
    time.sleep(5)  # 每 5 秒刷新一次
```

### 2. 生成舆情周报

将一天采集到的评论汇总，让 DeepSeek 写报告：

```python
comments_text = "\n".join(all_comments[:500])  # 限制长度

prompt = f"""这是今天采集到的 500 条关于某产品的用户评论（数据附后）。
请帮我写一份舆情日报，包含：
1. 整体舆论风向
2. 用户最吐槽的 3 个痛点
3. 用户最期待的 3 个功能
4. 给产品经理的建议

评论数据：
{comments_text}"""

report = analyze_with_deepseek(prompt)
```

## 📊 DeepSeek 方案对比本地模型

| 特性 | 本地 BERT/Bi-LSTM | DeepSeek API 方案 |
|------|------------------|------------------|
| 部署难度 | 高（需下载权重、配置环境、GPU） | 极低（只需 API Key） |
| 功能丰富度 | 仅能输出分类标签（正/负） | 输出 JSON（含关键词、摘要、评分） |
| 准确率 | 依赖训练数据质量 | 通用理解能力极强，无需训练 |
| 成本 | 硬件成本高（显卡） | 按量付费，适合中小规模数据 |
| 实时性 | 极快（本地推理） | 取决于网络延迟（通常 < 1 秒） |

## ⚠️ 注意事项

1. **API 限流**：DeepSeek API 可能有速率限制，批量处理时建议添加延时
2. **网络依赖**：需要稳定的网络连接才能调用 API
3. **成本控制**：按量付费，大规模使用时注意成本
4. **错误处理**：建议添加重试机制和错误处理逻辑

## 🎯 建议使用场景

- **中小规模数据分析**（每日 < 10 万条）
- **快速原型开发**
- **需要丰富输出**（关键词、摘要等）
- **无 GPU 环境**
- **需要快速部署**

## 📝 常见问题

### Q: API 调用失败怎么办？
A: 检查以下几点：
1. API Key 是否正确设置
2. 网络连接是否正常
3. 查看错误日志获取详细信息

### Q: 批量处理太慢怎么办？
A: 可以尝试：
1. 使用多线程/多进程并发处理
2. 减少延时时间（但可能触发限流）
3. 升级到更高级别的 API 服务

### Q: 输出格式不稳定怎么办？
A: 可以调整 Prompt 或降低 temperature 参数（已设置为 0.2）

## 🔗 相关链接

- [DeepSeek 官网](https://platform.deepseek.com/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

---

**作者**: 情感分析项目团队  
**日期**: 2026-04-15  
**版本**: 1.0.0

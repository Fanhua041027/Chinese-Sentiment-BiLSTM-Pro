# 批量分析功能测试指南

## ✅ 问题已修复

**问题**：点击"开始批量分析"按钮没有反应  
**原因**：JavaScript 代码中使用了 `event.target`，但函数没有传入 event 参数  
**解决方案**：已修改为使用 `document.querySelector()` 直接获取按钮元素

## 🧪 测试步骤

### 方法一：使用网页界面测试

1. **打开网页**：
   - 访问：http://localhost:8000/Chinese-Sentiment-BiLSTM/code/advanced_app.html
   - 点击"情感分析"标签

2. **准备测试文件**：
   - 使用已创建的测试文件：`test_batch.csv`
   - 或创建自己的 CSV 文件（包含 `text` 列）

3. **上传文件**：
   - 在"批量文件分析"区域点击"上传 CSV 或 Excel 文件"
   - 选择 `test_batch.csv` 文件

4. **开始分析**：
   - 点击"开始批量分析"按钮
   - 等待分析完成（每条文本约 1-2 秒）
   - 查看分析结果表格

### 方法二：使用 curl 命令测试

```bash
curl -X POST http://localhost:5000/api/batch_analyze \
  -F "file=@test_batch.csv" \
  -F "text_column=text"
```

### 方法三：使用 Python 测试

```python
import requests

# 准备文件
files = {'file': open('test_batch.csv', 'rb')}
data = {'text_column': 'text'}

# 发送请求
response = requests.post('http://localhost:5000/api/batch_analyze', 
                        files=files, data=data)

# 查看结果
print(response.json())
```

## 📊 测试文件示例

### CSV 文件格式 (test_batch.csv)

```csv
text
这是我今年看过的最好的电影，特效炸裂，强烈推荐！
太失望了，剧情烂俗，演员演技尴尬，浪费票钱。
还行吧，中规中矩，没有什么亮点但也不算太差。
剧情紧凑，演员演技在线，是一部值得二刷的好电影。
垃圾电影，浪费我的时间，建议大家避雷。
```

### Excel 文件格式

创建 Excel 文件，第一列为 `text`，每行是一条待分析的文本。

## 🎯 预期结果

### 成功响应示例

```json
{
  "success": true,
  "data": [
    {
      "original_text": "这是我今年看过的最好的电影，特效炸裂，强烈推荐！",
      "sentiment": "正面",
      "score": 95,
      "keywords": "最好，特效，推荐",
      "summary": "用户对电影给予高度评价"
    },
    {
      "original_text": "太失望了，剧情烂俗，演员演技尴尬，浪费票钱。",
      "sentiment": "负面",
      "score": 88,
      "keywords": "失望，烂俗，尴尬",
      "summary": "用户对电影表示强烈不满"
    }
  ],
  "total": 5,
  "processed": 5
}
```

### 网页显示效果

分析完成后，网页将显示：
- ✅ 分析完成提示
- 📊 结果统计（共处理 X 条数据，成功 X 条）
- 📋 结果表格（原文、情感、分数、关键词、摘要）

## ⚠️ 常见问题排查

### 问题 1：点击按钮后没有任何反应

**检查步骤**：
1. 打开浏览器开发者工具（F12）
2. 查看 Console 是否有错误信息
3. 确认文件已正确选择
4. 确认后端服务是否运行（访问 http://localhost:5000/api/health）

### 问题 2：提示"API 调用失败"

**解决方案**：
1. 检查后端服务是否运行：
   ```bash
   curl http://localhost:5000/api/health
   ```
2. 如果服务未运行，启动服务：
   ```bash
   cd Chinese-Sentiment-BiLSTM\code
   python deepseek_api.py
   ```

### 问题 3：分析结果为空或失败

**可能原因**：
1. API Key 无效或过期
2. 网络连接问题
3. 文本格式不正确
4. API 服务限流

**解决方案**：
1. 检查 API Key 配置是否正确
2. 查看后端服务日志
3. 确认 CSV 文件格式正确

## 🔧 代码修改说明

### 修改前（有问题）

```javascript
const submitBtn = event.target.querySelector('button[onclick="batchAnalyze()"]');
```

**问题**：`batchAnalyze()` 函数通过 `onclick` 属性调用，没有传入 `event` 参数，导致 `event` 未定义。

### 修改后（已修复）

```javascript
const submitBtn = document.querySelector('button[onclick="batchAnalyze()"]');
```

**解决方案**：直接使用 `document.querySelector()` 获取按钮元素，不依赖 `event` 对象。

## 📝 测试清单

- [ ] 后端服务已启动（http://localhost:5000）
- [ ] 前端页面可访问（http://localhost:8000）
- [ ] 测试文件已准备（test_batch.csv）
- [ ] 点击"开始批量分析"按钮
- [ ] 按钮显示"分析中..."状态
- [ ] 分析完成后显示结果表格
- [ ] 结果包含情感、分数、关键词、摘要

## 🎉 测试完成

如果以上测试都通过，说明批量分析功能已正常工作！

---

**最后更新**: 2026-04-15  
**版本**: 1.0.1（已修复）

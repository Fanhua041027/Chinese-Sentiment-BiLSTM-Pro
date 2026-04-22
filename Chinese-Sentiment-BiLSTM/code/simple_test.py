"""
简化的 API 测试
"""
import requests

print('测试本地模型 API...')

# 测试健康检查
print('\n1. 健康检查：')
try:
    r = requests.get('http://localhost:5002/api/local/health', timeout=5)
    print(f'   状态码：{r.status_code}')
    data = r.json()
    print(f'   model_loaded: {data.get("model_loaded")}')
    print(f'   vocab_loaded: {data.get("vocab_loaded")}')
    print(f'   preprocessor_loaded: {data.get("preprocessor_loaded")}')
    print(f'   status: {data.get("status")}')
except Exception as e:
    print(f'   ❌ 失败：{e}')

# 测试情感分析
print('\n2. 情感分析：')
try:
    r = requests.post(
        'http://localhost:5002/api/local/analyze',
        json={'text': '这部电影很好看'},
        timeout=10
    )
    print(f'   状态码：{r.status_code}')
    print(f'   响应：{r.json()}')
except Exception as e:
    print(f'   ❌ 失败：{e}')

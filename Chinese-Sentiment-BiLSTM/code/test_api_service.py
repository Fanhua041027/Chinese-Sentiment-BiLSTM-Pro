"""
测试本地模型 API 服务
"""
import requests

print('测试本地模型 API 服务...')
print()

# 测试健康检查
print('1. 测试健康检查：')
try:
    r = requests.get('http://localhost:5001/api/local/health', timeout=10)
    print(f'   状态码：{r.status_code}')
    print(f'   响应：{r.json()}')
except Exception as e:
    print(f'   ❌ 失败：{e}')

# 测试情感分析
print('\n2. 测试情感分析：')
try:
    r = requests.post(
        'http://localhost:5001/api/local/analyze',
        json={'text': '这部电影很好看'},
        timeout=30
    )
    print(f'   状态码：{r.status_code}')
    print(f'   响应：{r.json()}')
except Exception as e:
    print(f'   ❌ 失败：{e}')

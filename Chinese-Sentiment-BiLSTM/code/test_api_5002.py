"""
测试本地模型 API 服务 (端口 5002)
"""
import requests

print('测试本地模型 API 服务 (端口 5002)...')
print()

# 测试健康检查
print('1. 测试健康检查：')
try:
    r = requests.get('http://localhost:5002/api/local/health', timeout=10)
    print(f'   状态码：{r.status_code}')
    print(f'   响应：{r.json()}')
except Exception as e:
    print(f'   ❌ 失败：{e}')

# 测试情感分析
print('\n2. 测试情感分析：')
try:
    r = requests.post(
        'http://localhost:5002/api/local/analyze',
        json={'text': '这部电影很好看'},
        timeout=30
    )
    print(f'   状态码：{r.status_code}')
    result = r.json()
    print(f'   响应：{result}')

    if result.get('success'):
        print(f'\n   ✅ 分析成功！')
        print(f'   文本：这部电影很好看')
        print(f'   情感：{result["data"]["sentiment"]}')
        print(f'   置信度：{result["data"]["confidence"]}%')
    else:
        print(f'\n   ❌ 分析失败：{result.get("error")}')

except Exception as e:
    print(f'   ❌ 失败：{e}')

"""
测试本地模型 API HTTP 调用
"""
import requests
import json

print('=' * 60)
print('测试本地模型 API HTTP 调用...')
print('=' * 60)

# 测试健康检查
print('\n1. 测试健康检查接口...')
try:
    response = requests.get('http://localhost:5001/api/local/health', timeout=10)
    print(f'   状态码：{response.status_code}')
    print(f'   响应：{response.json()}')
except Exception as e:
    print(f'   ❌ 连接失败：{e}')

# 测试情感分析
print('\n2. 测试情感分析接口...')
test_text = '这部电影很好看，强烈推荐！'

try:
    response = requests.post(
        'http://localhost:5001/api/local/analyze',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'text': test_text}),
        timeout=30
    )
    print(f'   状态码：{response.status_code}')
    result = response.json()
    print(f'   响应：{json.dumps(result, ensure_ascii=False, indent=2)}')

    if result.get('success'):
        print(f'\n   ✅ 分析成功！')
        print(f'   文本：{test_text}')
        print(f'   情感：{result["data"]["sentiment"]}')
        print(f'   置信度：{result["data"]["confidence"]}%')
    else:
        print(f'\n   ❌ 分析失败：{result.get("error")}')

except Exception as e:
    print(f'   ❌ 请求失败：{e}')

print('\n' + '=' * 60)

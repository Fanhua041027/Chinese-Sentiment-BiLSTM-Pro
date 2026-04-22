"""
检查服务状态
"""
import requests
import time

print('检查服务状态...')
print()

# 检查 Web 服务器
print('1. Web 服务器 (端口 8000):')
try:
    r = requests.get('http://localhost:8000/advanced_app.html', timeout=5)
    print(f'   ✅ 可访问，状态码：{r.status_code}')
except Exception as e:
    print(f'   ❌ 不可访问：{e}')

# 检查本地模型服务
print('\n2. 本地模型服务 (端口 5002):')
try:
    r = requests.get('http://localhost:5002/api/local/health', timeout=5)
    print(f'   ✅ 运行正常')
    print(f'   状态：{r.json()}')
except Exception as e:
    print(f'   ❌ 未运行：{e}')

print('\n服务状态检查完成！')

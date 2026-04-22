"""
启动本地模型服务并测试
"""
import subprocess
import time
import requests
import sys

print('=' * 60)
print('启动本地模型服务并测试...')
print('=' * 60)

# 启动服务器
print('\n1. 启动 Flask 服务器...')
proc = subprocess.Popen(
    [sys.executable, 'local_model_api.py'],
    cwd=r'c:\Users\29838\Desktop\2026-4-3\Chinese-Sentiment-BiLSTM\code',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# 等待服务器启动
print('   等待服务器启动...')
time.sleep(8)

# 检查服务器输出
print('\n2. 服务器启动日志：')
for _ in range(15):
    line = proc.stdout.readline()
    if line:
        print(f'   {line.strip()}')
    else:
        break

# 测试 API
print('\n3. 测试健康检查：')
try:
    r = requests.get('http://localhost:5002/api/local/health', timeout=10)
    print(f'   状态码：{r.status_code}')
    print(f'   响应：{r.json()}')
except Exception as e:
    print(f'   ❌ 失败：{e}')

# 测试情感分析
print('\n4. 测试情感分析：')
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

# 清理
print('\n5. 关闭服务器...')
proc.terminate()
proc.wait(timeout=5)
print('   服务器已关闭')

print('\n' + '=' * 60)
print('测试完成！')
print('=' * 60)

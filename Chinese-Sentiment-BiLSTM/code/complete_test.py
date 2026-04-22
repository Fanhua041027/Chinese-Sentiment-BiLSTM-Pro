"""
完整的本地模型 API 测试
"""
import subprocess
import sys
import time
import requests
from multiprocessing import Process

def run_server():
    """在单独进程中运行服务器"""
    subprocess.run([
        sys.executable,
        'local_model_api.py'
    ], cwd=r'c:\Users\29838\Desktop\2026-4-3\Chinese-Sentiment-BiLSTM\code')

if __name__ == '__main__':
    print('=' * 60)
    print('本地模型 API 完整测试')
    print('=' * 60)

    # 启动服务器进程
    print('\n1. 启动本地模型服务器...')
    server_process = Process(target=run_server)
    server_process.start()

    # 等待服务器启动
    print('   等待服务器初始化（10秒）...')
    time.sleep(10)

    # 检查进程是否还在运行
    if not server_process.is_alive():
        print('   ❌ 服务器进程已退出')
        sys.exit(1)

    print('   ✅ 服务器进程正在运行')

    # 测试 API
    print('\n2. 测试健康检查接口...')
    try:
        r = requests.get('http://localhost:5002/api/local/health', timeout=10)
        print(f'   状态码：{r.status_code}')
        data = r.json()
        print(f'   响应：{data}')

        if data.get('status') == 'ok':
            print('   ✅ 健康检查通过！')
        else:
            print('   ⚠️ 服务器状态异常')
    except Exception as e:
        print(f'   ❌ 健康检查失败：{e}')

    # 测试情感分析
    print('\n3. 测试情感分析接口...')
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
            print(f'\n   ✅ 情感分析成功！')
            print(f'   文本：这部电影很好看')
            print(f'   情感：{result["data"]["sentiment"]}')
            print(f'   置信度：{result["data"]["confidence"]}%')
        else:
            print(f'\n   ❌ 情感分析失败：{result.get("error")}')
    except Exception as e:
        print(f'   ❌ 请求失败：{e}')

    # 清理
    print('\n4. 关闭服务器...')
    server_process.terminate()
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.kill()
    print('   ✅ 服务器已关闭')

    print('\n' + '=' * 60)
    print('测试完成！')
    print('=' * 60)

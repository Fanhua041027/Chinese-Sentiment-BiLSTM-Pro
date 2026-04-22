"""
检查 DeepSeek 服务状态
"""
import requests

def check_deepseek_service():
    print('检查 DeepSeek 服务状态...')
    try:
        response = requests.get('http://localhost:5003/api/health', timeout=5)
        print(f'✅ DeepSeek 服务运行正常')
        print(f'   状态：{response.json()}')
        return True
    except requests.exceptions.ConnectionError:
        print('❌ DeepSeek 服务未运行')
        return False
    except Exception as e:
        print(f'❌ 检查服务时出错：{e}')
        return False

if __name__ == '__main__':
    check_deepseek_service()
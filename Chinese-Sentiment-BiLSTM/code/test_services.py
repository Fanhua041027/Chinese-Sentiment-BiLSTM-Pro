"""
测试后端服务是否正常运行
"""
import requests
import time

print("=" * 60)
print("开始测试后端服务...")
print("=" * 60)

# 测试 DeepSeek API 服务 (端口 5000)
print("\n1. 测试 DeepSeek API 服务 (端口 5000)...")
try:
    response = requests.get("http://localhost:5000/api/health", timeout=5)
    if response.status_code == 200:
        print("   ✅ DeepSeek API 服务运行正常！")
        print(f"   响应：{response.json()}")
    else:
        print(f"   ❌ DeepSeek API 服务异常，状态码：{response.status_code}")
except Exception as e:
    print(f"   ❌ DeepSeek API 服务连接失败：{e}")
    print("   提示：请确保 deepseek_api.py 正在运行")

# 测试本地模型服务 (端口 5001)
print("\n2. 测试本地模型服务 (端口 5001)...")
try:
    response = requests.get("http://localhost:5001/api/local/health", timeout=5)
    if response.status_code == 200:
        print("   ✅ 本地模型服务运行正常！")
        print(f"   响应：{response.json()}")
    else:
        print(f"   ❌ 本地模型服务异常，状态码：{response.status_code}")
except Exception as e:
    print(f"   ❌ 本地模型服务连接失败：{e}")
    print("   提示：请确保 local_model_api.py 正在运行")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n如果两个服务都正常，您现在可以在网页中使用情感分析功能了。")
print("访问：http://localhost:8000/Chinese-Sentiment-BiLSTM/code/advanced_app.html")
print("\n使用说明：")
print("1. 在浏览器中打开上述网址")
print("2. 点击'情感分析'标签")
print("3. 选择分析模式（本地模型 或 DeepSeek AI）")
print("4. 输入文本并点击分析按钮")

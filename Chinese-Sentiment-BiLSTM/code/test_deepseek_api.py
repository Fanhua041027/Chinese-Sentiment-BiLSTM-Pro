"""
DeepSeek API 使用示例

本脚本演示如何使用 DeepSeek API 进行情感分析

使用前请确保：
1. 已设置环境变量 DEEPSEEK_API_KEY
2. 已安装必要的依赖包（pip install -r requirements_deepseek.txt）
"""

import os
from deepseek_api import analyze_with_deepseek

def main():
    print("=" * 60)
    print("DeepSeek API 情感分析示例")
    print("=" * 60)
    
    # 示例文本
    test_texts = [
        "这是我今年看过的最好的电影，特效炸裂，强烈推荐！",
        "太失望了，剧情烂俗，演员演技尴尬，浪费票钱。",
        "还行吧，中规中矩，没有什么亮点但也不算太差。"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n示例 {i}: {text}")
        print("-" * 60)
        
        # 调用 API 分析
        result = analyze_with_deepseek(text)
        
        if result:
            print(f"情感倾向：{result.get('sentiment', 'N/A')}")
            print(f"情感分数：{result.get('score', 'N/A')}/100")
            print(f"关键词：{', '.join(result.get('keywords', []))}")
            print(f"摘要：{result.get('summary', 'N/A')}")
        else:
            print("分析失败，请检查 API Key 和网络连接")
        
        print()
    
    print("=" * 60)
    print("示例运行完成")
    print("=" * 60)
    print("\n提示：")
    print("1. 运行 'python deepseek_api.py' 启动 API 服务")
    print("2. 在浏览器中打开 advanced_app.html 使用完整功能")
    print("3. 设置环境变量：setx DEEPSEEK_API_KEY 'your-api-key-here'")

if __name__ == '__main__':
    # 检查 API Key 是否设置
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "sk-your-api-key-here":
        print("⚠️  警告：未设置 DEEPSEEK_API_KEY 环境变量")
        print("\n获取 API Key 的步骤：")
        print("1. 访问 https://platform.deepseek.com/")
        print("2. 注册/登录账号")
        print("3. 在 API Keys 页面创建新的 API Key")
        print("4. 设置环境变量：setx DEEPSEEK_API_KEY 'your-api-key-here'")
        print("\n或者临时测试（不推荐）：")
        print("在 deepseek_api.py 中修改 DEEPSEEK_API_KEY 的值")
        return
    
    main()

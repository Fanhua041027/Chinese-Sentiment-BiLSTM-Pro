import os
from dotenv import load_dotenv

# 1. 加载 .env 文件中的变量
load_dotenv()

# 2. 获取变量
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")

# 3. 验证是否获取成功（调试用）
if api_key:
    print("✅ 环境变量加载成功！")
    # 这里接上你之前的代码逻辑
    # DEEPSEEK_API_URL = f"{base_url}/chat/completions"
else:
    print("❌ 环境变量加载失败，请检查 .env 文件")
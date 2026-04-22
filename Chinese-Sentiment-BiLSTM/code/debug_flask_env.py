"""
调试 Flask 服务器环境
"""
import os
import sys
sys.path.insert(0, '.')

# 模拟 Flask 应用环境
from local_model_api import app, load_local_model, predict_with_local_model

print('=' * 60)
print('调试 Flask 服务器环境...')
print('=' * 60)

# 测试 predict_with_local_model
print('\n1. 测试 predict_with_local_model 函数...')
text = '这部电影很好看'
result = predict_with_local_model(text)
print(f'   结果：{result}')

if result:
    print(f'   ✅ 成功！情感：{result["sentiment"]}, 置信度：{result["confidence"]}%')
else:
    print(f'   ❌ 失败')

print('\n' + '=' * 60)

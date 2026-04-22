"""
在 Flask 应用上下文中测试
"""
import sys
sys.path.insert(0, '.')

from local_model_api import app, predict_with_local_model

print('在 Flask 应用上下文中测试...')
print()

with app.app_context():
    # 测试 predict_with_local_model
    text = '这部电影很好看'
    print(f'1. 测试文本：{text}')
    result = predict_with_local_model(text)
    print(f'2. 结果：{result}')

    if result:
        print(f'   ✅ 成功！情感：{result["sentiment"]}, 置信度：{result["confidence"]}%')
    else:
        print('   ❌ 失败')

print('\n测试完成。')

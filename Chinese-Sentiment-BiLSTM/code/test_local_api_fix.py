"""
测试本地模型 API 修复
"""
import sys
sys.path.insert(0, '.')

from local_model_api import load_local_model, predict_with_local_model

print('=' * 60)
print('开始测试本地模型 API...')
print('=' * 60)

print('\n1. 加载模型...')
if load_local_model():
    print('   ✅ 模型加载成功！')
    
    print('\n2. 测试预测功能...')
    test_texts = ['这部电影很好看', '非常糟糕的体验', '一般般吧']
    for text in test_texts:
        result = predict_with_local_model(text)
        if result:
            print(f'   文本：{text}')
            print(f'   -> 情感：{result["sentiment"]}, 置信度：{result["confidence"]}%')
            print()
        else:
            print(f'   ❌ 文本：{text} -> 预测失败')
    
    print('=' * 60)
    print('✅ 所有测试完成！')
    print('=' * 60)
else:
    print('   ❌ 模型加载失败！')

"""
测试 Python 环境
"""
print('Python 环境测试')
print('导入必要的库...')

try:
    import flask
    import requests
    import pandas
    import tqdm
    print('✅ 所有依赖库都已安装')
except Exception as e:
    print(f'❌ 依赖库安装问题：{e}')

print('测试完成！')
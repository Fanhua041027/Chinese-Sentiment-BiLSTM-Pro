"""
测试 local_model_api.py
"""
import traceback

try:
    with open('local_model_api.py', 'r', encoding='utf-8') as f:
        code = f.read()
    exec(code)
except Exception as e:
    print('Error:', e)
    print('Traceback:')
    traceback.print_exc()
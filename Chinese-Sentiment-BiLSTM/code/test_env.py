import os
import sys

print('Python version:', sys.version)
print('Current directory:', os.getcwd())

# 测试Flask导入
try:
    import flask
    print('Flask available')
except ImportError as e:
    print('Flask not available:', e)

# 测试其他依赖
try:
    import torch
    print('PyTorch available')
except ImportError as e:
    print('PyTorch not available:', e)

try:
    import jieba
    print('Jieba available')
except ImportError as e:
    print('Jieba not available:', e)

try:
    import pandas
    print('Pandas available')
except ImportError as e:
    print('Pandas not available:', e)

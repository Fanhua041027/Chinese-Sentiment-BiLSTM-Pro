"""
调试 TextPreprocessor
"""
import sys
sys.path.insert(0, '.')

import pickle
from src.data.preprocess import TextPreprocessor

print('=' * 60)
print('调试 TextPreprocessor...')
print('=' * 60)

# 加载词表
with open('dataset/processed/vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)

if isinstance(vocab_data, dict) and 'vocab' in vocab_data:
    vocab = vocab_data['vocab']
    max_len = vocab_data.get('max_len', 128)
else:
    vocab = vocab_data
    max_len = 128

print(f'\n1. 词表信息：')
print(f'   词表大小：{len(vocab)}')
print(f'   max_len：{max_len}')

print(f'\n2. 创建 TextPreprocessor...')
preprocessor = TextPreprocessor(max_vocab_size=len(vocab), max_len=max_len)
print(f'   预处理器创建成功')
print(f'   preprocessor.max_len = {preprocessor.max_len}')
print(f'   preprocessor.vocab 类型：{type(preprocessor.vocab)}')
print(f'   preprocessor.vocab 大小：{len(preprocessor.vocab)}')

print(f'\n3. 赋值词表...')
preprocessor.vocab = vocab
print(f'   赋值后 preprocessor.vocab 大小：{len(preprocessor.vocab)}')

print(f'\n4. 测试 text_to_sequence...')
test_text = '这部电影很好看'
try:
    result = preprocessor.text_to_sequence(test_text)
    print(f'   成功！结果长度：{len(result)}')
    print(f'   前10个token ID：{result[:10]}')
except Exception as e:
    print(f'   ❌ 失败：{e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)

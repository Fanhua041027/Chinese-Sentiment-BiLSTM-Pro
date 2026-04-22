#!/usr/bin/env python3
"""
启动Flask可视化界面的脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print('正在启动Flask应用...')
    from app import app
    print('应用加载成功')
    print('可用路由:')
    for rule in app.url_map.iter_rules():
        print(f'  {rule.rule}')
    print('\nFlask服务器启动中...')
    print('访问地址: http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=True)
except Exception as e:
    print(f'启动失败: {e}')
    import traceback
    traceback.print_exc()
    input('按回车键退出...')

"""
测试 Flask 服务
"""
from flask import Flask

app = Flask(__name__)

@app.route('/test')
def test():
    return 'Flask 服务正常运行！'

if __name__ == '__main__':
    print('启动测试 Flask 服务...')
    app.run(host='127.0.0.1', port=5004, debug=False)
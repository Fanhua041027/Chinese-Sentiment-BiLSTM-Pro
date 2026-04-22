from flask import Flask, render_template, request, jsonify
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd

app = Flask(__name__)

# 读取评估报告
def read_report(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "报告未找到"

# 读取图表
def read_image(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None

# 添加read_image函数到模板上下文
@app.context_processor
def utility_processor():
    return dict(read_image=read_image)

# 主页
@app.route('/')
def index():
    return render_template('simple_index.html')

# 模型性能页面
@app.route('/performance')
def performance():
    # 读取评估报告
    bilstm_report = read_report('dataset/reports/test_report.txt')
    nb_report = read_report('dataset/reports/baseline_report.txt')
    bert_report = read_report('dataset/reports/bert_report.txt')
    
    # 读取图表
    model_comparison = read_image('dataset/figures/model_comparison.png')
    training_curve = read_image('dataset/figures/training_curve.png')
    confusion_matrix = read_image('dataset/figures/confusion_matrix.png')
    baseline_confusion = read_image('dataset/figures/baseline_confusion_matrix.png')
    bert_confusion = read_image('dataset/figures/bert_confusion_matrix.png')
    
    return render_template('simple_performance.html', 
                          bilstm_report=bilstm_report,
                          nb_report=nb_report,
                          bert_report=bert_report,
                          model_comparison=model_comparison,
                          training_curve=training_curve,
                          confusion_matrix=confusion_matrix,
                          baseline_confusion=baseline_confusion,
                          bert_confusion=bert_confusion)

if __name__ == '__main__':
    # 创建templates目录
    os.makedirs('templates', exist_ok=True)
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)

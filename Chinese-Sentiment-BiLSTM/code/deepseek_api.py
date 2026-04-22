"""
DeepSeek API 情感分析服务

本模块提供基于 DeepSeek API 的情感分析功能，支持：
1. 单条文本情感分析
2. 批量 CSV/Excel 文件分析
3. 实时舆情监控

核心优势：
- 无需本地模型权重，通过 API 调用云端智能
- 输出包含情感倾向、分数、关键词和摘要
- 支持自定义 Prompt 模板
"""

import json
import os
from typing import Dict, List, Optional
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from tqdm import tqdm
import time

# ==================== Flask 应用初始化 ====================
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ==================== DeepSeek API 配置 ====================
# 从环境变量获取 API Key，如果没有则使用提供的 API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-368cd8790f254db7bdb1fe6b8d6682da")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_URL = f"{DEEPSEEK_BASE_URL}/chat/completions"

# ==================== Prompt 模板 ====================
SYSTEM_PROMPT = """你是一个专业的舆情分析专家。请分析用户输入的文本情感。
请仅返回一个标准的 JSON 格式，不要包含 markdown 标记或其他废话。
JSON 字段包含：
- "sentiment": 情感倾向（正面/负面/中性）
- "score": 情感强烈程度（0-100）
- "keywords": 提取的 3 个关键词（数组格式）
- "summary": 一句话总结（不超过 30 字）"""


# ==================== 核心分析函数 ====================
def analyze_with_deepseek(text: str) -> Optional[Dict]:
    """
    使用 DeepSeek API 分析单条文本的情感
    
    参数：
        text (str): 待分析的文本
        
    返回：
        dict: 包含 sentiment, score, keywords, summary 的字典
        None: 如果分析失败
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"分析这段文本：{text}"}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    
    except Exception as e:
        print(f"API 调用错误：{e}")
        return None


# ==================== Flask API 路由 ====================
@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """
    单条文本情感分析接口
    
    请求格式：
    {
        "text": "待分析的文本"
    }
    
    返回格式：
    {
        "success": true/false,
        "data": {
            "sentiment": "正面/负面/中性",
            "score": 0-100,
            "keywords": ["关键词 1", "关键词 2", "关键词 3"],
            "summary": "一句话总结"
        },
        "error": "错误信息（如果有）"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "请提供文本内容"
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                "success": False,
                "error": "文本内容不能为空"
            }), 400
        
        # 调用 DeepSeek API
        result = analyze_with_deepseek(text)
        
        if result:
            return jsonify({
                "success": True,
                "data": result
            })
        else:
            return jsonify({
                "success": False,
                "error": "API 调用失败，请稍后重试"
            }), 500
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """
    批量分析接口（支持 CSV/Excel 文件上传）
    
    请求格式：multipart/form-data
    文件字段：file (CSV 或 Excel 文件)
    文本列名：text_column (可选，默认为 'text')
    
    返回格式：
    {
        "success": true/false,
        "data": [分析结果数组],
        "error": "错误信息（如果有）"
    }
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "请上传文件"
            }), 400
        
        file = request.files['file']
        text_column = request.form.get('text_column', 'text')
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "文件名为空"
            }), 400
        
        # 读取文件
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({
                "success": False,
                "error": "不支持的文件格式，请上传 CSV 或 Excel 文件"
            }), 400
        
        if text_column not in df.columns:
            return jsonify({
                "success": False,
                "error": f"文件中找不到 '{text_column}' 列"
            }), 400
        
        # 批量分析
        results = []
        total = len(df)
        
        for index, row in tqdm(df.iterrows(), total=total, desc="分析中"):
            text = row[text_column]
            
            # 调用 API
            analysis = analyze_with_deepseek(str(text))
            
            if analysis:
                results.append({
                    "original_text": text,
                    "sentiment": analysis.get('sentiment'),
                    "score": analysis.get('score'),
                    "keywords": ", ".join(analysis.get('keywords', [])),
                    "summary": analysis.get('summary')
                })
            else:
                results.append({
                    "original_text": text,
                    "sentiment": "Error",
                    "score": 0,
                    "keywords": "",
                    "summary": "分析失败"
                })
            
            # 简单的限流保护
            time.sleep(0.1)
        
        return jsonify({
            "success": True,
            "data": results,
            "total": total,
            "processed": len(results)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "service": "DeepSeek 情感分析服务"
    })


# ==================== 主函数 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("DeepSeek 情感分析服务启动中...")
    print("=" * 60)
    print(f"API 地址：http://localhost:5003")
    print(f"单条分析接口：POST /api/analyze")
    print(f"批量分析接口：POST /api/batch_analyze")
    print(f"健康检查接口：GET /api/health")
    print("=" * 60)
    
    # 启动 Flask 服务
    print("启动 Flask 服务...")
    app.run(host='127.0.0.1', port=5003, debug=False)

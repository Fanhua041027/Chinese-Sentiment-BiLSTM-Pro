"""
本地 Bi-LSTM 模型情感分析服务

本模块提供基于本地 Bi-LSTM + Attention 模型的情感分析功能，
与 DeepSeek API 服务并存，供用户选择使用。

核心优势：
- 无需联网，本地推理
- 快速响应，低延迟
- 隐私保护，数据不出本地
"""

import os
import sys
import json
import time
from typing import Dict, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.bilstm_attention import SentimentModel
from src.data.preprocess import TextPreprocessor

# ==================== Flask 应用初始化 ====================
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ==================== 本地模型配置 ====================
# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_CONFIG = {
    "vocab_path": os.path.join(BASE_DIR, "dataset/processed/vocab.pkl"),
    "model_path": os.path.join(BASE_DIR, "checkpoints/best_model.pth"),
    "embedding_dim": 128,
    "hidden_dim": 256,
    "output_dim": 1,
    "n_layers": 2,
    "dropout": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 全局变量存储模型和词表
model = None
vocab = None
preprocessor = None


def load_local_model():
    """
    加载本地 Bi-LSTM 模型、词表和预处理器
    """
    global model, vocab, preprocessor
    
    try:
        # 加载词表
        if not os.path.exists(LOCAL_MODEL_CONFIG["vocab_path"]):
            print(f"❌ 词表文件不存在：{LOCAL_MODEL_CONFIG['vocab_path']}")
            return False
        
        with open(LOCAL_MODEL_CONFIG["vocab_path"], "rb") as f:
            vocab_data = pickle.load(f)
        
        # 兼容不同格式
        if isinstance(vocab_data, dict) and "vocab" in vocab_data:
            vocab = vocab_data["vocab"]
            max_len = vocab_data.get("max_len", 128)
        else:
            vocab = vocab_data
            max_len = 128
        
        # 创建预处理器并加载词表
        preprocessor = TextPreprocessor(max_vocab_size=len(vocab), max_len=max_len)
        preprocessor.vocab = vocab
        
        # 加载模型
        if not os.path.exists(LOCAL_MODEL_CONFIG["model_path"]):
            print(f"❌ 模型文件不存在：{LOCAL_MODEL_CONFIG['model_path']}")
            return False
        
        model = SentimentModel(
            vocab_size=len(vocab),
            embedding_dim=LOCAL_MODEL_CONFIG["embedding_dim"],
            hidden_dim=LOCAL_MODEL_CONFIG["hidden_dim"],
            output_dim=LOCAL_MODEL_CONFIG["output_dim"],
            n_layers=LOCAL_MODEL_CONFIG["n_layers"],
            dropout=LOCAL_MODEL_CONFIG["dropout"],
        )
        
        # 加载权重
        checkpoint = torch.load(
            LOCAL_MODEL_CONFIG["model_path"],
            map_location=LOCAL_MODEL_CONFIG["device"]
        )
        
        # 兼容不同格式的模型文件
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # 如果 checkpoint 是包含 model_state_dict 的字典
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # 如果 checkpoint 直接就是 state_dict
                model.load_state_dict(checkpoint)
        else:
            print("❌ 模型文件格式不正确")
            return False
        
        model.to(LOCAL_MODEL_CONFIG["device"])
        model.eval()
        
        print(f"✅ 本地模型加载成功！设备：{LOCAL_MODEL_CONFIG['device']}")
        return True
        
    except Exception as e:
        print(f"❌ 加载本地模型失败：{e}")
        return False


def predict_with_local_model(text: str) -> Optional[Dict]:
    """
    使用本地 Bi-LSTM 模型预测单条文本的情感
    
    参数：
        text (str): 待分析的文本
        
    返回：
        dict: 包含 sentiment, confidence, analysis_time 的字典
        None: 如果预测失败
    """
    global model, vocab, preprocessor
    
    if model is None or vocab is None or preprocessor is None:
        return None
    
    try:
        start_time = time.time()
        
        # 检查组件是否正确加载
        if model is None:
            print("❌ 模型未加载")
            return None
        if vocab is None:
            print("❌ 词表未加载")
            return None
        if preprocessor is None:
            print("❌ 预处理器未加载")
            return None
        
        print(f"调试：开始预测，文本长度={len(text)}")
        print(f"调试：词表大小={len(vocab)}, 预处理器词表大小={len(preprocessor.vocab)}")
        
        # 文本预处理：转换为 ID 序列
        token_ids = preprocessor.text_to_sequence(text)
        print(f"调试：token_ids 长度={len(token_ids)}")
        
        # 转换为 tensor
        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(LOCAL_MODEL_CONFIG["device"])
        print(f"调试：tensor 形状={token_ids_tensor.shape}")
        
        # 模型推理
        with torch.no_grad():
            output, attention_weights = model(token_ids_tensor)
            print(f"调试：模型输出形状={output.shape}")
            prediction = torch.sigmoid(output).item()
            print(f"调试：预测值={prediction}")
        
        # 转换为情感标签和置信度
        sentiment = "正面" if prediction > 0.5 else "负面"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        
        analysis_time = time.time() - start_time
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence * 100, 2),
            "raw_score": round(prediction, 4),
            "analysis_time": round(analysis_time, 4)
        }
        
    except Exception as e:
        import traceback
        print(f"本地模型预测错误：{e}")
        print(f"详细错误：{traceback.format_exc()}")
        return None


# ==================== API 路由 ====================
@app.route('/api/local/analyze', methods=['POST'])
def analyze_text_local():
    """
    使用本地模型进行单条文本情感分析接口
    
    请求格式：
    {
        "text": "待分析的文本"
    }
    
    返回格式：
    {
        "success": true/false,
        "data": {
            "sentiment": "正面/负面",
            "confidence": 0-100,
            "raw_score": 0-1,
            "analysis_time": 秒数
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
        
        # 使用本地模型预测
        result = predict_with_local_model(text)
        
        if result:
            return jsonify({
                "success": True,
                "data": result,
                "model_type": "local_bilstm"
            })
        else:
            return jsonify({
                "success": False,
                "error": "本地模型预测失败，请检查模型文件是否完整"
            }), 500
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/local/health', methods=['GET'])
def local_health_check():
    """本地模型健康检查接口"""
    model_loaded = model is not None and vocab is not None and preprocessor is not None
    
    return jsonify({
        "status": "ok" if model_loaded else "degraded",
        "service": "本地 Bi-LSTM 情感分析服务",
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None,
        "preprocessor_loaded": preprocessor is not None,
        "device": LOCAL_MODEL_CONFIG["device"] if model_loaded else None
    })


# ==================== 主函数 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("本地 Bi-LSTM 情感分析服务启动中...")
    print("=" * 60)
    
    # 加载模型
    if load_local_model():
        print(f"API 地址：http://localhost:5001")
        print(f"单条分析接口：POST /api/local/analyze")
        print(f"健康检查接口：GET /api/local/health")
        print("=" * 60)
        
        # 启动 Flask 服务（使用端口 5002）
        app.run(host='0.0.0.0', port=5002, debug=False)
    else:
        print("=" * 60)
        print("❌ 模型加载失败，服务无法启动")
        print("请确保已训练模型并生成以下文件：")
        print(f"  1. {LOCAL_MODEL_CONFIG['vocab_path']}")
        print(f"  2. {LOCAL_MODEL_CONFIG['model_path']}")
        print("=" * 60)

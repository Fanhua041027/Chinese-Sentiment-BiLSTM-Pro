from flask import Flask, render_template, request, jsonify
import os
import sys
import torch
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pickle
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.bilstm_attention import SentimentModel
from src.data.domain_preprocessor import DomainAwareTextPreprocessor, DomainIdentifier

app = Flask(__name__)

model_path = 'checkpoints/best_model.pth'
vocab_path = 'dataset/processed/vocab.pkl'

model = None
vocab = None
device = None
model_loaded = False
domain_identifier = None
domain_preprocessor = None
max_len = 128

def load_vocab(vocab_path: str) -> dict:
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词表文件未找到: {vocab_path}")
    with open(vocab_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "vocab" in obj:
        return obj["vocab"]
    return obj

try:
    vocab = load_vocab(vocab_path)

    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.5
    max_len = 128

    model = SentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        n_layers=num_layers,
        dropout=dropout
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        model_loaded = True
        print("信息：基础模型加载成功")
    else:
        print(f"警告：模型文件未找到：{model_path}")

    domain_identifier = DomainIdentifier()
    domain_preprocessor = DomainAwareTextPreprocessor()
    domain_preprocessor.set_vocab(vocab)
    print("信息：领域识别器初始化成功")

except Exception as e:
    print(f"加载模型时出错：{str(e)}")

def preprocess_text(text):
    tokens = list(jieba.cut(text))
    token_ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids += [vocab.get('<PAD>', 0)] * (max_len - len(token_ids))
    return tokens, torch.tensor([token_ids], dtype=torch.long).to(device)

def predict(text):
    if not model_loaded or model is None:
        raise Exception("模型未加载，请先运行训练命令")

    tokens, input_ids = preprocess_text(text)
    with torch.no_grad():
        logits, attention_weights = model(input_ids)
        probability = torch.sigmoid(logits).item()
        sentiment = '正向' if probability >= 0.5 else '负向'

    attention = attention_weights.squeeze().cpu().numpy()
    valid_len = min(len(tokens), max_len)
    attention = attention[:valid_len]
    tokens = tokens[:valid_len]

    return sentiment, probability, tokens, attention

def generate_attention_heatmap(tokens, attention):
    plt.figure(figsize=(12, 6))
    sns.heatmap([attention], xticklabels=tokens, cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'})
    plt.title('注意力热力图')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/domain')
def domain_page():
    return render_template('domain_analysis.html')

@app.route('/api/domain/analyze', methods=['POST'])
def domain_analyze():
    if domain_preprocessor is None or domain_identifier is None:
        return jsonify({'error': '领域识别器未初始化'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    if not text or len(text.strip()) < 5:
        return jsonify({'error': '文本长度至少为5个字符'}), 400

    try:
        domain, confidence = domain_identifier.identify_with_confidence(text)

        tokens, input_ids = preprocess_text(text)
        with torch.no_grad():
            logits, _ = model(input_ids)
            prob = torch.sigmoid(logits).item()

        sentiment = 'positive' if prob >= 0.5 else 'negative'
        conf = prob if prob >= 0.5 else 1.0 - prob

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'sentiment_score': round(float(prob), 4),
            'confidence': round(float(conf), 4),
            'domain': domain,
            'detected_domain': domain,
            'domain_confidence': round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/analyze_with_domain', methods=['POST'])
def domain_analyze_with_domain():
    if domain_preprocessor is None:
        return jsonify({'error': '领域预处理器未初始化'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    domain = data.get('domain', 'mixed')

    if domain not in ['film', 'product', 'news', 'mixed']:
        return jsonify({'error': f'无效的领域: {domain}'}), 400

    try:
        tokens, input_ids = preprocess_text(text)
        with torch.no_grad():
            logits, _ = model(input_ids)
            prob = torch.sigmoid(logits).item()

        sentiment = 'positive' if prob >= 0.5 else 'negative'
        conf = prob if prob >= 0.5 else 1.0 - prob

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'sentiment_score': round(float(prob), 4),
            'confidence': round(float(conf), 4),
            'domain': domain
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/batch_analyze', methods=['POST'])
def domain_batch_analyze():
    if domain_preprocessor is None:
        return jsonify({'error': '领域预处理器未初始化'}), 503

    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': '缺少texts字段'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts必须是非空列表'}), 400

    if len(texts) > 100:
        return jsonify({'error': '批量大小不能超过100'}), 400

    auto_detect = data.get('auto_detect_domain', True)

    try:
        results = []
        domain_stats = {'film': 0, 'product': 0, 'news': 0, 'mixed': 0}
        positive_count = 0
        negative_count = 0

        for text in texts:
            if auto_detect:
                domain, _ = domain_identifier.identify_with_confidence(text)
            else:
                domain = 'mixed'

            tokens, input_ids = preprocess_text(text)
            with torch.no_grad():
                logits, _ = model(input_ids)
                prob = torch.sigmoid(logits).item()

            sentiment = 'positive' if prob >= 0.5 else 'negative'
            conf = prob if prob >= 0.5 else 1.0 - prob

            results.append({
                'text': text,
                'sentiment': sentiment,
                'sentiment_score': round(float(prob), 4),
                'confidence': round(float(conf), 4),
                'domain': domain
            })

            domain_stats[domain] = domain_stats.get(domain, 0) + 1
            if sentiment == 'positive':
                positive_count += 1
            else:
                negative_count += 1

        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'positive': positive_count,
                'negative': negative_count,
                'domain_distribution': domain_stats
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/detect_domain', methods=['POST'])
def detect_domain():
    if domain_identifier is None:
        return jsonify({'error': '领域识别器未初始化'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400

    text = data['text']
    if not text:
        return jsonify({'error': '文本不能为空'}), 400

    try:
        domain, confidence = domain_identifier.identify_with_confidence(text)

        all_domains = {}
        for d in ['film', 'product', 'news', 'mixed']:
            if d == domain:
                all_domains[d] = confidence
            else:
                all_domains[d] = round((1 - confidence) / 3, 4)

        return jsonify({
            'text': text,
            'domain': domain,
            'confidence': round(float(confidence), 4),
            'all_domains': all_domains
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/domain/health', methods=['GET'])
def domain_health():
    return jsonify({
        'status': 'ok' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'domain_support': True,
        'available_domains': ['film', 'product', 'news', 'mixed']
    })

@app.route('/api/domain/stats', methods=['GET'])
def domain_stats():
    return jsonify({
        'supported_domains': [
            {'name': 'film', 'description': '影评领域 - 电影、电视剧、综艺等评论'},
            {'name': 'product', 'description': '商品评论领域 - 购物、使用体验等评论'},
            {'name': 'news', 'description': '新闻领域 - 时事、政策、财经等新闻'},
            {'name': 'mixed', 'description': '混合/通用领域 - 未分类或跨领域文本'}
        ],
        'model_info': {
            'domain_adaptive': model_loaded,
            'auto_detection': True,
            'batch_support': True,
            'max_batch_size': 100
        }
    })

# 预测API
@app.route('/predict', methods=['POST'])
def predict_api():
    text = request.form['text']
    if not text or len(text.strip()) < 5:
        return jsonify({'error': '文本长度至少为5个字符'})
    
    try:
        sentiment, probability, tokens, attention = predict(text)
        attention_image = generate_attention_heatmap(tokens, attention)
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'probability': round(probability, 4),
            'attention_image': attention_image,
            'tokens': tokens,
            'attention': [round(float(a), 4) for a in attention]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# 模型性能页面
@app.route('/performance')
def performance():
    # 读取评估报告
    try:
        with open('dataset/reports/test_report.txt', 'r', encoding='utf-8') as f:
            bilstm_report = f.read()

        with open('dataset/reports/baseline_report.txt', 'r', encoding='utf-8') as f:
            nb_report = f.read()

        with open('dataset/reports/bert_report.txt', 'r', encoding='utf-8') as f:
            bert_report = f.read()

        # 读取图表
        import base64
        import os

        def encode_image(path):
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            return None

        model_comparison = encode_image('dataset/figures/model_comparison.png')
        training_curve = encode_image('dataset/figures/training_curve.png')

        return render_template('performance.html',
                              bilstm_report=bilstm_report,
                              nb_report=nb_report,
                              bert_report=bert_report,
                              model_comparison=model_comparison,
                              training_curve=training_curve)
    except Exception as e:
        return f'Error loading performance data: {str(e)}'

# 情感趋势分析页面
@app.route('/trend')
def trend_page():
    return render_template('trend.html')

# 情感趋势分析API
@app.route('/trend_api')
def trend_api():
    try:
        from src.sentiment_trend import SentimentTrendAnalyzer, CONFIG

        analyzer = SentimentTrendAnalyzer()

        analyzer.load_and_process_data()

        analyzer.aggregate_time_series(agg_freq=CONFIG["aggregation"])

        dates = [str(d.date()) if hasattr(d, 'date') else str(d) for d in analyzer.time_series.index]
        sentiment_values = analyzer.time_series.values.tolist()
        rolling_mean = analyzer.time_series.rolling(window=4, min_periods=1).mean().values.tolist()

        predictions = analyzer.predict_future(days=CONFIG["prediction_days"])
        predictions = predictions.tolist()

        last_date = analyzer.time_series.index[-1]
        prediction_dates = []
        for i in range(1, CONFIG["prediction_days"] + 1):
            pred_date = last_date + pd.Timedelta(weeks=i)
            prediction_dates.append(str(pred_date.date()) if hasattr(pred_date, 'date') else str(pred_date))

        stats = analyzer.generate_trend_report()

        historical_dates = dates[-30:] if len(dates) > 30 else dates
        historical_values = sentiment_values[-30:] if len(sentiment_values) > 30 else sentiment_values

        return jsonify({
            'dates': dates,
            'sentiment_values': sentiment_values,
            'rolling_mean': rolling_mean,
            'predictions': predictions,
            'prediction_dates': prediction_dates,
            'historical_dates': historical_dates,
            'historical_values': historical_values,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 创建templates目录
    os.makedirs('templates', exist_ok=True)
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)

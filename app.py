from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import json
import re

app = Flask(__name__)

model_path = 'imdb_rnn_model.h5'
try:
    model = load_model(model_path)
    print(f" Model loaded successfully!")
    print(f" Model input shape: {model.input_shape}")
    print(f" Model output shape: {model.output_shape}")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

vocab_size = 10000
max_length = 500

word_index = {}
try:
    if os.path.exists('word_index.json'):
        with open('word_index.json', 'r') as f:
            word_index = json.load(f)
        print(f"✅ Loaded vocabulary with {len(word_index)} words")
except Exception as e:
    print(f"⚠️  Could not load word_index.json: {e}")


tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

if word_index:
    reverse_word_index = {v: k for k, v in word_index.items()}
else:
    reverse_word_index = {}
    
    common_words = ['the', 'a', 'and', 'to', 'of', 'i', 'is', 'this', 'that', 'it', 
                    'was', 'in', 'for', 'be', 'with', 'as', 'movie', 'film', 'not', 'good']
    word_index = {word: idx + 1 for idx, word in enumerate(common_words)}
    reverse_word_index = {v: k for k, v in word_index.items()}

def clean_text(text):
    """Clean and normalize text for better tokenization"""
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, max_len=500):
    """
    Preprocess text for model prediction.
    Handles both IMDB word indexing and tokenizer approaches.
    """
    text = clean_text(text)
    
    if word_index:
        sequences = [[word_index.get(word, 2) for word in text.split()]]  # 2 is OOV token
    else:
        sequences = tokenizer.texts_to_sequences([text])
        if not sequences or not sequences[0]:
            sequences = [[ord(c) % 256 for c in text[:100]]]
    
    if not sequences or len(sequences[0]) == 0:
        sequences = [[1]]  
        padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded.astype(np.float32)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction on the review text
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Make sure the H5 file exists.'}), 500
        
        data = request.get_json()
        review_text = data.get('review', '').strip()
        
        if not review_text:
            return jsonify({'error': 'Please enter a movie review'}), 400
        
        processed_text = preprocess_text(review_text, max_length)
        
        print(f" Input text: {review_text[:100]}")
        print(f" Processed shape: {processed_text.shape}")
        print(f" Processed data type: {processed_text.dtype}")
        
        try:
            prediction = model.predict(processed_text, verbose=0)
            print(f" Raw prediction: {prediction}")
        except Exception as pred_error:
            print(f" Prediction error: {pred_error}")
            return jsonify({'error': f'Model prediction error: {str(pred_error)}'}), 500
        
        if prediction is None or len(prediction) == 0:
            return jsonify({'error': 'Model returned empty prediction'}), 500
        
        try:
            if isinstance(prediction, np.ndarray):
                score = float(prediction.flatten()[0])
            else:
                score = float(prediction)
        except (TypeError, IndexError) as e:
            print(f" Error extracting score: {e}")
            return jsonify({'error': f'Failed to process model output: {str(e)}'}), 500
        
        if not isinstance(score, (int, float)) or np.isnan(score):
            return jsonify({'error': f'Invalid prediction score: {score}'}), 500
        
        score = max(0.0, min(1.0, score))
        
        if score > 0.5:
            sentiment = "NEGATIVE"
            confidence_pct = score * 100
        else:
            sentiment = "POSITIVE"
            confidence_pct = (1 - score) * 100
        
        print(f" Sentiment: {sentiment}, Confidence: {confidence_pct:.2f}%")
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence_pct, 2),
            'raw_score': round(score, 4),
            'review': review_text[:100] + '...' if len(review_text) > 100 else review_text
        })
    
    except Exception as e:
        print(f" Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'Model server is running'}), 200

if __name__ == '__main__':
    print("="*60)
    print("🎬 Movie Sentiment Analysis Server Starting...")
    print("="*60)
    print(f" Model file: {model_path} - {' Found' if os.path.exists(model_path) else ' Not found'}")
    print(f" Vocabulary size: {vocab_size}")
    print(f" Max sequence length: {max_length}")
    print("="*60)
    print(" Open your browser at: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

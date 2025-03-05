from flask import Flask, render_template, request, jsonify
import torch
import os
from hate_speech_model import load_model, predict_hate_speech, TextPreprocessor

app = Flask(__name__)

# Load the model
MODEL_PATH = 'hate_speech_model.pt'  # This should be the saved model file, not the dataset

# Initialize model and vocab as None
model = None
vocab = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, vocab
    
    # Load model if not already loaded
    if model is None or vocab is None:
        try:
            print(f"Loading model from {MODEL_PATH}")
            model, vocab = load_model(MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            return jsonify({'error': f"Failed to load model: {str(e)}"})
    
    # Get text from request
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'})
    
    text = data['text']
    
    try:
        # Preprocess text
        preprocessed_text = TextPreprocessor.preprocess_text(text)
        
        # Make prediction
        prediction, confidence = predict_hate_speech(preprocessed_text, model, vocab)
        
        # Map prediction to class name
        class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        prediction_class = class_names.get(prediction, 'Unknown')
        
        return jsonify({
            'prediction': prediction,
            'class': prediction_class,
            'confidence': float(confidence),
            'text': text
        })
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error during prediction: {str(e)}")
        print(traceback_str)
        return jsonify({'error': f"Error during prediction: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)

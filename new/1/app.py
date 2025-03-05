from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import os
import pickle
from hate_speech_model import LSTM_CNN, preprocess_text, text_to_sequence
from transformers import RobertaTokenizer
from roberta_model import RoBERTaClassifier, preprocess_for_roberta

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables to store the models and vocabulary
lstm_cnn_model = None
roberta_model = None
roberta_tokenizer = None
vocab = None

# Model selection flag - 'roberta' or 'lstm_cnn'
active_model = 'roberta'  # Set RoBERTa as default

def load_models():
    global lstm_cnn_model, roberta_model, roberta_tokenizer, vocab
    
    try:
        # Load LSTM-CNN model
        vocab_path = './vocab.pkl'
        print(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded with {len(vocab)} entries")
        
        model_path = './hate_speech_model_lstm_cnn.pth'
        print(f"Loading LSTM-CNN model from {model_path}")
        model_info = torch.load(model_path, map_location=device)
        
        # Initialize LSTM-CNN model
        lstm_cnn_model = LSTM_CNN(
            model_info['vocab_size'],
            model_info['embed_dim'],
            model_info['lstm_hidden_dim'],
            model_info['cnn_hidden_dim'],
            model_info['num_classes'],
            model_info['dropout']
        )
        lstm_cnn_model.load_state_dict(model_info['state_dict'])
        lstm_cnn_model.to(device)
        lstm_cnn_model.eval()
        print("LSTM-CNN model loaded successfully!")

        # Load RoBERTa model
        roberta_path = './hate_speech_model_roberta.pth'
        print(f"Loading RoBERTa tokenizer")
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        if os.path.exists(roberta_path):
            print(f"Loading saved RoBERTa model from {roberta_path}")
            roberta_info = torch.load(roberta_path, map_location=device)
            roberta_model = RoBERTaClassifier()
            roberta_model.load_state_dict(roberta_info['state_dict'])
        else:
            print("Initializing new RoBERTa model")
            roberta_model = RoBERTaClassifier()
        
        roberta_model.to(device)
        roberta_model.eval()
        print("RoBERTa model loaded successfully!")
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load models at startup
if not load_models():
    print("WARNING: Failed to load models!")

# Simple HTML template for frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Hate Speech Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .container { 
            display: flex; 
            flex-direction: column; 
            gap: 20px;
        }
        textarea { 
            width: 100%; 
            height: 100px; 
            padding: 10px; 
            margin-top: 10px;
        }
        button { 
            padding: 10px 20px; 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            cursor: pointer; 
            width: 200px;
        }
        button:hover { background-color: #45a049; }
        .result { 
            margin-top: 20px; 
            padding: 15px; 
            border-radius: 5px;
            display: none;
        }
        .hate-speech { background-color: #ffdddd; }
        .non-hate-speech { background-color: #ddffdd; }
        .spinner { display: none; }
        .model-selector {
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hate Speech Detection</h1>
        <div class="model-selector">
            <label>Select Model:</label>
            <select id="model-selector" onchange="changeModel()">
                <option value="roberta">RoBERTa (Better accuracy)</option>
                <option value="lstm_cnn">LSTM-CNN (Faster)</option>
            </select>
        </div>
        <div>
            <label for="text-input">Enter text to analyze:</label>
            <textarea id="text-input" placeholder="Type or paste text here..."></textarea>
        </div>
        <div>
            <button onclick="detectHateSpeech()">Analyze Text</button>
            <span class="spinner" id="loading">Processing...</span>
        </div>
        <div id="result" class="result"></div>
    </div>

    <script>
        // Set initial model selection from server
        document.getElementById('model-selector').value = 'roberta';
        
        function changeModel() {
            const model = document.getElementById('model-selector').value;
            fetch('/set_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model: model })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Model changed:', data.model);
            });
        }
        
        function detectHateSpeech() {
            const text = document.getElementById('text-input').value.trim();
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'inline';
            
            // Clear previous result
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'none';
            resultDiv.className = 'result';
            
            // Send request to API
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
                
                // Display result
                resultDiv.style.display = 'block';
                resultDiv.className = data.label === 'Hate Speech' ? 
                                     'result hate-speech' : 'result non-hate-speech';
                
                resultDiv.innerHTML = `
                    <h3>Result:</h3>
                    <p><strong>Classification:</strong> ${data.label}</p>
                    <p><strong>Confidence:</strong> ${(data.score * 100).toFixed(2)}%</p>
                    <p><strong>Model:</strong> ${data.model}</p>
                `;
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error);
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/set_model', methods=['POST'])
def set_model():
    global active_model
    data = request.json
    if data and 'model' in data:
        active_model = data['model']
    return jsonify({"model": active_model})

@app.route('/predict', methods=['POST'])
def predict():
    global active_model
    
    # Get text from request
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    try:
        if active_model == 'roberta':
            if roberta_model is None or roberta_tokenizer is None:
                return jsonify({"error": "RoBERTa model not loaded"}), 500
                
            # Preprocess text for RoBERTa
            processed_text = preprocess_for_roberta(text)
            
            # Tokenize
            encoded = roberta_tokenizer(
                processed_text,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = roberta_model(input_ids, attention_mask)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
        else:  # LSTM-CNN model
            if lstm_cnn_model is None or vocab is None:
                return jsonify({"error": "LSTM-CNN model not loaded"}), 500
                
            # Preprocess text for LSTM-CNN
            processed_text = preprocess_text(text)
            
            # Convert to sequence
            sequence = text_to_sequence(processed_text, vocab, max_len=100)
            sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = lstm_cnn_model(sequence_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
        
        # Prepare response
        result = {
            'label': 'Hate Speech' if predicted_class == 1 else 'Non-Hate Speech',
            'score': float(confidence),
            'model': active_model
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

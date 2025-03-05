import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import pickle
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition
class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout=0.5):
        super(LSTM_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(lstm_hidden_dim*2, cnn_hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_hidden_dim, num_classes)
    
    def forward(self, x):
        # Handle both float and long inputs
        if x.dtype == torch.float32:
            x = x.long()
        
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, lstm_hidden_dim*2]
        
        # Reshape for CNN: [batch_size, lstm_hidden_dim*2, seq_len]
        lstm_out = lstm_out.permute(0, 2, 1)
        
        # Apply CNN
        conv_out = F.relu(self.conv1d(lstm_out))  # [batch_size, cnn_hidden_dim, seq_len]
        
        # Pooling
        pooled = self.pool(conv_out).squeeze(-1)  # [batch_size, cnn_hidden_dim]
        
        # Dropout and final classification
        dropped = self.dropout(pooled)
        output = self.fc(dropped)  # [batch_size, num_classes]
        
        return output

# Preprocess text function
def preprocess_text(text):
    """Preprocess text by removing URLs, mentions, hashtags, numbers, punctuation, and converting to lowercase"""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

# Function to convert text to sequence
def text_to_sequence(text, vocab_dict, max_len=100):
    """Convert text to a sequence of token indices using the vocabulary"""
    tokens = text.split()
    sequence = [vocab_dict.get(token, 0) for token in tokens]  # Use 0 for unknown tokens
    sequence = sequence[:max_len]  # Truncate if longer than max_len
    sequence += [0] * (max_len - len(sequence))  # Pad if shorter than max_len
    return sequence

def load_model_and_vocab(model_path, vocab_path):
    """Load the model and vocabulary from files."""
    try:
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # Load model
        model_info = torch.load(model_path, map_location=device)
        model = LSTM_CNN(
            model_info['vocab_size'],
            model_info['embed_dim'],
            model_info['lstm_hidden_dim'],
            model_info['cnn_hidden_dim'],
            model_info['num_classes'],
            model_info['dropout']
        )
        model.load_state_dict(model_info['state_dict'])
        model.to(device)
        model.eval()
        
        return model, vocab
    except Exception as e:
        print(f"Error loading model or vocabulary: {e}")
        return None, None

def classify_text(model, vocab, text):
    """Classify text as hate speech or non-hate speech."""
    processed_text = preprocess_text(text)
    sequence = text_to_sequence(processed_text, vocab, max_len=100)
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    labels = ['Non-Hate Speech', 'Hate Speech']
    return {
        'text': text,
        'processed_text': processed_text,
        'prediction': labels[predicted_class],
        'confidence': confidence
    }

if __name__ == "__main__":
    # Example usage
    model_path = './hate_speech_model_lstm_cnn.pth'
    vocab_path = './vocab.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        model, vocab = load_model_and_vocab(model_path, vocab_path)
        
        if model and vocab:
            test_texts = [
                "I love how diverse our community is becoming!",
                "I hate all people from that country, they should go back where they came from",
                "The movie was terrible, I hated every minute of it",
                "People of that religion are all terrorists and should be banned",
                "Everyone deserves equal rights regardless of their background"
            ]
            
            for text in test_texts:
                result = classify_text(model, vocab, text)
                print("="*70)
                print(f"Text: {result['text']}")
                print(f"Processed: {result['processed_text']}")
                print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")

"""
This script creates minimal model files for demo purposes
when you don't want to train full models from scratch.
"""
import torch
import torch.nn as nn
import pickle
import os
from hate_speech_model import LSTM_CNN
from roberta_model import RoBERTaClassifier

def create_minimal_lstm_cnn():
    print("Creating minimal LSTM-CNN model...")
    vocab_size = 5000
    embed_dim = 100
    lstm_hidden_dim = 128
    cnn_hidden_dim = 128
    num_classes = 2
    dropout = 0.5
    
    model = LSTM_CNN(vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout)
    
    model_info = {
        'state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'lstm_hidden_dim': lstm_hidden_dim,
        'cnn_hidden_dim': cnn_hidden_dim,
        'num_classes': num_classes,
        'dropout': dropout
    }
    
    torch.save(model_info, 'hate_speech_model_lstm_cnn.pth')
    print("LSTM-CNN model saved to: hate_speech_model_lstm_cnn.pth")
    
    # Create a minimal vocab
    vocab = {' ': 0, 'the': 1, 'hate': 2, 'speech': 3, 'detection': 4}
    for i in range(5, 5000):
        vocab[f'word_{i}'] = i
    
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    print("Minimal vocabulary saved to: vocab.pkl")

def create_minimal_roberta():
    print("Creating minimal RoBERTa model...")
    model = RoBERTaClassifier()
    
    model_info = {
        'state_dict': model.state_dict(),
        'epoch': 0,
        'val_f1': 0.5
    }
    
    torch.save(model_info, 'hate_speech_model_roberta.pth')
    print("RoBERTa model saved to: hate_speech_model_roberta.pth")

if __name__ == "__main__":
    print("Creating minimal model files for demo...")
    
    create_minimal_lstm_cnn()
    create_minimal_roberta()
    
    print("\nAll done! You can now run the app with these minimal model files.")
    print("For better results, train real models using train_models.py")

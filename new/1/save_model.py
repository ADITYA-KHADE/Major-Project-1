import torch
import pickle
from hate_speech_model import LSTM_CNN
import os

def create_and_save_demo_model():
    """Create and save a demonstration model and vocabulary"""
    print("Creating demonstration model and vocabulary...")
    
    # Define parameters
    vocab_size = 5000
    embed_dim = 100
    lstm_hidden_dim = 128
    cnn_hidden_dim = 128
    num_classes = 2
    dropout = 0.5
    
    # Create a small vocabulary for demonstration
    demo_vocab = {' ': 0}  # Padding token
    common_words = [
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'who', 'how', 'when', 'where', 'which', 'would', 'could', 'should',
        'you', 'they', 'we', 'he', 'she', 'it', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'not', 'no', 'yes', 'why', 'can', 'will', 'just', 'more', 'some',
        'hate', 'love', 'people', 'person', 'man', 'woman', 'child', 'say',
        'said', 'like', 'want', 'know', 'think', 'make', 'time', 'see',
        'go', 'come', 'take', 'get', 'give', 'tell', 'work', 'call', 'try',
        'bad', 'good', 'new', 'first', 'last', 'long', 'great', 'little',
        'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small',
        'large', 'next', 'early', 'young', 'important', 'few', 'public'
    ]
    
    # Add words to vocabulary
    for i, word in enumerate(common_words):
        demo_vocab[word] = i + 1
    
    print(f"Created vocabulary with {len(demo_vocab)} words")
    
    # Create model
    model = LSTM_CNN(vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout)
    
    # Save vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(demo_vocab, f)
    print(f"Saved vocabulary to vocab.pkl")
    
    # Save model
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
    print(f"Saved model to hate_speech_model_lstm_cnn.pth")
    
    print("Done!")

if __name__ == "__main__":
    create_and_save_demo_model()

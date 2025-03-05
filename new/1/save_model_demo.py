from hate_speech_model import LSTM_CNN
import torch
import pickle

# Create model
print("Creating LSTM-CNN model...")
model = LSTM_CNN(5000, 100, 128, 128, 2, 0.5)

# Create model info dictionary
model_info = {
    'state_dict': model.state_dict(),
    'vocab_size': 5000,
    'embed_dim': 100,
    'lstm_hidden_dim': 128,
    'cnn_hidden_dim': 128,
    'num_classes': 2,
    'dropout': 0.5
}

# Save model
torch.save(model_info, 'hate_speech_model_lstm_cnn.pth')
print("Model saved to hate_speech_model_lstm_cnn.pth")

# Create vocabulary
vocab = {' ': 0}
words = ['the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'hate', 'love',
         'you', 'they', 'we', 'he', 'she', 'it', 'is', 'are', 'was', 'were', 'people', 'person']
for i, word in enumerate(words):
    vocab[word] = i + 1

# Save vocabulary
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
print("Vocabulary saved to vocab.pkl with", len(vocab), "entries")
print("Demo model and vocab created.")

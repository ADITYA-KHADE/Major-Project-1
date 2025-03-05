import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import wordnet
import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

class TextPreprocessor:
    """Class for text preprocessing tasks"""
    
    @staticmethod
    def preprocess_text(text):
        """
        Preprocess text by removing URLs, mentions, hashtags, numbers, punctuation,
        and converting to lowercase.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@user)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class DataAugmenter:
    """Class for text data augmentation"""
    
    def __init__(self, aug_type='synonym', aug_count=5):
        """Initialize data augmenter."""
        self.aug_type = aug_type
        self.aug_count = aug_count
        
        if aug_type == 'synonym':
            self.augmenter = naw.SynonymAug(aug_src='wordnet', aug_max=3)
        elif aug_type == 'contextual':
            # This requires a pre-trained model
            self.augmenter = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        else:
            raise ValueError(f"Unsupported augmentation type: {aug_type}")
    
    def augment_text(self, text):
        """Augment a single text."""
        if not text or len(text.strip()) == 0:
            return []
        
        try:
            augmented_texts = [self.augmenter.augment(text) for _ in range(self.aug_count)]
            return augmented_texts
        except Exception as e:
            logger.warning(f"Error augmenting text: {e}")
            return []
    
    def augment_dataset(self, texts, labels):
        """Augment a dataset."""
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Add original text
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Add augmented texts
            aug_texts = self.augment_text(text)
            augmented_texts.extend(aug_texts)
            augmented_labels.extend([label] * len(aug_texts))
        
        return augmented_texts, augmented_labels

class HateSpeechDatasetLSTM_CNN(Dataset):
    """Dataset for LSTM+CNN model."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class LSTM_CNN(nn.Module):
    """LSTM + CNN model for text classification"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim_lstm, hidden_dim_cnn, output_dim, dropout=0.5):
        """Initialize model."""
        super(LSTM_CNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim_lstm, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim_lstm * 2, hidden_dim_cnn, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim_cnn, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass."""
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        lstm_out, _ = self.lstm(embedded)
        # lstm_out shape: [batch_size, seq_len, hidden_dim_lstm * 2]
        
        # Transpose for CNN
        lstm_out = lstm_out.permute(0, 2, 1)
        # lstm_out shape: [batch_size, hidden_dim_lstm * 2, seq_len]
        
        conv_out = self.conv(lstm_out)
        # conv_out shape: [batch_size, hidden_dim_cnn, seq_len]
        
        pooled = self.pool(conv_out).squeeze(2)
        # pooled shape: [batch_size, hidden_dim_cnn]
        
        dropped = self.dropout(pooled)
        output = self.fc(dropped)
        # output shape: [batch_size, output_dim]
        
        return output

def text_to_sequence(text, vocab, max_len=100):
    """Convert text to sequence of token IDs."""
    if not isinstance(text, str):
        text = ""
    
    # Tokenize
    tokens = text.lower().split()
    
    # Convert to IDs
    ids = [vocab.get(token, 1) for token in tokens]  # 1 is <UNK>
    
    # Pad or truncate
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))  # 0 is <PAD>
    else:
        ids = ids[:max_len]
    
    return ids

def build_vocab(texts, max_vocab_size=10000):
    """Build vocabulary from texts."""
    word_counts = {}
    
    # Count words
    for text in texts:
        if not isinstance(text, str):
            continue
        
        tokens = text.lower().split()
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, _) in enumerate(sorted_words[:max_vocab_size-2]):
        vocab[word] = i + 2
    
    return vocab

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    """Train the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Check if loaders have data
    if len(train_loader) == 0:
        print("Error: Training loader is empty. Cannot train model.")
        return model
        
    print(f"Training with {len(train_loader)} batches, validation with {len(val_loader)} batches")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training...")
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move batch to device
            try:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    input_ids, labels = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    
                    # Debug info
                    if i == 0:
                        print(f"Batch shapes - Input: {input_ids.shape}, Labels: {labels.shape}")
                        print(f"Label values: {labels.tolist()}")
                    
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Print progress
                    if i % 5 == 0:
                        print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
                else:
                    print(f"Unexpected batch format: {type(batch)}")
            except Exception as e:
                print(f"Error in training batch {i}: {e}")
                import traceback
                traceback.print_exc()
            
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Evaluate on validation set if available
        if len(val_loader) > 0:
            print("Evaluating...")
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    # Move batch to device
                    try:
                        if isinstance(batch, (list, tuple)) and len(batch) == 2:
                            input_ids, labels = batch
                            input_ids = input_ids.to(device)
                            labels = labels.to(device)
                            
                            outputs = model(input_ids)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            val_batch_count += 1
                            
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            
                            # Print progress
                            if i % 5 == 0:
                                print(f"  Validation Batch {i}/{len(val_loader)}")
                        else:
                            print(f"Unexpected validation batch format: {type(batch)}")
                    except Exception as e:
                        print(f"Error in validation batch {i}: {e}")
                    
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            accuracy = correct / total if total > 0 else 0
            print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f} ({correct}/{total})')
        else:
            print("No validation data available. Skipping validation step.")
    
    return model

def augment_and_prepare_data(df, text_column='clean_tweet', label_column='label', aug_count=5, test_size=0.2):
    """Augment and prepare data for training."""
    # Reset index to ensure contiguous integer indexing
    df = df.reset_index(drop=True)
    print(f"Starting data preparation with {len(df)} samples")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(df[text_column])
    print(f"Vocabulary size: {len(vocab)}")
    
    # Augment data - more aggressive for small datasets
    print("Augmenting data...")
    if len(df) < 20:
        print("Small dataset detected. Using more aggressive augmentation.")
        aug_count = max(aug_count, 15)  # At least 15 augmentations per sample for very small datasets
    
    augmenter = DataAugmenter(aug_type='synonym', aug_count=aug_count)
    augmented_texts = []
    augmented_labels = []
    
    for i, text in enumerate(df[text_column]):
        try:
            augmented = augmenter.augment_text(text)
            if len(augmented) > 0:
                augmented_texts.extend(augmented)
                augmented_labels.extend([df[label_column].iloc[i]] * len(augmented))
            
            # Print progress every 10 samples
            if i % 10 == 0:
                print(f"Augmented {i}/{len(df)} samples, generated {len(augmented_texts)} texts so far")
        except Exception as e:
            print(f"Error augmenting text at index {i}: {e}")
    
    print(f"Augmentation complete. Generated {len(augmented_texts)} new samples")
    
    # Create augmented DataFrame
    df_augmented = pd.DataFrame({text_column: augmented_texts, label_column: augmented_labels})
    
    # Combine and reset index
    df_combined = pd.concat([df, df_augmented]).reset_index(drop=True)
    print(f"Combined dataset size: {len(df_combined)} samples")
    
    # Convert texts to sequences
    print("Converting texts to sequences...")
    X_combined = []
    for i, text in enumerate(df_combined[text_column]):
        try:
            seq = text_to_sequence(text, vocab, max_len=100)
            X_combined.append(seq)
            # Print progress every 100 samples
            if i % 100 == 0 and i > 0:
                print(f"Converted {i}/{len(df_combined)} texts to sequences")
        except Exception as e:
            print(f"Error converting text at index {i}: {e}")
            # Add a zero sequence as fallback
            X_combined.append([0] * 100)
    
    X_combined = torch.tensor(X_combined, dtype=torch.long)
    print(f"Tensor shape after conversion: {X_combined.shape}")
    
    # For very small datasets, use a smaller validation set or no validation
    if len(df_combined) < 10:
        print("Warning: Very small dataset. Using 90% for training, 10% for validation.")
        test_size = 0.1
    
    # Split data
    print("Splitting data into train and validation sets...")
    try:
        # For very small datasets, stratification might fail
        if len(df_combined) < 10 or len(df_combined[label_column].unique()) < 2:
            print("Dataset too small for stratified split. Using random split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_combined, df_combined[label_column], test_size=test_size, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_combined, df_combined[label_column], test_size=test_size, 
                random_state=42, stratify=df_combined[label_column]
            )
        print(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    except ValueError as e:
        print(f"Stratified split failed: {e}. Falling back to random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, df_combined[label_column], test_size=test_size, random_state=42
        )
        print(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    
    # Adjust batch size for small datasets
    batch_size = 16
    if len(X_train) < 32:
        batch_size = max(1, len(X_train) // 2)
        print(f"Small training set. Adjusting batch size to {batch_size}")
    
    # Create datasets
    print("Creating PyTorch datasets...")
    train_dataset = HateSpeechDatasetLSTM_CNN(X_train, y_train)
    val_dataset = HateSpeechDatasetLSTM_CNN(X_val, y_val)
    
    # Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("Data preparation complete!")
    return train_loader, val_loader, vocab

def predict_hate_speech(text, model, vocab, max_len=100):
    """Predict if text contains hate speech."""
    # Preprocess text
    if isinstance(text, str):
        text = TextPreprocessor.preprocess_text(text)
    
    # Convert to sequence
    sequence = text_to_sequence(text, vocab, max_len)
    sequence_tensor = torch.tensor([sequence], dtype=torch.long)
    
    # Make prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sequence_tensor = sequence_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()

def save_model(model, vocab, file_path='hate_speech_model.pt'):
    """Save model and vocabulary."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_params': {
            'vocab_size': model.embedding.num_embeddings,
            'embedding_dim': model.embedding.embedding_dim,
            'hidden_dim_lstm': model.lstm.hidden_size,
            'hidden_dim_cnn': model.conv.out_channels,
            'output_dim': model.fc.out_features,
            'dropout': model.dropout.p
        }
    }, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Load model and vocabulary from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    # Load the saved state
    state = torch.load(file_path, map_location=torch.device('cpu'))
    
    # Extract model parameters
    vocab = state['vocab']
    model_params = state['model_params']
    
    # Create model with the same parameters as when it was saved
    vocab_size = model_params['vocab_size']
    embedding_dim = model_params['embedding_dim']
    hidden_dim_lstm = model_params['hidden_dim_lstm']
    hidden_dim_cnn = model_params['hidden_dim_cnn']
    output_dim = model_params['output_dim']
    dropout = model_params['dropout']
    
    # Initialize model with the same architecture
    model = LSTM_CNN(
        vocab_size, 
        embedding_dim, 
        hidden_dim_lstm, 
        hidden_dim_cnn, 
        output_dim, 
        dropout
    )
    
    # Load the state dict
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    return model, vocab

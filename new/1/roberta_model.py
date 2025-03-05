import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import pickle
import os
from transformers import RobertaModel, RobertaTokenizer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoBERTaClassifier(nn.Module):
    def __init__(self, pretrained_model="roberta-base", num_classes=2, dropout_rate=0.3):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout_rate)
        # RoBERTa hidden size is 768 for base model
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the RoBERTa classifier
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention masks
            
        Returns:
            Output logits
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def preprocess_for_roberta(text):
    """
    Minimal preprocessing for RoBERTa - it handles punctuation and capitalization well
    so we only need to clean obvious problematic elements
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Keep some punctuation that RoBERTa can use for context
    # but remove others that might introduce noise
    text = re.sub(r'[^\w\s\.\,\!\?\'\"\-]', '', text)
    return text

def load_roberta_model(model_path=None):
    """
    Load a pretrained RoBERTa model, either from local path or Hugging Face
    
    Args:
        model_path: Path to saved model or None to load from Hugging Face
        
    Returns:
        model: Loaded RoBERTa model
        tokenizer: Corresponding tokenizer
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    if model_path and os.path.exists(model_path):
        print(f"Loading saved RoBERTa model from {model_path}")
        model_info = torch.load(model_path, map_location=device)
        model = RoBERTaClassifier()
        model.load_state_dict(model_info['state_dict'])
    else:
        print("Initializing new RoBERTa model")
        model = RoBERTaClassifier()
    
    model.to(device)
    model.eval()
    return model, tokenizer

def classify_text_roberta(model, tokenizer, text, max_length=128):
    """
    Classify text using RoBERTa model
    
    Args:
        model: RoBERTa classifier model
        tokenizer: RoBERTa tokenizer
        text: Input text to classify
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with classification results
    """
    processed_text = preprocess_for_roberta(text)
    
    # Tokenize the text
    encoded_input = tokenizer(
        processed_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
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
    model_path = './hate_speech_model_roberta.pth'
    
    model, tokenizer = load_roberta_model(model_path if os.path.exists(model_path) else None)
    
    test_texts = [
        "I love how diverse our community is becoming!",
        "I hate all people from that country, they should go back where they came from",
        "The movie was terrible, I hated every minute of it",
        "People of that religion are all terrorists and should be banned",
        "Everyone deserves equal rights regardless of their background"
    ]
    
    for text in test_texts:
        result = classify_text_roberta(model, tokenizer, text)
        print("="*70)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")

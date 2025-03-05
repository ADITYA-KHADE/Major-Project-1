"""
This file contains examples for testing the hate speech detection model.
IMPORTANT: These examples are for educational and testing purposes only.
Some examples contain offensive content to test the model's detection capabilities.
"""

import torch
from hate_speech_model import load_model_and_vocab, classify_text
from roberta_model import load_roberta_model, classify_text_roberta

# Test examples - a mix of hate speech and non-hate speech
test_examples = [
    # Non-hate speech examples
    "I love how diverse our community is becoming!",
    "The weather is beautiful today.",
    "This movie was terrible, I really hated the plot.",
    "I disagree with your political views but respect your right to have them.",
    "The team lost the game because they didn't play well.",
    
    # Potential hate speech examples (for testing detection)
    "All people from that country are criminals and should be banned.",
    "That religious group is destroying our country and should be removed.",
    "Women don't deserve equal pay because they're not as smart as men.",
    "People of that race are naturally violent and dangerous.",
    "That ethnic group is taking all our jobs and ruining the economy."
]

def test_both_models():
    """Test both LSTM-CNN and RoBERTa models on the same examples"""
    print("Loading LSTM-CNN model...")
    lstm_model, vocab = load_model_and_vocab('hate_speech_model_lstm_cnn.pth', 'vocab.pkl')
    
    print("Loading RoBERTa model...")
    roberta_model, tokenizer = load_roberta_model('hate_speech_model_roberta.pth')
    
    print("\n" + "="*80)
    print("TESTING HATE SPEECH DETECTION MODELS".center(80))
    print("="*80 + "\n")
    
    for i, example in enumerate(test_examples):
        print(f"Example {i+1}: \"{example}\"")
        
        # LSTM-CNN prediction
        lstm_result = classify_text(lstm_model, vocab, example)
        lstm_pred = lstm_result['prediction']
        lstm_conf = lstm_result['confidence']
        
        # RoBERTa prediction
        roberta_result = classify_text_roberta(roberta_model, tokenizer, example)
        roberta_pred = roberta_result['prediction']
        roberta_conf = roberta_result['confidence']
        
        # Print results
        print(f"LSTM-CNN: {lstm_pred} (Confidence: {lstm_conf:.2f})")
        print(f"RoBERTa: {roberta_pred} (Confidence: {roberta_conf:.2f})")
        
        # Check if models agree
        if lstm_pred == roberta_pred:
            print("✓ Models agree")
        else:
            print("✗ Models disagree")
        
        print("-"*80)

if __name__ == "__main__":
    test_both_models()

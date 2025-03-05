import pandas as pd
from hate_speech_model import load_model, predict_hate_speech, TextPreprocessor

def test_sample(model_path, data_path, num_samples=5):
    """
    Test the model on a few samples from the dataset
    
    Args:
        model_path (str): Path to the saved model
        data_path (str): Path to the dataset
        num_samples (int): Number of samples to test
    """
    # Load model
    print(f"Loading model from {model_path}")
    model, vocab = load_model(model_path)
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Determine column names
    text_col = 'text' if 'text' in df.columns else 'tweet'
    label_col = 'label' if 'label' in df.columns else 'class'
    
    print(f"Using '{text_col}' as text column and '{label_col}' as label column")
    
    # Map class names
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    
    # Test on samples
    print(f"\nTesting on {num_samples} samples:")
    for i in range(min(num_samples, len(df))):
        text = df.iloc[i][text_col]
        true_label = df.iloc[i][label_col]
        
        # Preprocess text
        preprocessed_text = TextPreprocessor.preprocess_text(text)
        
        # Make prediction
        prediction, confidence = predict_hate_speech(preprocessed_text, model, vocab)
        
        print(f"\nSample {i+1}:")
        print(f"Text: {text}")
        print(f"True label: {true_label} ({class_names.get(true_label, 'Unknown')})")
        print(f"Predicted: {prediction} ({class_names.get(prediction, 'Unknown')})")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    model_path = "hate_speech_model.pt"
    data_path = "../../Data/Dataset_2.csv"
    
    test_sample(model_path, data_path, num_samples=5)

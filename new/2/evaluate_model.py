import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from hate_speech_model import load_model, predict_hate_speech, TextPreprocessor

def evaluate_model(model_path, test_data_path):
    """
    Evaluate the model on test data and print performance metrics.
    
    Args:
        model_path (str): Path to the saved model
        test_data_path (str): Path to the test data CSV file
    """
    print(f"Loading model from {model_path}")
    model, vocab = load_model(model_path)
    
    print(f"Loading test data from {test_data_path}")
    try:
        test_data = pd.read_csv(test_data_path)
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return
    
    # Check if required columns exist
    if 'text' not in test_data.columns and 'tweet' not in test_data.columns:
        print("Test data must contain 'text' or 'tweet' column")
        return
    
    if 'label' not in test_data.columns and 'class' not in test_data.columns:
        print("Test data must contain 'label' or 'class' column")
        return
    
    # Map column names if needed
    text_col = 'text' if 'text' in test_data.columns else 'tweet'
    label_col = 'label' if 'label' in test_data.columns else 'class'
    
    print(f"Using '{text_col}' as text column and '{label_col}' as label column")
    print(f"Test data loaded: {len(test_data)} samples")
    
    # Make predictions
    predictions = []
    confidences = []
    true_labels = []
    
    for idx, row in test_data.iterrows():
        try:
            text = row[text_col]
            label = row[label_col]
            
            # Preprocess text
            preprocessed_text = TextPreprocessor.preprocess_text(text)
            
            # Predict
            prediction, confidence = predict_hate_speech(preprocessed_text, model, vocab)
            
            predictions.append(prediction)
            confidences.append(confidence)
            true_labels.append(label)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(test_data)} samples")
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    # Print metrics
    print("\n--- Evaluation Results ---")
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    print("\nClassification Report:")
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    target_names = [class_names.get(i, f"Class {i}") for i in sorted(set(true_labels) | set(predictions))]
    print(classification_report(true_labels, predictions, target_names=target_names))
    
    print("\nAverage Confidence Score:", np.mean(confidences))
    
    # Print some example predictions
    print("\n--- Example Predictions ---")
    for i in range(min(10, len(test_data))):
        text = test_data.iloc[i][text_col]
        true_label = test_data.iloc[i][label_col]
        pred_label = predictions[i]
        conf = confidences[i]
        
        print(f"\nText: {text}")
        print(f"True Label: {class_names.get(true_label, f'Class {true_label}')}")
        print(f"Predicted: {class_names.get(pred_label, f'Class {pred_label}')} (Confidence: {conf:.4f})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate hate speech detection model")
    parser.add_argument("--model", default="hate_speech_model.pt", help="Path to the model file")
    parser.add_argument("--data", default="../../Data/Dataset_2.csv", help="Path to the test data CSV file")
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data)

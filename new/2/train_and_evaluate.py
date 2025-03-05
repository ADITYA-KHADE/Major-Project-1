import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from hate_speech_model import (
    TextPreprocessor, augment_and_prepare_data, 
    LSTM_CNN, train_model, save_model, load_model, predict_hate_speech
)

def train_hate_speech_model(data_path, model_save_path):
    """
    Train the hate speech detection model
    
    Args:
        data_path (str): Path to the dataset
        model_save_path (str): Path to save the model
    
    Returns:
        tuple: Trained model and vocabulary
    """
    print(f"Loading data from {data_path}...")
    try:
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} rows")
        
        # Check columns
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Determine text and label columns
        text_col = 'text' if 'text' in df.columns else 'tweet'
        label_col = 'label' if 'label' in df.columns else 'class'
        
        print(f"Using '{text_col}' as text column and '{label_col}' as label column")
        
        # Clean text
        df['clean_text'] = df[text_col].apply(TextPreprocessor.preprocess_text)
        
        # Ensure label column is numeric
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)
        
        # Check class distribution
        class_counts = df[label_col].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # For small datasets, use all available data
        if len(df) < 100:
            print("Small dataset detected. Using all available data instead of balancing.")
            df_balanced = df.copy()
        else:
            # Balance classes
            min_class_count = class_counts.min()
            print(f"Balancing classes to {min_class_count} samples each")
            
            balanced_dfs = []
            for class_val in class_counts.index:
                class_df = df[df[label_col] == class_val]
                if len(class_df) > min_class_count:
                    class_df = class_df.sample(min_class_count, random_state=42)
                balanced_dfs.append(class_df)
            
            df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
        
        print(f"Final dataset size: {len(df_balanced)} rows")
        
        # Augment and prepare data
        print("Augmenting and preparing data...")
        train_loader, val_loader, vocab = augment_and_prepare_data(
            df_balanced, 
            text_column='clean_text', 
            label_column=label_col,
            aug_count=10,  # Increase augmentation for small datasets
            test_size=0.2
        )
        
        # Initialize model
        print("Initializing model...")
        vocab_size = len(vocab) + 2  # Add padding and unknown tokens
        embedding_dim = 100
        hidden_dim_lstm = 64  # Reduced for small datasets
        hidden_dim_cnn = 64   # Reduced for small datasets
        output_dim = len(class_counts)  # Number of classes
        dropout = 0.3  # Reduced dropout for small datasets
        
        model = LSTM_CNN(
            vocab_size, 
            embedding_dim, 
            hidden_dim_lstm, 
            hidden_dim_cnn, 
            output_dim, 
            dropout
        )
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        print("Training model...")
        trained_model = train_model(
            model, 
            criterion, 
            optimizer, 
            train_loader, 
            val_loader, 
            num_epochs=10  # Increased for better learning
        )
        
        # Save model
        print(f"Saving model to {model_save_path}...")
        save_model(trained_model, vocab, model_save_path)
        
        print("Training completed successfully!")
        return trained_model, vocab
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_model(model, vocab, test_data_path):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        vocab: Vocabulary
        test_data_path (str): Path to the test data
    """
    print(f"Evaluating model on {test_data_path}...")
    try:
        # Load test data
        test_data = pd.read_csv(test_data_path)
        
        # Determine text and label columns
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
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Paths
    train_data_path = "../../Data/Dataset_2.csv"
    model_save_path = "hate_speech_model.pt"
    
    # Train model
    print("=== Training Model ===")
    model, vocab = train_hate_speech_model(train_data_path, model_save_path)
    
    if model is not None and vocab is not None:
        # Evaluate on the same dataset
        print("\n=== Evaluating Model ===")
        evaluate_model(model, vocab, train_data_path)
    else:
        print("Training failed. Loading saved model for evaluation...")
        try:
            model, vocab = load_model(model_save_path)
            evaluate_model(model, vocab, train_data_path)
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()

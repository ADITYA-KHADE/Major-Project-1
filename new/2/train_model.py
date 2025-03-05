import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from hate_speech_model import (
    TextPreprocessor, augment_and_prepare_data, 
    LSTM_CNN, train_model, save_model
)

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    try:
        # Dataset path
        df = pd.read_csv('../../Data/Dataset_2.csv')
        print(f"Loaded dataset with {len(df)} rows")
        
        # Check columns
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Clean text from the 'tweet' column
        df['clean_tweet'] = df['tweet'].apply(TextPreprocessor.preprocess_text)
        
        # Use 'class' as the label column
        label_col = 'class'
        
        # Check class distribution
        class_counts = df[label_col].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        # Ensure label column is numeric
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)
        
        # For small datasets, we'll use a different approach
        # Instead of balancing, let's use all available data
        if len(df) < 100:
            print("Small dataset detected. Using all available data instead of balancing.")
            df_balanced = df.copy()
        else:
            # Balance classes if needed
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
        
        # Check if we have enough data to proceed
        if len(df_balanced) < 10:
            print("Warning: Very small dataset. Training may not be effective.")
            
        # Augment and prepare data
        print("Augmenting and preparing data...")
        train_loader, val_loader, vocab = augment_and_prepare_data(
            df_balanced, 
            text_column='clean_tweet', 
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
        output_dim = len(class_counts)  # Number of classes (0, 1, 2)
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
        print("Saving model...")
        save_model(trained_model, vocab, 'hate_speech_model.pt')
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from roberta_model import RoBERTaClassifier, preprocess_for_roberta

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = preprocess_for_roberta(text)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_roberta_model(train_data, val_data, batch_size=16, epochs=5):
    """Train a RoBERTa model for hate speech detection"""
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RoBERTaClassifier(pretrained_model='roberta-base')
    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = HateSpeechDataset(
        texts=train_data['text'].values,
        labels=train_data['label'].values,
        tokenizer=tokenizer
    )
    
    val_dataset = HateSpeechDataset(
        texts=val_data['text'].values,
        labels=val_data['label'].values,
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Define class weights for imbalanced dataset
    class_weights = torch.tensor([0.3, 0.7]).to(device)  # Adjust based on your dataset
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"\n{'=' * 30}\nEpoch {epoch+1}/{epochs}\n{'=' * 30}")
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            train_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                val_losses.append(loss.item())
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute validation metrics
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Compute F1 score for hate speech class (class 1)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, pos_label=1)
        
        print(f"Validation Loss: {avg_val_loss:.4f}, F1 (hate speech): {val_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Non-Hate', 'Hate']))
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print("New best model found! Saving...")
            
            model_info = {
                'state_dict': model.state_dict(),
                'val_f1': val_f1,
                'epoch': epoch
            }
            
            torch.save(model_info, 'hate_speech_model_roberta.pth')
    
    print("\nTraining complete!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    return model, tokenizer

if __name__ == "__main__":
    # Load your data here
    # This is a placeholder - replace with your actual data loading code
    try:
        # Change this line to use Dataset_1.csv instead of Dataset_2.csv
        data_path = '../../Data/Dataset_1.csv'
        df = pd.read_csv(data_path, encoding='latin1')
        
        # Map classes to binary labels if needed
        # Note: You may need to adjust this based on Dataset_1.csv's structure
        if 'class' in df.columns:
            df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)  # Adjust based on your dataset
        
        # If Dataset_1.csv has different column names than Dataset_2.csv,
        # you'll need to modify this code to match the column structure
        if 'tweet' in df.columns:
            df = df[['tweet', 'label']].rename(columns={'tweet': 'text'})
        elif 'text' in df.columns:
            df = df[['text', 'label']]
        else:
            # Try to identify the text column
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()]
            if text_cols:
                df = df[[text_cols[0], 'label']].rename(columns={text_cols[0]: 'text'})
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        print(f"Training data: {len(train_df)} samples")
        print(f"Validation data: {len(val_df)} samples")
        
        # Balance training data if needed
        from sklearn.utils import resample
        
        # Separate majority and minority classes
        train_majority = train_df[train_df.label == 0]
        train_minority = train_df[train_df.label == 1]
        
        # Upsample minority class
        train_minority_upsampled = resample(
            train_minority, 
            replace=True,
            n_samples=len(train_majority),
            random_state=42
        )
        
        # Combine upsampled minority class with majority class
        train_df_balanced = pd.concat([train_majority, train_minority_upsampled])
        
        print(f"Balanced training data: {len(train_df_balanced)} samples")
        print(f"Class distribution: {train_df_balanced.label.value_counts().to_dict()}")
        
        # Train the model
        model, tokenizer = train_roberta_model(
            train_df_balanced, 
            val_df, 
            batch_size=16, 
            epochs=5
        )
        
        print("Model saved successfully to hate_speech_model_roberta.pth")
        
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()

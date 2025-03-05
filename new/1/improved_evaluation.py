from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_comprehensive(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics with focus on hate speech class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )
    
    print(f"Non-hate speech - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    print(f"Hate speech - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-hate', 'Hate'], 
                yticklabels=['Non-hate', 'Hate'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def cross_validation_evaluation(model_class, X, y, n_splits=5):
    """Perform cross-validation to get more reliable performance estimates"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize and train model
        model = model_class()
        train_model(model, X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model_comprehensive(model, X_val, y_val)
        fold_metrics.append(metrics)
        print(f"Fold {fold+1} - Hate Speech F1: {metrics['f1'][1]:.4f}")
    
    # Average metrics across folds
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in fold_metrics], axis=0),
        'recall': np.mean([m['recall'] for m in fold_metrics], axis=0),
        'f1': np.mean([m['f1'] for m in fold_metrics], axis=0)
    }
    
    print(f"Average Hate Speech F1 across folds: {avg_metrics['f1'][1]:.4f}")
    return avg_metrics

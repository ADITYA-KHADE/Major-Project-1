import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_and_prepare_data(filepaths):
    """Load and combine multiple datasets for more robust training"""
    dataframes = []
    
    for path in filepaths:
        df = pd.read_csv(path)
        # Map dataset-specific labels to standardized format
        # Adjust column mapping based on each dataset's structure
        dataframes.append(standardize_format(df))
    
    # Combine all datasets
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Apply SMOTE for better class balance
    X = combined_df['clean_text']
    y = combined_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE only to training data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        X_train.to_frame(), y_train
    )
    
    return X_resampled, X_test, y_resampled, y_test

def standardize_format(df):
    """Standardize different dataset formats to common schema"""
    # Implementation depends on specific datasets used
    # ...
    return standardized_df

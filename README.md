# Hate Speech Detection in Social Media

A machine learning system for detecting and classifying hate speech in social media content using state-of-the-art NLP techniques.

![Hate Speech Detection](https://img.shields.io/badge/NLP-Hate%20Speech%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.6%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-red)

## Overview

This project implements a comprehensive hate speech detection system that analyzes text content from social media platforms to identify and classify hate speech. The system employs multiple advanced deep learning models, including hybrid LSTM-CNN architectures and transformer-based models like RoBERTa, with an interactive web interface for real-time classification.

## Features

- **Multiple Model Implementation**:
  - LSTM+CNN Hybrid Architecture
  - RoBERTa Transformer Model
  - Model ensemble capabilities

- **Advanced Techniques**:
  - Adversarial Training for robustness
  - Data Augmentation to address class imbalance
  - Explainable AI components for model interpretation

- **User Interface**:
  - Interactive web application for real-time hate speech detection
  - Option to switch between different models
  - Confidence scores for predictions

- **Comprehensive Evaluation**:
  - Precision/recall metrics focused on hate speech identification
  - Confusion matrices for error analysis
  - Cross-validation for robust performance assessment

## Model Architecture

### LSTM+CNN Hybrid

This model combines the strengths of both LSTM and CNN architectures:
1. Word embeddings are fed into a bidirectional LSTM to capture contextual information
2. LSTM outputs are processed through a CNN layer to extract higher-level features
3. Adaptive pooling and fully connected layers produce the final classification

### RoBERTa Model

Our implementation leverages the RoBERTa transformer architecture:
1. Pre-trained on a large corpus of text data
2. Fine-tuned on hate speech datasets
3. Enhanced contextual understanding for better detection of implicit hate speech

## Unified Training Approach

Both models (LSTM-CNN and RoBERTa) are trained on Dataset_1.csv to ensure fair comparison and consistent performance. Our unified training script (`train_models.py`) handles:

- Consistent data loading and preprocessing for both models
- Identical train/validation splits (80/20)
- Equal data balancing techniques to address class imbalance
- Standardized evaluation metrics for fair comparison

This approach ensures that performance differences between models are due to architectural differences rather than data discrepancies.

## Installation & Setup

### Prerequisites

- Python 3.6+
- PyTorch 1.9+
- Flask
- Transformers library
- Other dependencies in `requirements.txt`

### Setup Steps

1. **Install dependencies**:
   ```
   pip install torch transformers flask flask_cors pandas numpy scikit-learn matplotlib seaborn tqdm nltk
   ```

2. **Generate model files**:
   ```
   python save_model_demo.py
   ```

3. **Download pre-trained RoBERTa model** (optional - will download automatically when needed):
   ```
   python -c "from transformers import RobertaTokenizer, RobertaModel; tokenizer = RobertaTokenizer.from_pretrained('roberta-base'); model = RobertaModel.from_pretrained('roberta-base')"
   ```

## Usage

### Web Interface

1. **Start the Flask server**:
   ```
   python app.py
   ```

2. **Access the web interface** at `http://localhost:5000`

3. **Enter text** and select the model you want to use for classification

## Training Your Own Models

To train both models on Dataset_1.csv:

1. **Run the unified training script**:
   ```
   python train_models.py
   ```
   This trains both LSTM-CNN and RoBERTa models on Dataset_1.csv with identical preprocessing and data splits.

2. **Monitor training progress** to see performance comparisons across models.

## Project Structure

- `app.py` - Flask web application
- `hate_speech_model.py` - Core LSTM-CNN model implementation
- `roberta_model.py` - RoBERTa model implementation
- `improved_models.py` - Enhanced model architectures
- `train_models.py` - Unified script to train both models on the same dataset
- `save_model_demo.py` - Quick script to generate model files
- `model_explainer.py` - Model interpretability tools

## Performance

Our models achieve the following performance metrics on test data from Dataset_1.csv:

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| LSTM-CNN | 0.92 | 0.88 | 0.90 |
| RoBERTa | 0.95 | 0.91 | 0.93 |

## Dataset Information

Both models are trained on the same dataset:
- Dataset: `Dataset_1.csv` - Twitter hate speech dataset
- Location: `../../Data/Dataset_1.csv`
- Preprocessing: Identical for both models
- Split: 80% training / 20% validation

## Future Work

- Implement multimodal analysis (text + image)
- Add support for more languages
- Develop more sophisticated context-aware classification
- Create a browser extension for real-time hate speech detection

# Hate Speech Detection Model

This project implements a robust hate speech detection model using a hybrid LSTM+CNN architecture with adversarial training techniques.

## Project Structure

- `hate_speech_model.py`: Core model implementation and utility functions
- `train_model.py`: Script to train and save the model
- `app.py`: Flask web application for the frontend
- `templates/index.html`: HTML template for the web interface
- `train_and_evaluate.py`: Comprehensive script for training and evaluation
- `evaluate_model.py`: Script to evaluate model performance
- `test_api.py`: Script to test the Flask API
- `test_sample.py`: Script to test individual samples

## Setup and Installation

1. Install the required dependencies:

```bash
pip install torch pandas numpy nltk nlpaug scikit-learn flask transformers
```

2. Download NLTK resources:

```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

## Dataset

The model is trained on Dataset_2.csv with the following characteristics:
- Total samples: 100
- Class distribution:
  - Class 1 (Offensive Language): 91 samples
  - Class 2 (Neither): 7 samples
  - Class 0 (Hate Speech): 2 samples

Due to this class imbalance, the training process balances the classes by downsampling to the minimum class size and then applies data augmentation.

## Training the Model

1. Prepare your dataset:
   - The dataset should be a CSV file with at least two columns: one for the text (tweet) and one for the label (class)
   - The model supports the following class labels:
     - 0: Hate Speech
     - 1: Offensive Language
     - 2: Neither

2. Run the training script:

```bash
python train_model.py
```

Or use the comprehensive training and evaluation script:

```bash
python train_and_evaluate.py
```

This will:
- Load and preprocess your dataset
- Augment the data to improve model robustness
- Train the LSTM+CNN model
- Save the trained model to `hate_speech_model.pt`
- Evaluate the model performance

## Evaluation Results

When trained on Dataset_2.csv, the model achieved:
- Overall accuracy: 93%
- Precision for Hate Speech (Class 0): 100%
- Precision for Offensive Language (Class 1): 95%
- Precision for Neither (Class 2): 50%
- Average confidence score: ~41%

Confusion Matrix:
```
[[ 2  0  0]
 [ 0 89  2]
 [ 0  5  2]]
```

The model performs well on the majority class (Offensive Language) and the minority class (Hate Speech), but struggles with the "Neither" class due to class imbalance.

## Running the Web Application

After training the model, you can run the web application:

```bash
python app.py
```

This will start a Flask server at `http://127.0.0.1:5000/`. Open this URL in your browser to access the hate speech detection interface.

## Usage

1. Enter text in the provided text area
2. Click "Analyze Text"
3. The application will display:
   - The predicted class (Hate Speech, Offensive Language, or Neither)
   - The confidence score of the prediction
   - A visual indicator of the confidence level

## Model Architecture

The model uses a hybrid architecture:
- Embedding layer to convert tokens to vectors
- Bidirectional LSTM to capture contextual information
- 1D Convolutional layer to extract local features
- Fully connected layer for classification

## Data Augmentation

To handle small datasets and class imbalance, the model uses:
- Synonym replacement
- Random insertion
- Random swap
- Random deletion
- Back translation (when available)

## Troubleshooting

If you encounter the KeyError 365 issue:
- This is fixed in the current implementation by properly converting pandas Series to tensors in the dataset class
- The error occurred because of indexing issues when accessing labels in the dataset

## Future Improvements

- Implement more advanced data augmentation techniques
- Add model interpretability features
- Enhance the frontend with visualization of model confidence
- Add support for multiple languages
- Collect more training data, especially for underrepresented classes
- Implement transfer learning with pre-trained language models

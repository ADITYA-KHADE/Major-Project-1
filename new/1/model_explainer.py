import lime
import lime.lime_text
import shap
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class HateSpeechExplainer:
    def __init__(self, model, vocab, preprocess_fn):
        self.model = model
        self.vocab = vocab
        self.preprocess_fn = preprocess_fn
        self.lime_explainer = lime.lime_text.LimeTextExplainer(class_names=['Non-Hate', 'Hate'])
        
    def predict_proba(self, texts):
        """Wrapper for model prediction to work with LIME"""
        processed = [self.preprocess_fn(text) for text in texts]
        # Convert to model input format and get predictions
        # ...
        return probabilities  # Shape (n_samples, n_classes)
        
    def explain_prediction(self, text, num_features=10):
        """Generate explanation for a single prediction"""
        exp = self.lime_explainer.explain_instance(
            text, 
            self.predict_proba, 
            num_features=num_features
        )
        
        # Get prediction
        probabilities = self.predict_proba([text])[0]
        predicted_class = np.argmax(probabilities)
        
        # Display explanation
        print(f"Text: {text}")
        print(f"Prediction: {'Hate Speech' if predicted_class == 1 else 'Non-Hate Speech'}")
        print(f"Confidence: {probabilities[predicted_class]:.4f}")
        print("\nExplanation:")
        
        # Get features with highest impact
        features = exp.as_list()
        for feature, impact in features:
            print(f"- '{feature}': {'Increases' if impact > 0 else 'Decreases'} hate score by {abs(impact):.4f}")
        
        # Create visualization
        fig = plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title("Features contributing to hate speech classification")
        plt.tight_layout()
        return exp

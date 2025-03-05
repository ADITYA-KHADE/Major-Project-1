import requests
import json

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"

# Test texts
test_texts = [
    "I love this product, it's amazing!",
    "This is the worst thing I've ever seen.",
    "I hate people who don't respect others.",
    "You are stupid and worthless.",
    "The weather is nice today."
]

# Make predictions for each test text
for text in test_texts:
    payload = {"text": text}
    
    try:
        # Send POST request
        response = requests.post(url, json=payload)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            if 'error' in result:
                print(f"\nError for text '{text}': {result['error']}")
            else:
                print(f"\nText: {text}")
                print(f"Prediction Class: {result.get('class', 'Unknown')}")
                print(f"Confidence: {result.get('confidence', 0):.4f} ({int(result.get('confidence', 0) * 100)}%)")
        else:
            print(f"\nError for text '{text}': {response.text}")
    except Exception as e:
        print(f"\nException for text '{text}': {str(e)}")

print("\nTesting completed.")

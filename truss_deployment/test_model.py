"""
Local test script for the Truss model.
This simulates what Truss does when serving the model.
"""

import sys
sys.path.insert(0, '.')

from model.model import Model

def main():
    print("Loading model...")
    model = Model()
    model.load()
    print("Model loaded successfully!\n")

    # Test cases
    test_inputs = [
        {"text": "I love this product! It's amazing!"},
        {"text": "This is terrible and I hate it."},
        {"text": "The weather is okay today."},
        {"text": "Best purchase I've ever made!"},
        {"text": "Worst experience of my life."},
    ]

    print("Running predictions:\n")
    print("-" * 60)

    for test_input in test_inputs:
        result = model.predict(test_input)
        label = result["prediction"]["label"]
        score = result["prediction"]["score"]

        print(f"Input: \"{test_input['text']}\"")
        print(f"Result: {label} (confidence: {score:.2%})")
        print("-" * 60)

if __name__ == "__main__":
    main()

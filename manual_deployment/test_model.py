"""
Test script for manual deployment model.
Tests that the model loads and works correctly.
"""

from transformers import pipeline
import torch
import time

def main():
    print("Loading model (manual deployment style)...")
    start_time = time.time()

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",
        device=device
    )

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds\n")

    # Test cases
    test_inputs = [
        "I love this product! It's amazing!",
        "This is terrible and I hate it.",
        "The weather is okay today.",
        "Best purchase I've ever made!",
        "Worst experience of my life.",
    ]

    print("Running predictions:\n")
    print("-" * 60)

    for text in test_inputs:
        result = model(text)
        label = result[0]["label"]
        score = result[0]["score"]

        print(f"Input: \"{text}\"")
        print(f"Result: {label} (confidence: {score:.2%})")
        print("-" * 60)

if __name__ == "__main__":
    main()

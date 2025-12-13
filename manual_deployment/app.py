"""
Manual ML Model Deployment - Flask API
This demonstrates the traditional way of deploying an ML model as an API.
Requires: Flask setup, manual request handling, model loading logic, etc.
"""

from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to hold the model
model = None


def load_model():
    """
    Load the sentiment analysis model.
    This needs to be called at startup and handled carefully.
    """
    global model
    logger.info("Loading sentiment analysis model...")
    start_time = time.time()

    # Load HuggingFace sentiment analysis pipeline
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",  # Force PyTorch
        device=device
    )

    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    return model


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint - need to implement manually.
    """
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint - lots of boilerplate code needed:
    - Request parsing
    - Input validation
    - Error handling
    - Response formatting
    """
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    # Parse request
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    text = data["text"]

    # Validate text type
    if not isinstance(text, str):
        return jsonify({"error": "'text' must be a string"}), 400

    if len(text.strip()) == 0:
        return jsonify({"error": "'text' cannot be empty"}), 400

    # Run inference
    try:
        start_time = time.time()
        result = model(text)
        inference_time = time.time() - start_time

        # Format response
        response = {
            "input": text,
            "prediction": result[0],
            "inference_time_seconds": round(inference_time, 4)
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with API info."""
    return jsonify({
        "name": "Sentiment Analysis API (Manual Deployment)",
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/predict": "POST - Sentiment prediction"
        },
        "example_request": {
            "method": "POST",
            "url": "/predict",
            "body": {"text": "I love this product!"}
        }
    })


# Application startup
if __name__ == "__main__":
    # Load model at startup
    load_model()

    # Run Flask app
    # In production, you'd use gunicorn/uvicorn, configure workers, etc.
    app.run(host="0.0.0.0", port=8080, debug=False)

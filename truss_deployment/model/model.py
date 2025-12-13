"""
Truss Model Deployment
Simple sentiment analysis model using HuggingFace transformers.
"""

from transformers import pipeline


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        """Load the model once at startup."""
        self._model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def predict(self, model_input):
        """Run inference on input text."""
        return self._model(model_input)

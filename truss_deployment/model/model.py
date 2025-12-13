"""
Truss Model Deployment
This demonstrates how simple deployment becomes with Truss.
Just implement load() and predict() - that's it!
"""

from transformers import pipeline
import torch


class Model:
    """
    Truss Model class - only needs two methods:
    - load(): Initialize/load your model
    - predict(): Run inference

    No Flask, no Dockerfile, no boilerplate!
    """

    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        """
        Load the model. Called once at startup.
        Truss handles all the infrastructure around this.
        """
        device = 0 if torch.cuda.is_available() else -1
        self._model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",  # Force PyTorch
            device=device
        )

    def predict(self, model_input):
        """
        Run inference. Truss handles:
        - HTTP endpoint creation
        - Request parsing
        - Response formatting
        - Error handling
        """
        text = model_input.get("text", "")
        result = self._model(text)

        return {
            "input": text,
            "prediction": result[0]
        }

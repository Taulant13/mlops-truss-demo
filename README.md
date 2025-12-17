# MLOps Project 3: Truss by Baseten

A demonstration project comparing traditional ML model deployment vs deployment using **Truss**, an open-source model serving framework by Baseten.

## What is Truss?

[Truss](https://github.com/basetenlabs/truss) is an open-source tool that simplifies packaging and deploying ML models to production. It eliminates the need for boilerplate code, Dockerfiles, and complex infrastructure setup.

**Key Features:**
- Minimal code - just `load()` and `predict()` methods
- No Docker knowledge required
- Automatic dependency management
- Live reload during development
- Framework agnostic (PyTorch, TensorFlow, HuggingFace, etc.)

## Project Structure

```
.
├── manual_deployment/          # Traditional Flask-based deployment
│   ├── app.py                  # Flask API (~100 lines)
│   ├── requirements.txt        # Dependencies
│   ├── Dockerfile              # Container configuration
│   └── test_model.py           # Test script
│
├── truss_deployment/           # Truss-based deployment
│   ├── config.yaml             # Configuration (~5 lines)
│   ├── model/
│   │   └── model.py            # Model class (~30 lines)
│   └── test_model.py           # Test script
│
└── README.md
```

## The Use Case

Both deployments serve the same ML model: a HuggingFace sentiment analysis model (`distilbert-base-uncased-finetuned-sst-2-english`) that classifies text as POSITIVE or NEGATIVE.

## Installation

### Prerequisites
- Python 3.10+
- Docker (optional, for manual deployment)

### Setup
```bash
# Clone the repository
git clone https://github.com/Taulant13/mlops-truss-demo.git
cd mlops-truss-demo

# Install dependencies
pip install transformers torch flask truss
```

## Usage

### Option 1: Manual Deployment (Flask)

```bash
# Navigate to folder
cd manual_deployment

# Start the server
python app.py

# In another terminal, test the API (cmd)
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"text\": \"I love this product!\"}"
```

**Response:**
```json
{
  "input": "I love this product!",
  "prediction": {"label": "POSITIVE", "score": 0.9998},
  "inference_time_seconds": 0.03
}
```

### Option 2: Truss Deployment

```bash
# Navigate to folder
cd truss_deployment
```

**Step 1:** Get your [Baseten API Key](http://app.baseten.co/settings/account/api_keys)

**Step 2:** Deploy your model
```bash
truss push
```

**Step 3:** Invoke your model
```bash
truss predict -d '"I love this product!"'
```

**Output:**
```
Input: "I love this product! It's amazing!"
Result: POSITIVE (confidence: 99.99%)

Input: "This is terrible and I hate it."
Result: NEGATIVE (confidence: 99.95%)
```

## Comparison

| Aspect | Manual (Flask) | Truss |
|--------|----------------|-------|
| Lines of code | ~100+ | ~30 |
| Files needed | 4 (app.py, Dockerfile, requirements.txt, test_model.py) | 3 (model.py, config.yaml, test_model.py) |
| Docker knowledge | Required | Not needed |
| Live reload | No | Yes |
| Health checks | Manual implementation | Automatic |
| Dependency management | Manual | Declarative (YAML) |

## Pros and Cons of Truss

### Pros
- Minimal boilerplate code
- No Docker/containerization knowledge needed
- Automatic dependency management
- Built-in health checks and error handling
- Live reload for fast development
- Consistent dev/prod environments

### Cons
- Vendor-oriented (primarily designed for Baseten platform)
- Less flexibility for custom API behavior
- Smaller ecosystem compared to Flask/FastAPI

## Resources

- [Truss GitHub Repository](https://github.com/basetenlabs/truss)
- [Truss Documentation](https://truss.baseten.co/)
- [Baseten Platform](https://www.baseten.co/)


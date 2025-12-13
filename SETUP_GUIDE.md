# Step-by-Step Setup Guide

## Prerequisites

```bash
pip install transformers torch flask truss
```

---

## Option 1: Manual Deployment (Flask)

### Step 1: Navigate to folder
```bash
cd manual_deployment
```

### Step 2: Start the Flask server
```bash
python app.py
```
Wait for: `Model loaded successfully`

### Step 3: Test the API (in a new terminal)
```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"text\": \"I love this product!\"}"
```

### Step 4: Check health endpoint
```bash
curl http://localhost:8080/health
```

### Step 5: Stop server
Press `Ctrl+C` in the terminal running the server.

---

## Option 2: Truss Deployment

### Step 1: Navigate to folder
```bash
cd truss_deployment
```

### Step 2: Test the model directly
```bash
python test_model.py
```

### Step 3: (Optional) Run as server with hot reload
```bash
truss run
```

### Step 4: (Optional) Deploy to Baseten
```bash
truss push
```

---

## Quick Test (Without Server)

### Manual version
```bash
cd manual_deployment
python test_model.py
```

### Truss version
```bash
cd truss_deployment
python test_model.py
```

Both will output:
```
Input: "I love this product! It's amazing!"
Result: POSITIVE (confidence: 99.99%)

Input: "This is terrible and I hate it."
Result: NEGATIVE (confidence: 99.95%)
```

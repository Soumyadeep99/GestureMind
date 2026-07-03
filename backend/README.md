# SignSense AI — Backend

## Setup
```bash
cd "D:/Downloads/data_collection/backend"
pip install -r requirements.txt
```

## Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Test
Open browser: http://localhost:8000
Health check: http://localhost:8000/health

## Endpoints
- GET  /         — health check
- GET  /health   — detailed status
- POST /predict  — gesture prediction

## Request format (POST /predict)
```json
{
  "frames": [
    { "keypoints": [0.1, 0.2, ..., 0.9] },  // 1662 floats
    // ... 30 frames total
  ]
}
```

## Response
```json
{
  "gesture": "hello",
  "confidence": 0.9821,
  "is_confident": true,
  "all_probabilities": { "hello": 0.9821, "thank_you": 0.003, ... },
  "inference_time_ms": 12.4
}
```

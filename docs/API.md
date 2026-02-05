# API Documentation

This document describes the API endpoints available in the Object Detection Comparison application.

## Base URL

- **Local Development**: `http://localhost:80`
- **Production**: Your deployed domain

## Endpoints

### Health Check

Check if the application and TensorFlow Serving containers are healthy.

```http
GET /health
```

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "services": {
    "flask": "healthy",
    "tf_serving_a": "healthy",
    "tf_serving_b": "healthy"
  }
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | All services healthy |
| 503 | One or more services unavailable |

---

### Upload and Detect

Upload an image and run detection with both models.

```http
POST /detect
Content-Type: multipart/form-data
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Image file (PNG, JPG, JPEG, GIF, BMP, WebP) |

#### Response

Redirects to `/results?id={image_id}` on success.

#### Error Response

```json
{
  "error": "No file provided"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 302 | Redirect to results page |
| 400 | Invalid request (no file, invalid format) |
| 500 | Server error during processing |

---

### View Results

View detection results for a specific image.

```http
GET /results?id={image_id}
```

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | No | UUID of the processed image |

If `id` is not provided, displays recent results.

#### Response

HTML page with detection results, or JSON if `Accept: application/json` header is present:

```json
{
  "image_id": "abc123-def456",
  "original_filename": "test.jpg",
  "timestamp": "2024-01-15 10:30:00",
  "model_a": {
    "inference_time": 0.152,
    "detection_count": 5,
    "detections": [
      {
        "class_name": "person",
        "confidence": 0.95,
        "bbox": [100, 150, 200, 300]
      }
    ]
  },
  "model_b": {
    "inference_time": 0.445,
    "detection_count": 7,
    "detections": [...]
  }
}
```

---

### Get Recent Results

Retrieve a list of recent detection results.

```http
GET /api/recent
```

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 20 | Maximum number of results |
| `offset` | integer | 0 | Number of results to skip |

#### Response

```json
{
  "results": [
    {
      "image_id": "abc123-def456",
      "original_filename": "test.jpg",
      "timestamp": "2024-01-15 10:30:00",
      "model_a_detection_count": 5,
      "model_b_detection_count": 7
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

---

### Metrics Dashboard

View aggregated metrics and analytics.

```http
GET /metrics
```

#### Response

HTML page with metrics dashboard, or JSON if `Accept: application/json`:

```json
{
  "total_images": 150,
  "model_a": {
    "avg_time": 0.152,
    "avg_detections": 4.5,
    "total_detections": 675,
    "min_time": 0.098,
    "max_time": 0.312,
    "class_distribution": {
      "person": 245,
      "car": 180,
      "dog": 120
    }
  },
  "model_b": {
    "avg_time": 0.445,
    "avg_detections": 6.2,
    "total_detections": 930,
    "min_time": 0.312,
    "max_time": 0.678,
    "class_distribution": {
      "person": 320,
      "car": 250,
      "dog": 180
    }
  }
}
```

---

## Error Handling

All endpoints return errors in a consistent format:

```json
{
  "error": "Error message description",
  "request_id": "req-abc123",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 413 | Payload Too Large - File exceeds limit |
| 500 | Internal Server Error |
| 503 | Service Unavailable - TF Serving down |

---

## Rate Limiting

Currently, there is no rate limiting implemented. For production deployments, consider adding rate limiting via a reverse proxy (nginx, Traefik) or Flask extension.

---

## CORS

Cross-Origin Resource Sharing is enabled for all origins in development. Configure appropriately for production:

```python
CORS(app, origins=['https://yourdomain.com'])
```

---

## Examples

### cURL - Upload Image

```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost/detect
```

### cURL - Get Recent Results

```bash
curl -H "Accept: application/json" \
  http://localhost/api/recent?limit=10
```

### JavaScript - Fetch Results

```javascript
const response = await fetch('/api/recent?limit=10', {
  headers: { 'Accept': 'application/json' }
});
const data = await response.json();
console.log(data.results);
```

### Python - Upload Image

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost/detect',
        files={'file': f}
    )
print(response.url)  # Redirected results URL
```

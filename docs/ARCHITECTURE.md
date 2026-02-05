# System Architecture

This document describes the architecture and design decisions of the Object Detection Comparison application.

## Overview

The application follows a microservices architecture with three main components:

1. **Flask Web Application** - User interface and API
2. **TensorFlow Serving (Model A)** - SSD model inference
3. **TensorFlow Serving (Model B)** - Faster R-CNN model inference

## Architecture Diagram

```
                                    ┌─────────────────────────────────┐
                                    │         User Browser            │
                                    │    (HTML/CSS/JavaScript)        │
                                    └───────────────┬─────────────────┘
                                                    │
                                                    │ HTTP/HTTPS
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              Docker Network                                    │
│                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        Flask Application (Port 80)                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │   Routes     │  │  Detection   │  │    Image     │  │   Static   │  │  │
│  │  │  /detect     │  │   Service    │  │  Processing  │  │   Files    │  │  │
│  │  │  /results    │  │              │  │   (PIL)      │  │            │  │  │
│  │  │  /metrics    │  │              │  │              │  │            │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                           │                    │                              │
│              ┌────────────┴────────────────────┴────────────┐                │
│              │                                               │                │
│              ▼                                               ▼                │
│  ┌────────────────────────┐                 ┌────────────────────────┐       │
│  │  TensorFlow Serving A  │                 │  TensorFlow Serving B  │       │
│  │      (Port 8501)       │                 │      (Port 8502)       │       │
│  │                        │                 │                        │       │
│  │  ┌──────────────────┐  │                 │  ┌──────────────────┐  │       │
│  │  │   SSD ResNet50   │  │                 │  │  Faster R-CNN    │  │       │
│  │  │   FPN 640x640    │  │                 │  │  ResNet101       │  │       │
│  │  └──────────────────┘  │                 │  └──────────────────┘  │       │
│  └────────────────────────┘                 └────────────────────────┘       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        Persistent Storage      │
                    │  ┌─────────────────────────┐  │
                    │  │  detection_results.csv  │  │
                    │  │  /static/originals/     │  │
                    │  │  /results_details/      │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

## Components

### 1. Flask Web Application

**Purpose**: Serves the web interface, handles file uploads, orchestrates model inference, and manages results.

**Key Responsibilities**:
- Serve HTML templates with Jinja2
- Handle multipart file uploads
- Preprocess images for model inference
- Send requests to TensorFlow Serving
- Post-process detections and draw bounding boxes
- Store results in CSV and JSON formats
- Serve static files (CSS, JS, images)

**Technologies**:
- Flask 3.0+
- Pillow (PIL) for image processing
- NumPy for array operations
- Requests for HTTP client
- Flask-CORS for cross-origin support

### 2. TensorFlow Serving (Model A - SSD)

**Purpose**: Serve SSD ResNet50 FPN model for fast object detection.

**Model Details**:
- Architecture: SSD (Single Shot MultiBox Detector)
- Backbone: ResNet50 with FPN
- Input Size: 640x640
- Speed: ~150ms inference time
- Trade-off: Faster but potentially fewer detections

**Configuration**:
```protobuf
config {
  name: 'ssd'
  base_path: '/models/ssd/'
  model_platform: 'tensorflow'
}
```

### 3. TensorFlow Serving (Model B - Faster R-CNN)

**Purpose**: Serve Faster R-CNN ResNet101 model for accurate object detection.

**Model Details**:
- Architecture: Faster R-CNN (Region-based CNN)
- Backbone: ResNet101
- Input Size: 640x640
- Speed: ~450ms inference time
- Trade-off: More accurate but slower

**Configuration**:
```protobuf
config {
  name: 'faster_rcnn'
  base_path: '/models/faster_rcnn/'
  model_platform: 'tensorflow'
}
```

## Data Flow

### Image Upload and Detection

```
1. User uploads image via web interface
                    │
                    ▼
2. Flask receives multipart form data
                    │
                    ▼
3. Image validation (format, size)
                    │
                    ▼
4. Save original image to disk
                    │
                    ▼
5. Preprocess image (resize, normalize)
                    │
                    ├───────────────────┐
                    ▼                   ▼
6a. Send to TF Serving A    6b. Send to TF Serving B
    (SSD model)                 (Faster R-CNN)
                    │                   │
                    ▼                   ▼
7a. Receive predictions     7b. Receive predictions
                    │                   │
                    └───────┬───────────┘
                            ▼
8. Post-process detections (filter by confidence)
                            │
                            ▼
9. Draw bounding boxes on images
                            │
                            ▼
10. Save results (CSV, JSON, images)
                            │
                            ▼
11. Redirect user to results page
```

### Results Display

```
1. User requests /results?id=<uuid>
                    │
                    ▼
2. Flask loads data from CSV and JSON
                    │
                    ▼
3. Render results.html template with data
                    │
                    ▼
4. Browser displays side-by-side comparison
                    │
                    ▼
5. Chart.js renders performance visualizations
```

## Storage Strategy

### Why CSV Instead of Database?

For this portfolio project, CSV storage was chosen for:

1. **Simplicity** - No database setup required
2. **Portability** - Easy to export and analyze
3. **Transparency** - Human-readable format
4. **Demo purposes** - Sufficient for showcasing ML comparison

For production at scale, consider:
- PostgreSQL for relational data
- Redis for caching inference results
- S3/MinIO for image storage

### File Structure

```
/static/originals/{uuid}_original.{ext}   # Original uploaded image
/static/originals/{uuid}_model_a.{ext}    # Model A result with boxes
/static/originals/{uuid}_model_b.{ext}    # Model B result with boxes
/results_details/{uuid}_model_a.json      # Model A detections
/results_details/{uuid}_model_b.json      # Model B detections
/detection_results.csv                     # Aggregate metrics
```

## Security Considerations

### Current Implementation

1. **File Upload Validation**
   - Extension whitelist (png, jpg, jpeg, gif, bmp, webp)
   - MIME type checking
   - File size limits (16MB)

2. **Secret Key Management**
   - Environment variable support
   - Fallback for development

3. **CORS Configuration**
   - Enabled for development
   - Should be restricted in production

### Production Recommendations

1. **Authentication** - Add user authentication (OAuth, JWT)
2. **Rate Limiting** - Implement via nginx or Flask-Limiter
3. **Input Sanitization** - Additional checks on uploaded files
4. **HTTPS** - Terminate SSL at reverse proxy
5. **Secrets Management** - Use Docker secrets or HashiCorp Vault

## Scalability

### Horizontal Scaling

The architecture supports horizontal scaling:

```
                    Load Balancer
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Flask App 1      Flask App 2      Flask App N
        │                │                │
        └────────────────┼────────────────┘
                         │
                  Shared Storage
                    (NFS/S3)
```

### GPU Support

For production with GPU:

```yaml
# docker-compose.yml
services:
  tf-serving-a:
    image: tensorflow/serving:latest-gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Monitoring

### Health Checks

- `/health` endpoint for container orchestration
- TensorFlow Serving status checks
- Request ID tracing for debugging

### Recommended Additions

1. **Prometheus** - Metrics collection
2. **Grafana** - Visualization
3. **ELK Stack** - Log aggregation
4. **Jaeger** - Distributed tracing

## Design Decisions

### Why Two Separate TF Serving Containers?

1. **Isolation** - Independent scaling and failure handling
2. **Resource Allocation** - Different memory/CPU requirements
3. **Model Updates** - Update one without affecting the other
4. **Clarity** - Clear separation of concerns

### Why Not Use TF Serving Multi-Model?

While TF Serving supports multiple models in one container, separate containers provide:

- Better resource isolation
- Independent health checks
- Simpler debugging
- Cleaner Docker configuration

### Why Flask Instead of FastAPI?

Flask was chosen for:

1. **Simplicity** - Easier template rendering
2. **Maturity** - Battle-tested in production
3. **Ecosystem** - Rich extension library

FastAPI would be preferred for:
- Pure API services
- OpenAPI documentation needs
- Async-first architecture

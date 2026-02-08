# Object Detection Model Comparison Platform

> Microservices-based REST API for benchmarking SSD vs. Faster R-CNN inference on wildlife imagery, with real-time performance metrics and side-by-side visualization

**Tech Stack**: TensorFlow Serving • Flask • Docker • SSD ResNet50 FPN • Faster R-CNN ResNet101

---

## Problem Statement

Selecting the optimal object detection model requires quantifying the speed-accuracy trade-off on domain-specific data. This platform enables direct comparison of two production architectures—SSD (single-shot, optimized for latency) and Faster R-CNN (region-based, optimized for precision)—against a custom African wildlife dataset (Ostrich, Warthog, Lion). Results inform deployment decisions: edge scenarios prioritize inference speed, while conservation monitoring may favor detection accuracy.

---

## Key Features

- **Dual-model inference pipeline**: Simultaneous processing through SSD ResNet50 FPN and Faster R-CNN ResNet101 via TensorFlow Serving
- **Real-time performance metrics**: Per-image inference timing, detection counts, and confidence scores logged to persistent storage
- **Side-by-side visualization**: Bounding box overlays rendered with PIL, enabling direct visual comparison of detection quality
- **RESTful API**: JSON endpoints for programmatic access to metrics, comparison data, and health status
- **Containerized microservices**: Three-container architecture (Flask app + 2× TensorFlow Serving) with GPU support via alternate compose file
- **Request tracing**: UUID-based request IDs propagated through logs for debugging distributed inference workflows

---

## System Design

```
                              ┌──────────────────────────┐
                              │      User Browser        │
                              │   (Upload / Results UI)  │
                              └────────────┬─────────────┘
                                           │ HTTP
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Docker Compose Network                              │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Flask Application (Port 80)                          │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │ │
│  │  │  Routes     │  │  Inference   │  │    Image     │  │  Results    │  │ │
│  │  │  /inference │  │  Orchestrator│  │  Processing  │  │  Storage    │  │ │
│  │  │  /results   │  │              │  │  (PIL/NumPy) │  │  (CSV/JSON) │  │ │
│  │  │  /metrics   │  │              │  │              │  │             │  │ │
│  │  └─────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                          │                    │                               │
│             ┌────────────┴────────────────────┴────────────┐                 │
│             ▼                                               ▼                 │
│  ┌─────────────────────────┐               ┌─────────────────────────┐       │
│  │  TensorFlow Serving     │               │  TensorFlow Serving     │       │
│  │  SSD ResNet50 FPN       │               │  Faster R-CNN ResNet101 │       │
│  │  (Port 8501)            │               │  (Port 8502)            │       │
│  │  ~2-3s inference (CPU)  │               │  ~2-4s inference (CPU)  │       │
│  └─────────────────────────┘               └─────────────────────────┘       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Components**:
- **API Layer**: Flask 2.0 with CORS support, request ID middleware, and structured error handling (400/404/500/503)
- **Inference Orchestrator**: Parallel requests to both TensorFlow Serving endpoints with 120s timeout and health monitoring
- **Image Processor**: PIL-based preprocessing, confidence filtering (threshold: 0.5), bounding box rendering with cross-platform font support
- **Results Storage**: CSV aggregate metrics + per-image JSON detection details, supporting analytics aggregation

---

## Models Implemented

| Model | Architecture | Backbone | Input Size | Avg. Inference (CPU) | Use Case |
|-------|--------------|----------|------------|---------------------|----------|
| SSD | Single Shot MultiBox Detector | ResNet50 + FPN | 640×640 | ~2-3s | Lower latency, real-time applications |
| Faster R-CNN | Region-based CNN | ResNet101 | 640×640 | ~2-4s | Higher precision, detailed analysis |

**Detection Classes** (Custom-trained):
| Class ID | Scientific Name | Common Name |
|----------|-----------------|-------------|
| 1 | *Struthio camelus* | Ostrich |
| 2 | *Phacochoerus africanus* | Warthog |
| 3 | *Panthera leo* | Lion |

---

## Quick Start API

**Upload & Detect**:
```bash
curl -X POST http://localhost/inference \
  -F "file=@wildlife_image.jpg"
```

**Get Comparison Results**:
```bash
curl -H "Accept: application/json" \
  http://localhost/api/comparison/{image_id}
```

**Response**:
```json
{
  "image": {
    "image_id": "7b98b3c1-59af-4796-86aa-5b3f3468d42d",
    "filename": "lion-pride.jpg",
    "file_size": 1464721
  },
  "inferences": {
    "model_a": {
      "inference_time": 2.72,
      "detection_count": 4,
      "detections": [
        {"class_name": "Pantheraleo", "confidence": 0.94, "bbox_normalized": [0.12, 0.08, 0.45, 0.67]}
      ]
    },
    "model_b": {
      "inference_time": 2.78,
      "detection_count": 5,
      "detections": [...]
    }
  }
}
```

**Health Check**:
```bash
curl http://localhost/health
```

📘 **Full API documentation**: [docs/API.md](docs/API.md)

---

## How to Run the Project (Two Supported Paths)

This project provides two ways to run the object detection comparison platform, depending on your needs and whether you want to use the pre-trained models or provide your own.

### Prerequisites

- [Docker](https://www.docker.com/get-started) (v20.10+) and Docker Compose
- 4GB+ RAM available for containers
- **For GPU**: NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

### Option A — Recommended: Run the Prebuilt Docker Image (Works Out of the Box)

A fully working image with model variables included is published on Docker Hub.

**Docker Hub**: https://hub.docker.com/r/ahakrami98/object-detection-comparison

```bash
docker pull ahakrami98/object-detection-comparison:latest
docker run -p 80:80 ahakrami98/object-detection-comparison:latest
```

Open in browser: http://localhost

**This image contains:**
- Flask app
- TensorFlow Serving
- SSD + Faster R-CNN models
- All required variables/ directories

**This is the reference implementation and the only way to run the project without additional setup.**

### Option B — Use the GitHub Code With Your Own Models

If you want to:
- Replace the models
- Benchmark different architectures
- Retrain on a new dataset

You can do so by supplying your own TensorFlow SavedModels.

**Clone the repository:**
```bash
git clone https://github.com/amirosseinz/object-detection-comparison.git
cd object-detection-comparison
```

**Required directory structure:**
```
models/
├── label_map.pbtxt
├── ssd/
│   └── 1/
│       ├── saved_model.pb
│       └── variables/
│           ├── variables.index
│           └── variables.data-00000-of-00001
└── faster_rcnn/
    └── 1/
        ├── saved_model.pb
        └── variables/
            ├── variables.index
            └── variables.data-00000-of-00001
```

**Once models are in place:**
```bash
docker compose up --build
```

**Note**: Without valid model variables, TensorFlow Serving will start but inference will fail. This is expected behavior.

### Docker Compose Files

| File | TensorFlow Serving Image | Use Case |
|------|--------------------------|----------|
| `docker-compose.yml` | `tensorflow/serving:latest` | CPU-only deployment (default) |
| `docker-compose.gpu.yml` | `tensorflow/serving:latest-gpu` | GPU-accelerated inference (NVIDIA) |

### GPU Deployment (NVIDIA)

For faster inference on systems with NVIDIA GPU:

```bash
# Verify NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Build and start with GPU support (Option B only)
docker compose -f docker-compose.gpu.yml up --build
```

**GPU Requirements**:
- NVIDIA GPU with CUDA support
- NVIDIA Driver 450.80.02+
- NVIDIA Container Toolkit installed
- Docker configured with `nvidia` runtime

### Verify Deployment

```bash
# Check container status
docker compose ps

# Test health endpoint
curl http://localhost/health

# Test TensorFlow Serving models
curl http://localhost:8501/v1/models/ssd
curl http://localhost:8502/v1/models/faster_rcnn
```

### Stop Services

```bash
# CPU deployment
docker compose down

# GPU deployment
docker compose -f docker-compose.gpu.yml down

# Remove volumes and images (full cleanup)
docker compose down -v --rmi all
```

🚀 **Production deployment guide**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## Technical Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| **Separate TF Serving containers** | Independent scaling, isolated failure domains, different memory/CPU requirements per model architecture |
| **Flask over FastAPI** | Template rendering for web UI, mature ecosystem, simpler session management; async not required for synchronous inference |
| **CSV + JSON storage** | Portability for analysis, no database setup required; appropriate for demonstration scope (production would use PostgreSQL + S3) |
| **50% confidence threshold** | Balances precision/recall for wildlife domain; configurable via `CONFIDENCE_THRESHOLD` |
| **Request ID middleware** | Enables distributed tracing across Flask → TF Serving logs for debugging inference failures |
| **Cross-platform font loading** | Ensures bounding box labels render correctly across Windows, macOS, and Linux containers |

---

## Observed Performance

**Test Environment**: Docker containers on CPU (results vary with hardware and image complexity)

Based on logged inference data across 40+ processed images:

| Metric | SSD (Model A) | Faster R-CNN (Model B) |
|--------|---------------|------------------------|
| Median Inference Time | ~2.0s | ~2.3s |
| Range | 0.8s - 5.7s | 0.3s - 7.0s |
| Typical Detection Count | 1-3 per image | 1-5 per image |

**Observations**:
- First inference after container start incurs model loading overhead (~5-7s)
- Faster R-CNN occasionally detects additional objects at lower confidence
- Large images (>1MB) increase preprocessing time proportionally

---

## Professional Skills Demonstrated

**ML Engineering**:
- Integration of pre-trained TensorFlow SavedModels with TensorFlow Serving inference API
- Confidence thresholding and non-maximum suppression handling
- Bounding box coordinate transformation (normalized → pixel space)

**Backend Development**:
- RESTful API design with proper HTTP status codes and content negotiation
- File upload validation (extension whitelist, MIME checking, 16MB limit)
- Structured logging with request correlation IDs

**DevOps/Infrastructure**:
- Multi-container orchestration with Docker Compose
- Service health checks for container orchestration readiness
- GPU deployment configuration with NVIDIA runtime support
- Environment-based configuration management

**Software Engineering Practices**:
- Error handling with graceful degradation (app remains healthy if TF Serving unavailable)
- Directory structure with separation of concerns (routes, processing, storage)
- Cross-platform compatibility (font loading, path handling)

---

## Repository Structure

```
ObjectDetectionComparisonFlaskApp/
├── app.py                    # Flask application (706 lines) - routes, inference, metrics
├── docker-compose.yml        # CPU deployment orchestration
├── docker-compose.gpu.yml    # GPU deployment with NVIDIA runtime
├── Dockerfile                # Flask container (Python 3.8-slim)
├── requirements.txt          # Dependencies: Flask, Pillow, NumPy, requests
├── detection_results.csv     # Aggregate inference metrics
├── models/
│   ├── label_map.pbtxt       # Class ID → name mapping
│   ├── ssd/                  # SSD SavedModel directory (structure only, no weights)
│   └── faster_rcnn/          # Faster R-CNN SavedModel directory (structure only, no weights)
├── results_details/          # Per-image detection JSON files
├── static/
│   ├── classified/           # Processed images with bounding boxes
│   ├── css/                  # Tailwind-based styling
│   └── js/                   # Chart.js visualizations
├── templates/                # Jinja2 templates (index, results, metrics, errors)
└── docs/
    ├── API.md                # Endpoint documentation
    ├── ARCHITECTURE.md       # System design details
    └── DEPLOYMENT.md         # Cloud deployment guides (AWS, GCP, Azure)
```

**Note**: The `models/ssd/` and `models/faster_rcnn/` directories in the repository contain only the model structure and configuration files. The trained weights (variables) are not included due to GitHub's file size limits. See the "Models and Weights" section above for instructions on obtaining the complete models.

---

## Future Enhancements

- **Batch inference endpoint**: Process multiple images in single request for throughput optimization
- **Model versioning**: A/B testing framework for comparing model iterations
- **Prometheus metrics**: Export inference latency histograms for Grafana dashboards
- **TensorRT optimization**: INT8 quantization for edge deployment scenarios

---

## Contributing

For questions or collaboration:
- Open an issue for bugs or feature requests
- See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design context
- See [API.md](docs/API.md) for endpoint specifications

---

**License**: MIT

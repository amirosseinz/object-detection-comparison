# Deployment Guide

This guide covers deploying the Object Detection Comparison application to various environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Considerations](#production-considerations)
- [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.10+
- Docker Desktop (for TF Serving)
- 8GB RAM minimum

### Setup

1. **Clone and setup virtual environment**
   ```bash
   git clone https://github.com/amirosseinz/object-detection-comparison.git
   cd object-detection-comparison
   
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   
   pip install -r requirements.txt
   ```

2. **Start TensorFlow Serving containers**
   ```bash
   docker run -d --name tf-serving-a \
     -p 8501:8501 \
     -v $(pwd)/models/ssd:/models/ssd \
     tensorflow/serving \
     --model_name=ssd --model_base_path=/models/ssd
   
   docker run -d --name tf-serving-b \
     -p 8502:8501 \
     -v $(pwd)/models/faster_rcnn:/models/faster_rcnn \
     tensorflow/serving \
     --model_name=faster_rcnn --model_base_path=/models/faster_rcnn
   ```

3. **Configure environment**
   ```bash
   export TF_SERVING_HOST_A=localhost
   export TF_SERVING_HOST_B=localhost
   export FLASK_ENV=development
   export SECRET_KEY=dev-secret-key
   ```

4. **Run Flask application**
   ```bash
   flask run --debug --port 80
   ```

---

## Docker Deployment

### Basic Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker compose build
   docker compose up -d
   ```

2. **Verify containers are running**
   ```bash
   docker compose ps
   ```

3. **Check logs**
   ```bash
   docker compose logs -f flask-app
   ```

### Custom Configuration

Create a `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  flask-app:
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - FLASK_ENV=production
    ports:
      - "8080:80"  # Change port
    
  tf-serving-a:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
    
  tf-serving-b:
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '2'
```

### Building for Production

```bash
# Build with no cache for fresh dependencies
docker compose build --no-cache

# Build for specific platform (e.g., Linux/amd64)
docker compose build --platform linux/amd64
```

---

## Cloud Deployment

### AWS EC2

1. **Launch EC2 instance**
   - AMI: Amazon Linux 2 or Ubuntu 22.04
   - Instance type: t3.large minimum (4GB RAM)
   - Storage: 30GB SSD
   - Security Group: Allow ports 22, 80, 443

2. **Install Docker**
   ```bash
   # Amazon Linux 2
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Deploy application**
   ```bash
   git clone https://github.com/amirosseinz/object-detection-comparison.git
   cd object-detection-comparison
   docker compose up -d
   ```

### Google Cloud Run

1. **Build and push to Container Registry**
   ```bash
   # Build Flask app image
   docker build -t gcr.io/PROJECT_ID/detection-app .
   docker push gcr.io/PROJECT_ID/detection-app
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy detection-app \
     --image gcr.io/PROJECT_ID/detection-app \
     --platform managed \
     --region us-central1 \
     --memory 2Gi \
     --port 80
   ```

   > Note: TF Serving would need to be deployed separately (e.g., GKE, Vertex AI)

### Azure Container Instances

```bash
# Create resource group
az group create --name detection-rg --location eastus

# Deploy container group
az container create \
  --resource-group detection-rg \
  --name detection-app \
  --image youracr.azurecr.io/detection-app \
  --cpu 2 --memory 4 \
  --ports 80 \
  --dns-name-label detection-demo
```

---

## Production Considerations

### 1. Reverse Proxy (nginx)

Add nginx for SSL termination and caching:

```nginx
# nginx.conf
upstream flask_app {
    server flask-app:80;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    client_max_body_size 16M;

    location / {
        proxy_pass http://flask_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /app/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

### 2. Environment Variables

Never commit secrets. Use:

```bash
# .env file (not committed)
SECRET_KEY=your-production-secret-key-min-32-chars
FLASK_ENV=production

# Or Docker secrets
echo "your-secret-key" | docker secret create flask_secret_key -
```

### 3. Health Checks

Add to docker-compose.yml:

```yaml
services:
  flask-app:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 4. Logging

Configure structured logging:

```python
# In app.py
import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
```

### 5. Monitoring

Add Prometheus metrics:

```yaml
# docker-compose.yml addition
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs flask-app
docker compose logs tf-serving-a
docker compose logs tf-serving-b

# Check if ports are in use
netstat -tulpn | grep -E '80|8501|8502'

# Restart containers
docker compose restart
```

### TensorFlow Serving Errors

```bash
# Verify model files exist
ls -la models/ssd/1/
ls -la models/faster_rcnn/1/

# Check TF Serving health
curl http://localhost:8501/v1/models/ssd
curl http://localhost:8502/v1/models/faster_rcnn
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Reduce batch size or model complexity
# Or increase container memory limits in docker-compose.yml
```

### Slow Inference

1. **Check CPU vs GPU mode**
   ```bash
   # TF Serving logs will show if using GPU
   docker logs tf-serving-a 2>&1 | grep -i gpu
   ```

2. **Optimize container resources**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 4G
       reservations:
         cpus: '2'
         memory: 2G
   ```

3. **Consider GPU deployment**
   ```bash
   docker run --gpus all tensorflow/serving:latest-gpu ...
   ```

### Permission Issues

```bash
# Fix file permissions
sudo chown -R 1000:1000 ./models ./static

# On Linux, may need to set SELinux context
sudo chcon -Rt svirt_sandbox_file_t ./models ./static
```

---

## Quick Reference

### Common Commands

```bash
# Start application
docker compose up -d

# Stop application
docker compose down

# View logs
docker compose logs -f

# Rebuild after changes
docker compose build && docker compose up -d

# Clean up
docker compose down -v --rmi all
docker system prune -af
```

### Useful URLs

| Service | Local URL | Purpose |
|---------|-----------|---------|
| Web App | http://localhost | Main application |
| TF Serving A | http://localhost:8501/v1/models/ssd | Model A status |
| TF Serving B | http://localhost:8502/v1/models/faster_rcnn | Model B status |
| Health Check | http://localhost/health | Application health |

---

## Support

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/amirosseinz/object-detection-comparison/issues)
2. Review container logs
3. Ensure all prerequisites are met
4. Create a new issue with:
   - Operating system
   - Docker version
   - Error messages
   - Steps to reproduce

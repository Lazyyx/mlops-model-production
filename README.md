# MLOps Face Detection System

A production-grade MLOps project for MTCNN-based real-time face detection. Built with modern Python technologies and containerized for seamless deployment.

**Authors:** Eugénie Beauvillain & Léo Lopes  
**Institution:** EPITA  

---

## 📋 Project Overview

This project demonstrates a complete MLOps pipeline for deploying a machine learning model in production. It features an advanced MTCNN (Multi-task Cascaded Convolutional Networks) face detection service with a professional API backend and an interactive web interface.

The system detects faces in images with configurable detection thresholds, returning precise bounding boxes, confidence scores, and facial keypoints (eyes, mouth, nose).

### Key Objectives
- Deploy a pre-trained MTCNN model in a production environment
- Provide both REST API and web-based interfaces
- Implement security best practices (API key authentication)
- Demonstrate containerization and orchestration

---

## 🏗️ Architecture

The project follows a microservices architecture with three main components:

```
┌─────────────────────────────────────────────────────┐
│                    NGINX Reverse Proxy               │
│                   (Port 80 Gateway)                  │
└──────────┬───────────────────────┬──────────────────┘
           │                       │
    ┌──────▼────────┐    ┌────────▼─────────┐
    │   Frontend     │    │    Backend API   │
    │   (Streamlit)  │    │    (FastAPI)     │
    │   :8501        │    │    :8000         │
    └────────────────┘    └────────┬─────────┘
                                   │
                          ┌────────▼──────────┐
                          │  MTCNN Detector   │
                          │  (TensorFlow)     │
                          └───────────────────┘
```

### Services

#### 🔙 Backend API (FastAPI)
- **Framework:** FastAPI 0.121.1+
- **Port:** 8000
- **Features:**
  - RESTful API with OpenAPI/Swagger documentation
  - Multiple detection endpoints
  - Configurable MTCNN thresholds (PNet, RNet, ONet)
  - API key authentication via `X-API-Key` header
  - Health check endpoint
  - Detailed face detection with bounding boxes, keypoints, and confidence scores

#### 🎨 Frontend UI (Streamlit)
- **Framework:** Streamlit 1.51.0+
- **Port:** 8501
- **Features:**
  - Interactive web interface for image upload
  - Real-time face detection visualization
  - Bounding box rendering with configurable thresholds
  - Facial keypoints visualization (eyes, mouth, nose)
  - Confidence score display

#### 🌐 Reverse Proxy (Nginx)
- **Container:** Nginx Alpine
- **Port:** 80
- **Routes:**
  - `/` → Frontend (Streamlit)
  - `/api/*` → Backend API
  - `/openapi.json` → OpenAPI specification

---

## 📁 Project Structure

```
mlops-model-production/
├── README.md                  # This file
├── pyproject.toml             # Project dependencies (uv package manager)
├── docker-compose.yml         # Production Docker Compose configuration
├── docker-compose.dev.yml     # Development Docker Compose configuration
├── nginx.conf                 # Nginx reverse proxy configuration
├── locustfile.py              # Load testing configuration
│
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI application & endpoints
│   │   ├── security.py        # API authentication logic
│   │   └── Dockerfile         # Backend container build config
│   │
│   ├── models/
│   │   └── MTCNN.py           # Face detection model wrapper
│   │
│   └── ui/
│       ├── app.py             # Streamlit web application
│       └── Dockerfile         # Frontend container build config
│
```

---

## 🚀 Getting Started

### Prerequisites
- **Docker & Docker Compose** (recommended)
- **Python 3.12+** (for local development)
- **4GB+ RAM** (for TensorFlow/MTCNN models)

### Quick Start (Docker)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lazyyx/mlops-model-production.git
   cd mlops-model-production
   ```

2. **Set up environment variables:**
   ```bash
   touch .env
   # Edit .env and set your APP_TOKEN
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```

4. **Access the services:**
   - 🌐 **Web UI:** http://localhost
   - 📚 **API Docs:** http://localhost/api/docs
   - 🏥 **Health Check:** http://localhost/api/health

### Local Development

1. **Install dependencies** (requires [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv sync
   ```

2. **Start backend:**
   ```bash
   source .venv/bin/activate
   uvicorn src.api.main:app --reload --port 8000
   ```

3. **Start frontend** (in another terminal):
   ```bash
   source .venv/bin/activate
   streamlit run src/ui/app.py --server.port=8501
   ```

---

## 🔌 API Documentation

### Authentication
All API endpoints (except health check) require an `X-API-Key` header:
```bash
curl -H "X-API-Key: your-secret-token" http://localhost:8000/detect
```

### Endpoints

#### Health Check
```
GET /health
```
Returns service status.

#### Basic Face Detection
```
POST /detect
```
Returns only bounding boxes.

**Parameters:**
- `file` (required): Image file to analyze
- `min_face_size` (default: 20): Minimum face size in pixels
- `threshold_pnet` (default: 0.6): PNet confidence threshold
- `threshold_rnet` (default: 0.7): RNet confidence threshold
- `threshold_onet` (default: 0.7): ONet confidence threshold

**Response:**
```json
{
  "boxes": [[x1, y1, x2, y2], ...],
  "keypoints": [{"left_eye": [x, y], ...}, ...],
  "scores": [0.95, 0.87, ...]
}
```

#### Full API Documentation
Visit http://localhost:8000/docs for interactive Swagger UI with all endpoints.

---

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.12+ |
| **Backend** | FastAPI | 0.121.1+ |
| **Frontend** | Streamlit | 1.51.0+ |
| **ML Framework** | TensorFlow | 2.20.0+ |
| **Face Detection** | MTCNN | 1.0.0+ |
| **Image Processing** | Pillow, Matplotlib | Latest |
| **Server** | Uvicorn | 0.38.0+ |
| **Package Manager** | uv | Latest |
| **Containerization** | Docker | Latest |
| **Orchestration** | Docker Compose | Latest |
| **Reverse Proxy** | Nginx | Alpine |
| **Load Testing** | Locust | Latest |

---

## 📦 Deployment

### Docker Compose (Production)
```bash
docker-compose up -d
```

Uses pre-built images from Google Cloud Artifact Registry:
- `us-central1-docker.pkg.dev/project-cd53e609-dbd3-4daf-8e2/mlops-repo/backend:latest`
- `us-central1-docker.pkg.dev/project-cd53e609-dbd3-4daf-8e2/mlops-repo/frontend:latest`

### Docker Compose (Development)
```bash
docker-compose -f docker-compose.dev.yml up -d
```

Builds images locally for development and testing.

### Manual Deployment
Each service can be deployed independently:

**Backend:**
```bash
docker build -t face-backend:latest -f src/api/Dockerfile .
docker run -p 8000:8000 -e APP_TOKEN=your-secret face-backend:latest
```

**Frontend:**
```bash
docker build -t face-frontend:latest -f src/ui/Dockerfile .
docker run -p 8501:8501 -e API_URL=http://backend:8000 face-frontend:latest
```

---

## 🧪 Testing & Performance

### Load Testing
The project includes Locust configuration for load testing:
```bash
locust -f locustfile.py -H http://localhost:8000
```

### Configuration Files
- **`.env`** - Environment variables (API tokens, service URLs)
- **`docker-compose.yml`** - Production orchestration
- **`docker-compose.dev.yml`** - Development orchestration
- **`nginx.conf`** - Reverse proxy routing rules
- **`pyproject.toml`** - Python dependencies

---

## 🔐 Security Considerations

1. **API Authentication:** All endpoints protected with API key authentication
2. **Environment Variables:** Sensitive data (tokens) stored in `.env` file
3. **Reverse Proxy:** Nginx handles routing and serves as security layer
4. **Containerization:** Isolated microservices limit attack surface
5. **HTTPS:** Add SSL certificates in production (Nginx ready)

### Best Practices
- Store `APP_TOKEN` in secrets management system (not in `.env`)
- Enable CORS only when necessary
- Monitor API requests and implement rate limiting
- Regular security updates for dependencies
- Use read-only file systems where possible

---

## 📊 Model Information

### MTCNN (Multi-task Cascaded Convolutional Networks)
- **Purpose:** Real-time face detection
- **Architecture:** 3-stage cascade (PNet, RNet, ONet)
- **Input:** RGB images (any size)
- **Output:** 
  - Bounding boxes: `[x1, y1, x2, y2]` coordinates
  - Confidence scores: Detection confidence (0-1)
  - Facial keypoints: Left eye, right eye, nose, mouth corners

---

## 📝 Environment Variables

Create a `.env` file in the project root:

```bash
# API Authentication
APP_TOKEN=your-secure-api-token-here

# Optional: Override service URLs
API_URL=http://backend:8000
FRONTEND_URL=http://frontend:8501
```

---

## 🤝 Contributing

This project was developed as part of the EPITA MLOps curriculum by Eugénie Beauvillain and Léo Lopes.

---
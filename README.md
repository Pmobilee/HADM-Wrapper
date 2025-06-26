# HADM Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)

A FastAPI-based server interface for Human Artifact Detection in Machine-generated images (HADM), providing both local and global detection capabilities through RESTful APIs.

## 🎯 Overview

HADM Server is a production-ready web service that wraps the [HADM (Human Artifact Detection in Machine-generated images)](https://github.com/wangkaihong/HADM) algorithms, enabling real-time detection of AI-generated image artifacts through a simple HTTP API. The service supports both HADM-L (Local) and HADM-G (Global) detection methods.

### Key Features

- **🚀 High Performance**: Asynchronous FastAPI framework for optimal performance
- **🔍 Dual Detection**: Support for both local and global artifact detection
- **📊 Multiple Formats**: Accept various image formats (JPEG, PNG, WebP, etc.)
- **🛡️ Production Ready**: Built-in security, rate limiting, and error handling
- **📖 Auto Documentation**: Interactive API documentation with Swagger UI
- **🐳 Containerized**: Docker support for easy deployment
- **⚡ Scalable**: Designed for horizontal scaling and load balancing

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   HADM Server    │───▶│   HADM Models   │
│                 │    │   (FastAPI)      │    │   (L & G)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   File Storage   │
                       │   & Processing   │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **CUDA-capable GPU** (recommended) or CPU
- **Git** for repository management
- **4GB+ RAM** for optimal performance

### ⚡ One-Command Setup

```bash
# Clone and setup everything automatically
git clone https://github.com/yourusername/HADM_server.git
cd HADM_server
chmod +x setup.sh && ./setup.sh
```

### 🔧 Manual Installation

<details>
<summary>Click to expand manual setup steps</summary>

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HADM_server.git
   cd HADM_server
   ```

2. **Run setup script**
   ```bash
   chmod +x scripts/setup_environment.sh
   ./scripts/setup_environment.sh
   ```

3. **Download model files**
   ```bash
   ./scripts/download_models.sh
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

</details>

### 🚀 Launch Options

#### **Development Mode**
```bash
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### **Production Mode**
```bash
docker-compose up -d
```

#### **Quick Test**
```bash
# Check if everything works
curl http://localhost:8000/api/v1/health
```

### 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8000/docs | Interactive Swagger UI |
| **Alternative Docs** | http://localhost:8000/redoc | ReDoc interface |
| **Health Check** | http://localhost:8000/api/v1/health | Service status |
| **Service Info** | http://localhost:8000/api/v1/info | Detailed information |

## 📋 API Endpoints

### 🎯 Core Detection Services

| Method | Endpoint | Purpose | Response Time |
|--------|----------|---------|---------------|
| `POST` | `/api/v1/detect/local` | **HADM-L**: Bounding box detection | ~500ms |
| `POST` | `/api/v1/detect/global` | **HADM-G**: Whole-image classification | ~300ms |
| `POST` | `/api/v1/detect/both` | **Combined**: Local + Global analysis | ~800ms |
| `POST` | `/api/v1/detect` | **Universal**: Configurable detection | Variable |

### 🔧 Utility Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/v1/health` | Service health & model status |
| `GET` | `/api/v1/info` | Detailed service information |
| `GET` | `/docs` | Interactive API documentation |

### 💡 Usage Examples

#### **Python Client**
```python
import requests
from pathlib import Path

# Single detection
def detect_artifacts(image_path: str, detection_type: str = "both"):
    url = f"http://localhost:8000/api/v1/detect/{detection_type}"
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            url,
            files={'image': f},
            data={'confidence_threshold': 0.7}
        )
    
    return response.json()

# Example usage
result = detect_artifacts('suspicious_image.jpg')
print(f"Found {len(result.get('local_detections', []))} local artifacts")
```

#### **cURL Commands**
```bash
# Local detection with custom threshold
curl -X POST "http://localhost:8000/api/v1/detect/local" \
     -F "image=@image.jpg" \
     -F "confidence_threshold=0.8"

# Global detection
curl -X POST "http://localhost:8000/api/v1/detect/global" \
     -F "image=@image.jpg"

# Health check
curl http://localhost:8000/api/v1/health
```

#### **JavaScript/Node.js**
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function detectArtifacts(imagePath) {
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));
    form.append('detection_type', 'both');
    
    const response = await axios.post(
        'http://localhost:8000/api/v1/detect',
        form,
        { headers: form.getHeaders() }
    );
    
    return response.data;
}
```

## 🐳 Docker Deployment

### Build and run with Docker

```bash
# Build the image
docker build -t hadm-server .

# Run the container
docker run -p 8000:8000 hadm-server
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## 📊 Performance

- **Throughput**: Up to 100 requests/second (depending on hardware)
- **Latency**: < 500ms average response time
- **Memory**: ~2GB RAM recommended for optimal performance
- **GPU Support**: CUDA acceleration available for faster processing

## 🔧 Configuration

Environment variables can be set in a `.env` file:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Configuration
MODEL_PATH=./models
DEVICE=cuda  # or cpu

# Security
MAX_FILE_SIZE=10485760  # 10MB
RATE_LIMIT=100  # requests per minute
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original HADM research by [Wang et al.](https://github.com/wangkaihong/HADM)
- FastAPI framework by [Sebastián Ramirez](https://github.com/tiangolo)
- The open-source community for their valuable contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/HADM_server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/HADM_server/discussions)
- **Email**: support@yourproject.com

## 🗺️ Roadmap

See our [TODO.md](TODO.md) for the complete development roadmap and current progress.

---

**Note**: This project is based on the HADM research for academic and research purposes. Please ensure compliance with applicable licenses and regulations when using in production environments.
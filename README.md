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

2. **Create virtual environment**
   ```bash
   python3 -m venv venv --prompt HADM_server
   source venv/bin/activate
   ```

3. **Install system dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential ninja-build cmake pkg-config \
       libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev libpng-dev
   ```

4. **Install Python dependencies**
   ```bash
   pip install -U pip wheel setuptools openmim ninja psutil cmake
   
   # Install PyTorch with CUDA support
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
       --index-url https://download.pytorch.org/whl/cu121
   
   # Install Detectron2
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html
   ```

5. **⚠️ CRITICAL: Install MMCV-Full with CUDA ops**
   ```bash
   # Set environment variables
   export MMCV_WITH_OPS=1
   export CUDA_HOME=/usr/local/cuda-12.1
   
   # Try pre-built wheel first
   pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html
   
   # If pre-built wheel fails, build from source (10-20 minutes)
   git clone --depth 1 https://github.com/open-mmlab/mmcv.git
   cd mmcv
   pip install -v -e .
   cd ..
   ```

6. **Install remaining dependencies**
   ```bash
   pip install fastapi uvicorn[standard] python-multipart pydantic \
       Pillow==9.0.0 opencv-python numpy cloudpickle fairscale
   ```

7. **Clone HADM repository**
   ```bash
   git clone https://github.com/marlinfiggins/HADM.git
   ```

8. **Download model files**
   ```bash
   chmod +x scripts/download_models.sh
   ./scripts/download_models.sh
   ```

9. **Verify installation**
   ```bash
   python diagnose_models.py
   ```

</details>

### 🚨 Important Notes

- **MMCV-Full is CRITICAL**: The server requires `mmcv-full` with CUDA ops compiled. The regular `mmcv` package will NOT work.
- **CUDA Version**: Ensure your CUDA version matches the PyTorch and MMCV installations (default: CUDA 12.1).
- **GPU Memory**: Each model requires ~5GB VRAM. Ensure sufficient GPU memory is available.
- **Build Time**: Building MMCV from source takes 10-20 minutes but ensures compatibility.

### 🚀 Launch Options

#### **Development Mode**
```bash
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

#### **Production Mode**
```bash
docker-compose up -d
```

#### **Quick Test**
```bash
# Check if everything works
curl http://localhost:8080/api/v1/health
```

### 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8080/docs | Interactive Swagger UI |
| **Alternative Docs** | http://localhost:8080/redoc | ReDoc interface |
| **Health Check** | http://localhost:8080/api/v1/health | Service status |
| **Service Info** | http://localhost:8080/api/v1/info | Detailed information |

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
    url = f"http://localhost:8080/api/v1/detect/{detection_type}"
    
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
curl -X POST "http://localhost:8080/api/v1/detect/local" \
     -F "image=@image.jpg" \
     -F "confidence_threshold=0.8"

# Global detection
curl -X POST "http://localhost:8080/api/v1/detect/global" \
     -F "image=@image.jpg"

# Health check
curl http://localhost:8080/api/v1/health
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
        'http://localhost:8080/api/v1/detect',
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
docker run -p 8080:8080 hadm-server
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
PORT=8080
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

## 🚨 Troubleshooting

### Common Issues

#### "Dependencies not available for model loading"
This error indicates MMCV-Full with CUDA ops is not properly installed:

```bash
# Check if MMCV ops are available
python -c "from mmcv import ops; print('MMCV ops OK')"

# If this fails, reinstall MMCV-Full:
pip uninstall mmcv mmcv-full -y
export MMCV_WITH_OPS=1
export CUDA_HOME=/usr/local/cuda-12.1
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html
```

#### "No module named 'mmcv._ext'"
You have the lite version of MMCV. Install the full version:

```bash
pip uninstall mmcv -y
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0/index.html
```

#### Models not loading into GPU memory
Check if CUDA is properly configured:

```bash
# Verify CUDA installation
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run diagnostics
python diagnose_models.py
```

#### Build errors during MMCV compilation
Ensure all build tools are installed:

```bash
sudo apt-get install -y build-essential ninja-build cmake
pip install ninja psutil
```

### Getting Help

1. **Run diagnostics first**: `python diagnose_models.py`
2. **Check logs**: Look at server logs for specific error messages
3. **Verify dependencies**: Ensure all requirements are properly installed
4. **GPU memory**: Make sure you have enough VRAM (10GB+ recommended)

## 🗺️ Roadmap

See our [TODO.md](TODO.md) for the complete development roadmap and current progress.

---

**Note**: This project is based on the HADM research for academic and research purposes. Please ensure compliance with applicable licenses and regulations when using in production environments.
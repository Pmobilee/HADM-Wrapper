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

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HADM_server.git
   cd HADM_server
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

## 📋 API Endpoints

### Core Detection Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect/local` | HADM-L: Local artifact detection |
| `POST` | `/detect/global` | HADM-G: Global artifact detection |
| `POST` | `/detect/both` | Run both detection methods |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/info` | Service information and version |

### Example Usage

```python
import requests

# Local detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/local',
        files={'image': f}
    )
    result = response.json()
    print(f"Artifact probability: {result['probability']}")
```

```bash
# Using curl
curl -X POST "http://localhost:8000/detect/local" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@your_image.jpg"
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
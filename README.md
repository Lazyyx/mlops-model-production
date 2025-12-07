# Face Detection API

This project provides a face detection API using a pre-trained MTCNN model from the `facenet-pytorch` library. The project includes a FastAPI backend and a Streamlit web UI.

## Features
- **API**: Detect faces in images.
- **Web UI**: Simple interface to upload an image and see the detected faces.
- **Dockerized**: Easily deployable using Docker.
- **CI/CD**: Automated deployment to a VM using GitHub Actions.

## Project Structure
```
.
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── README.md
├── src
│   ├── api
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── models
│   │   └── MTCNN.py
│   └── ui
│       ├── app.py
│       └── Dockerfile
└── tests
```

## Setup

### Prerequisites
- Python 3.12
- Docker

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Lazyyx/mlops-model-production.git
   cd mlops-model-production
   ```

2. Run the application using docker-compose:
   ```bash
   docker-compose up -d
   ```

The API swagger will be available at `http://localhost:8000/docs` and the UI at `http://localhost:8501`.

## API

The API has a main endpoint `/detect` that accepts an image file and returns the bounding boxes of the detected faces.

### Example Usage
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8000/detect
```

## CI/CD

This project uses GitHub Actions for CI/CD. On every push to the `master` branch, the application is built and deployed to a VM.


## License
This project is licensed under the MIT License.

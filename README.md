# Sentiment Analysis API

This project provides a sentiment analysis API for movie reviews using a pre-trained `distilbert-base-uncased` model from Hugging Face. The project includes a FastAPI backend, a web UI, and CI/CD pipeline for deployment.

## Features
- **API**: Analyze sentiment of movie reviews.
- **Web UI**: Simple interface to interact with the API.
- **Dockerized**: Easily deployable using Docker.
- **CI/CD**: Automated deployment to a VM using GitHub Actions.

## Setup

### Prerequisites
- Python 3.9+
- Docker
- Node.js (optional, for UI development)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Lazyyx/mlops-model-production.git
   cd mlops-model-production
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   uvicorn src.api.main:app --reload
   ```

4. Open the web UI:
   Open `src/ui/index.html` in your browser.

## Docker

### Build and Run
1. Build the Docker image:
   ```bash
   docker build -t sentiment-analysis-api .
   ```

2. Run the container:
   ```bash
   docker run -d -p 80:80 sentiment-analysis-api
   ```

## CI/CD

This project uses GitHub Actions for CI/CD. On every push to the `master` branch, the application is built and deployed to a VM.

### Deployment
1. Set up SSH access to your VM.
2. Update the VM IP and user in `.github/workflows/deploy.yml`.
3. Push changes to `master` to trigger deployment.

## License
This project is licensed under the MIT License.
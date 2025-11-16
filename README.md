# ML Ops Major Assignment

## Overview
This project implements an **end-to-end MLOps pipeline** integrating modern tools for model training, containerization, automation, and deployment.  

The workflow covers the complete ML lifecycle — from training to production — using the following technologies:

- **Scikit-Learn** → Model training and evaluation (Decision Tree Classifier on the Olivetti Faces dataset)  
- **GitHub Actions (CI/CD)** → Automated build and deployment pipeline  
- **Docker** → Containerization of the Flask inference service  
- **Kubernetes** → Scalable deployment with **3 replicas**

---

## Project Workflow
1. **Model Training**  
   - A `DecisionTreeClassifier` is trained on the **Olivetti Faces dataset** using scikit-learn.  
   - The trained model is serialized and stored as `savedmodel.pth`.

2. **Flask API for Inference**  
   - The model is wrapped inside a Flask web application (`app.py`) providing an HTTP endpoint for predictions.  
   - The API accepts a 64×64 grayscale image and returns the predicted face class.

3. **Containerization**  
   - A `Dockerfile` is created to build the image.  
   - The container exposes port **5000** and serves predictions via Flask.

4. **CI/CD Automation**  
   - **GitHub Actions** pipeline automatically builds and pushes the Docker image to Docker Hub whenever changes are pushed to the main branch.  

5. **Kubernetes Deployment**  
   - The application is deployed as **3 replicas** using `k8s-deployment.yaml`.  
   - Kubernetes ensures load balancing and high availability via a service (`mlops-major-service`).

---

## Tech Stack
| Component | Technology Used |
|------------|----------------|
| **Modeling** | Scikit-Learn (Decision Tree Classifier) |
| **Language** | Python 3.11 |
| **Web Framework** | Flask |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes |
| **CI/CD** | GitHub Actions |
| **Dataset** | Olivetti Faces (from `sklearn.datasets`) |

---

## 🐳 Docker Commands
```bash
# Build Docker image
docker build -t paresh2045/mlops-major:latest .

# Run container
docker run -p 5000:5000 paresh2045/mlops-major:latest


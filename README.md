# 🚗 Vehicle Insurance MLOps

<p align="center">
  <b>End-to-End Machine Learning + MLOps Pipeline</b><br>
  <i>From data ingestion and validation to model training, AWS model registry, Docker, CI/CD, and production deployment.</i>
</p>

---

## 📌 Project Overview

This project demonstrates a **production-oriented MLOps workflow** for a Vehicle Insurance machine learning application.

The goal is not only to train a machine learning model, but to build the complete ecosystem required to take an ML project from **development → training → evaluation → model registry → containerization → CI/CD → cloud deployment**.

### ✨ What this project demonstrates

- 🏗️ Modular and reusable Python project structure
- 📦 Local package management with `setup.py` and `pyproject.toml`
- 🐍 Isolated Conda environment
- 🍃 MongoDB Atlas for data storage
- 📊 Exploratory Data Analysis and Feature Engineering
- 🔄 Automated Data Ingestion
- ✅ Data Validation using schema-driven validation
- 🛠️ Data Transformation and preprocessing
- 🤖 Model Training
- 📈 Model Evaluation
- ☁️ AWS S3-based model registry
- 🚀 Prediction Pipeline
- 🌐 FastAPI web application
- 🐳 Docker containerization
- ⚙️ GitHub Actions CI/CD
- ☁️ AWS ECR + EC2 deployment
- 🏃 GitHub self-hosted runner
- 🔐 Environment variables and GitHub Secrets

---

## 🧠 End-to-End MLOps Workflow

```text
                         ┌──────────────────────┐
                         │     MongoDB Atlas    │
                         │   Raw Vehicle Data   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Data Ingestion    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Data Validation    │
                         │   Schema Checking    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Data Transformation│
                         │ Preprocessing / FE    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Model Trainer     │
                         │ Train ML Model       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Model Evaluation  │
                         │ Compare Performance │
                         └──────────┬───────────┘
                                    │
                          Better Model?
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    AWS S3 Registry  │
                         │   Store / Retrieve   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Prediction Pipeline  │
                         │       FastAPI        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Docker Image    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │       GitHub Actions CI/CD     │
                    └────────────────┬───────────────┘
                                     │
                           ┌─────────┴─────────┐
                           ▼                   ▼
                    ┌────────────┐      ┌────────────┐
                    │ AWS ECR    │      │ AWS EC2    │
                    │ Image Store│─────▶│ Production │
                    └────────────┘      └─────┬──────┘
                                              │
                                              ▼
                                      🌐 Vehicle App
                                      Port `5080`
```

---

# 🛠️ Tech Stack

| Area | Technology |
|---|---|
| Programming | Python 3.10 |
| Data Storage | MongoDB Atlas |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| API | FastAPI |
| Templating | Jinja2 / HTML |
| Environment | Conda |
| Packaging | `setup.py`, `pyproject.toml` |
| Logging | Python Logging |
| Error Handling | Custom Exception Handling |
| Cloud | AWS |
| Object Storage | AWS S3 |
| Containerization | Docker |
| Container Registry | AWS ECR |
| Compute | AWS EC2 |
| CI/CD | GitHub Actions |
| CI Runner | GitHub Self-hosted Runner |
| Version Control | Git + GitHub |

---

# 📂 Project Structure

```text
vehicle-insurance-mlops/
│
├── .github/
│   └── workflows/
│       └── aws.yaml
│
├── notebook/
│   ├── mongoDB_demo.ipynb
│   └── EDA_and_Feature_Engineering.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   ├── mongo_db_connections.py
│   │   └── aws_connection.py
│   │
│   ├── data_access/
│   │   └── proj1_data.py
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   └── s3_estimator.py
│   │
│   ├── aws_storage/
│   │   └── ...
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── utils/
│   │   └── main_utils.py
│   │
│   ├── exception.py
│   └── logger.py
│
├── static/
├── template/
├── app.py
├── demo.py
├── template.py
├── requirements.txt
├── setup.py
├── pyproject.toml
├── config/
│   └── schema.yaml
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

# 🚀 Getting Started

## 1️⃣ Create the Project Template

Run the project template generator:

```bash
python template.py
```

This creates the initial project structure and directories.

---

## 2️⃣ Configure Local Package Import

Configure:

```text
setup.py
pyproject.toml
```

These files allow the project to be installed as a local Python package.

For example:

```bash
pip install -r requirements.txt
```

and verify the local package installation using:

```bash
pip list
```

---

# 🐍 3️⃣ Create the Virtual Environment

Create the Conda environment:

```bash
conda create -n vehicle python=3.10 -y
```

Activate it:

```bash
conda activate vehicle
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Verify:

```bash
pip list
```

---

# 🍃 MongoDB Atlas Setup

MongoDB Atlas is used as the project's source database.

### Setup

1. Create a MongoDB Atlas account.
2. Create a new project.
3. Create a cluster.
4. Select the **M0** service.
5. Create a database user.
6. Configure network access.
7. Add your connection IP/network according to your deployment requirements.
8. Get the Python driver connection string.
9. Replace the password placeholder with your database password.

> ⚠️ **Security:** For production, avoid exposing your database to `0.0.0.0/0` unless there is a specific reason. Prefer restricting access to trusted IP addresses or networks.

### MongoDB Demo

Create:

```text
notebook/mongoDB_demo.ipynb
```

Select the `vehicle` Python kernel and load the dataset.

The notebook demonstrates:

```text
Dataset
   ↓
Python / Pandas
   ↓
MongoDB Atlas
   ↓
MongoDB Collection
```

After inserting the data, verify it from:

```text
MongoDB Atlas
→ Database
→ Browse Collections
```

---

# 📝 Logging & Exception Handling

A production ML project should provide useful logs and meaningful errors.

This project implements:

- Centralized logging
- Custom exception handling
- Error location tracking
- Debug-friendly error messages

Test both using:

```text
demo.py
```

This makes failures significantly easier to diagnose during development and deployment.

---

# 📊 Exploratory Data Analysis

Before building the pipeline, perform:

- Dataset exploration
- Missing-value analysis
- Distribution analysis
- Categorical feature analysis
- Numerical feature analysis
- Outlier investigation
- Feature engineering
- Relationship analysis

Notebook:

```text
notebook/EDA_and_Feature_Engineering.ipynb
```

---

# 🔄 Data Ingestion

The Data Ingestion component is responsible for retrieving data from MongoDB and converting it into a form suitable for ML processing.

### Flow

```text
MongoDB Atlas
      ↓
MongoDB Connection
      ↓
Data Access Layer
      ↓
Key-Value Documents
      ↓
Pandas DataFrame
      ↓
Train/Test Dataset
```

Important modules include:

```text
constants
configuration
data_access
entity
components
pipeline
```

The ingestion configuration and artifact classes provide a clean separation between:

- Configuration
- Runtime artifacts
- Component implementation

---

# ✅ Data Validation

The Data Validation component checks whether the incoming dataset follows the expected structure.

Schema information is maintained in:

```text
config/schema.yaml
```

Typical validation responsibilities include:

- Expected columns
- Data types
- Number of columns
- Dataset structure
- Validation status
- Valid/invalid data artifacts

```text
Raw Dataset
     ↓
Schema Validation
     ↓
Valid Dataset ──────▶ Continue Pipeline
     │
     └───────────────▶ Invalid → Stop / Report
```

---

# 🛠️ Data Transformation

The Data Transformation component converts validated raw data into ML-ready features.

It can include:

- Missing-value handling
- Encoding categorical variables
- Numerical transformations
- Feature preprocessing
- Train/test transformations
- Saving preprocessing estimators

The transformation estimator is maintained within the `entity` layer and is reused during prediction.

---

# 🤖 Model Training

The Model Trainer component:

1. Loads transformed training data.
2. Loads transformed testing data.
3. Trains the selected ML model.
4. Evaluates training/test performance.
5. Saves the trained model artifact.

```text
Transformed Data
       ↓
Model Training
       ↓
Trained Model
       ↓
Model Artifact
```

---

# 📈 Model Evaluation

Before deploying a newly trained model, its performance is compared with the currently accepted model.

The evaluation threshold is configured using:

```python
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
```

Conceptually:

```text
New Model
    │
    ▼
Evaluate Performance
    │
    ▼
Compare with Existing Model
    │
    ├── Better enough ──▶ Push Model
    │
    └── Not better ─────▶ Reject / Keep Existing
```

This prevents blindly replacing a production model with a model that does not provide sufficient improvement.

---

# ☁️ AWS Model Registry

AWS S3 is used to store and retrieve trained models.

Required configuration includes:

```python
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
```

The S3 estimator layer provides functionality for:

- Uploading models
- Downloading models
- Maintaining model artifacts
- Connecting the ML pipeline with cloud storage

### AWS Services Used

```text
AWS IAM
   ↓
AWS S3
   ↓
Model Registry
```

---

# 🔮 Prediction Pipeline

The prediction pipeline loads the trained model and preprocessing objects to generate predictions for new input data.

```text
User Input
    ↓
FastAPI
    ↓
Prediction Pipeline
    ↓
Preprocessing
    ↓
Trained Model
    ↓
Prediction
    ↓
Web Response
```

The application is exposed through:

```text
app.py
```

Static assets and HTML templates are maintained in:

```text
static/
template/
```

---

# 🌐 Application Routes

The application includes a prediction workflow and a model-training route.

Example:

```text
/
```

for the main application and:

```text
/training
```

for triggering model training.

> Exact routes may vary depending on the final FastAPI implementation.

---

# 🐳 Docker

The application is containerized using Docker.

Important files:

```text
Dockerfile
.dockerignore
```

Build the image:

```bash
docker build -t vehicleproj .
```

Run the container:

```bash
docker run -p 5080:5080 vehicleproj
```

The application can then be accessed on:

```text
http://localhost:5080
```

---

# ⚙️ CI/CD with GitHub Actions

The project implements an automated CI/CD workflow.

```text
Developer
    │
    ▼
Git Push
    │
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ▼
Self-Hosted Runner
    │
    ▼
Docker Build
    │
    ▼
AWS ECR
    │
    ▼
AWS EC2
    │
    ▼
🚀 Deployed Application
```

Workflow file:

```text
.github/workflows/aws.yaml
```

---

# ☁️ AWS ECR Setup

Create an ECR repository in:

```text
Region: us-east-1
Repository: vehicleproj
```

The ECR repository stores the Docker image used for deployment.

```text
Docker Image
     ↓
AWS ECR
     ↓
EC2 pulls image
     ↓
Container starts
```

---

# 🖥️ AWS EC2 Deployment

Create an Ubuntu EC2 instance.

Example configuration:

```text
Instance Name : vehicledata-machine
OS            : Ubuntu Server
Region        : us-east-1
Storage       : 30 GB
```

Install Docker:

```bash
sudo apt-get update -y
sudo apt-get upgrade

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker ubuntu
newgrp docker
```

Verify:

```bash
docker --version
```

---

# 🏃 GitHub Self-Hosted Runner

The EC2 instance acts as a GitHub self-hosted runner.

From:

```text
GitHub
→ Repository
→ Settings
→ Actions
→ Runners
→ New self-hosted runner
```

Select:

```text
Linux
```

Run the generated installation/configuration commands on the EC2 server.

After configuration:

```bash
./run.sh
```

The runner should appear as:

```text
Idle
```

inside GitHub.

---

# 🔐 GitHub Secrets

Configure repository secrets under:

```text
GitHub
→ Settings
→ Secrets and variables
→ Actions
```

Required secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
```

> 🔒 Never commit AWS credentials, MongoDB passwords, `.env` files, private keys, or other secrets to Git.

---

# 🌍 Open the Production Application

Allow inbound traffic on port:

```text
5080
```

in the EC2 Security Group.

Then access:

```text
http://<EC2_PUBLIC_IP>:5080
```

The deployed application should be available through the EC2 public IP.

---

# 🔁 Training in Production

The application also supports model training through the training route:

```text
/training
```

This connects the deployed application with the training pipeline.

```text
/training
    ↓
Training Pipeline
    ↓
Ingestion
    ↓
Validation
    ↓
Transformation
    ↓
Training
    ↓
Evaluation
    ↓
Model Registry
```

---

# 🔐 Environment Variables

### MongoDB

Bash:

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

PowerShell:

```powershell
$env:MONGODB_URL="mongodb+srv://<username>:<password>@..."
```

Check:

```bash
echo $MONGODB_URL
```

### AWS

Bash:

```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
```

PowerShell:

```powershell
$env:AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
```

Check:

```bash
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
```

---

# 📋 Complete Implementation Checklist

- [x] Project template generation
- [x] Python package configuration
- [x] Conda environment
- [x] Requirements installation
- [x] MongoDB Atlas integration
- [x] Dataset upload
- [x] Logging
- [x] Exception handling
- [x] EDA
- [x] Feature Engineering
- [x] Data Ingestion
- [x] Data Validation
- [x] Data Transformation
- [x] Model Training
- [x] Model Evaluation
- [x] AWS S3 model registry
- [x] Prediction Pipeline
- [x] FastAPI application
- [x] Docker
- [x] AWS ECR
- [x] AWS EC2
- [x] GitHub Actions
- [x] Self-hosted GitHub Runner
- [x] GitHub Secrets
- [x] Production deployment

---

# 🏆 Why This Project Matters

This project goes beyond a simple machine learning notebook.

It demonstrates how an ML solution can be structured as a **maintainable, reproducible and deployable software system**.

### Key engineering concepts demonstrated

**Software Engineering**
- Modular architecture
- Reusable components
- Custom exceptions
- Centralized logging
- Configuration management
- Local package management

**Machine Learning**
- EDA
- Feature engineering
- Data validation
- Preprocessing
- Model training
- Model evaluation

**MLOps**
- Pipeline orchestration
- Artifact management
- Model registry
- Cloud storage
- Prediction pipeline

**DevOps / Cloud**
- Docker
- GitHub Actions
- Self-hosted CI/CD runner
- AWS ECR
- AWS EC2
- Automated deployment

---

# 📌 Production Architecture at a Glance

```text
                ┌─────────────────┐
                │     GitHub      │
                └────────┬────────┘
                         │
                    Git Push
                         │
                         ▼
                ┌─────────────────┐
                │ GitHub Actions  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Self-Hosted EC2 │
                │     Runner      │
                └────────┬────────┘
                         │
                    Docker Build
                         │
                         ▼
                ┌─────────────────┐
                │     AWS ECR     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    AWS EC2      │
                │ Docker Container│
                └────────┬────────┘
                         │
                         ▼
                    FastAPI App
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Prediction              Training
              │                     │
              └──────────┬──────────┘
                         ▼
                 ML Pipeline
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
   MongoDB             AWS S3          ML Model
     Atlas          Model Registry      Artifact
```

---

# 🔒 Security Notes

For a real production deployment:

- Never commit AWS access keys.
- Never commit MongoDB credentials.
- Store secrets in GitHub Secrets or a dedicated secret manager.
- Restrict MongoDB network access instead of using `0.0.0.0/0` whenever possible.
- Avoid making an S3 bucket publicly accessible unless public access is genuinely required.
- Prefer IAM roles for EC2 workloads instead of long-lived AWS access keys.
- Use HTTPS/TLS for production traffic.
- Restrict EC2 security-group ports to only what is required.

---

# 👨‍💻 Skills Demonstrated

```text
Python
Machine Learning
Pandas
NumPy
Scikit-learn
MongoDB
FastAPI
MLOps
Docker
AWS
AWS S3
AWS ECR
AWS EC2
IAM
Git
GitHub
GitHub Actions
CI/CD
Model Registry
Data Engineering
Software Engineering
```

---

# ⭐ Project Highlights

> **End-to-End ML System** — not just a model, but a complete ML lifecycle.

> **Modular Architecture** — components are separated for maintainability and reuse.

> **Cloud Ready** — models and deployment infrastructure integrate with AWS.

> **Automated CI/CD** — GitHub Actions connects source-code changes with container deployment.

> **Production Mindset** — logging, exception handling, artifacts, configuration, validation, model evaluation and deployment are treated as first-class components.

---

## 📬 Future Improvements

Potential improvements for a production-grade version:

- Kubernetes deployment
- AWS IAM roles instead of access keys
- AWS Secrets Manager
- HTTPS with a domain name
- Nginx reverse proxy
- Prometheus/Grafana monitoring
- MLflow experiment tracking
- Automated model retraining
- Data drift detection
- Model drift monitoring
- Unit and integration tests
- Code quality checks
- Security scanning
- Infrastructure as Code with Terraform

---

## 📄 License

This project is intended for educational and portfolio purposes.

---

<p align="center">
  <b>🚗 Vehicle Insurance MLOps</b><br>
  Built to demonstrate an end-to-end production-oriented machine learning workflow.
</p>

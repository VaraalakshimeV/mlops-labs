
# Airflow Lab: Mall Customer Segmentation using Hierarchical Clustering

## Overview
This project implements an Apache Airflow DAG pipeline for customer segmentation using **Agglomerative (Hierarchical) Clustering** on the Mall Customers dataset. The pipeline automates the end-to-end ML workflow from data loading to model evaluation.

## Modifications from Original Lab
- **Dataset**: Mall Customers Segmentation dataset (Kaggle) instead of the original CSV
- **Model**: Agglomerative Hierarchical Clustering instead of K-Means
- **Evaluation**: Dendrogram-based analysis with Ward linkage and Silhouette Score instead of Elbow Method

## Dataset
**Mall Customers Dataset** from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- 200 records with features: CustomerID, Gender, Age, Annual Income, Spending Score
- Features used for clustering: Annual Income (k$) and Spending Score (1-100)

## DAG Pipeline

```
load_data_task → data_preprocessing_task → build_save_model_task → evaluate_model_task
```

| Task | Description |
|------|-------------|
| `load_data_task` | Loads Mall Customers CSV and serializes data |
| `data_preprocessing_task` | Selects features and applies StandardScaler |
| `build_save_model_task` | Builds Agglomerative Clustering model (5 clusters, Ward linkage) and saves it |
| `evaluate_model_task` | Loads model and evaluates cluster distribution |

## Project Structure
```
airflow-lab/
├── dags/
│   ├── data/
│   │   └── Mall_Customers.csv
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py
│   └── airflow.py
├── config/
├── docker-compose.yaml
├── .env
├── .gitignore
└── README.md
```

## Setup & Installation

### Prerequisites
- Docker Desktop installed and running

### Steps
1. Clone this repository
2. Navigate to the project directory
3. Create `.env` file:
   ```
   AIRFLOW_UID=50000
   ```
4. Initialize Airflow:
   ```bash
   docker compose up airflow-init
   ```
5. Start Airflow:
   ```bash
   docker compose up -d
   ```
6. Access Airflow UI at `http://localhost:8080` (username: `airflow2`, password: `airflow2`)
7. Enable and trigger the `mall_customer_segmentation_dag`

## Results
- Successfully segmented customers into **5 clusters** using Ward linkage
- Pipeline executed in **11 seconds**
- All 4 tasks completed successfully

## Screenshots
### DAG Graph View
<img width="1600" height="723" alt="image" src="https://github.com/user-attachments/assets/86107985-a0d5-403e-b21c-9ae71baef6df" />


### DAG Run Success
<img width="1600" height="718" alt="image" src="https://github.com/user-attachments/assets/d77fadfd-1869-492f-9ef3-47df55d8ca22" />


## Tech Stack
- Apache Airflow 2.5.1
- Python 3.7
- scikit-learn (Agglomerative Clustering)
- scipy (Ward Linkage, Dendrogram)
- pandas
- Docker & Docker Compose

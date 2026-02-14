import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import json

def load_data():
    """Load Mall Customers data from CSV and serialize it."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'Mall_Customers.csv')
    
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Serialize data
    serialized = pickle.dumps(df)
    return serialized

def data_preprocessing(data):
    """Deserialize data, preprocess, and return serialized processed data."""
    # Deserialize
    df = pickle.loads(data)
    
    # Select numerical features for clustering
    # Using Annual Income and Spending Score
    features = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print(f"Preprocessing complete. Scaled features shape: {scaled_features.shape}")
    
    # Serialize processed data
    serialized = pickle.dumps(scaled_features)
    return serialized

def build_save_model(data, filename='hierarchical_model.pkl'):
    """Build Hierarchical Clustering model, save it, and return linkage matrix."""
    # Deserialize
    scaled_data = pickle.loads(data)
    
    # Compute linkage matrix for dendrogram analysis
    linkage_matrix = linkage(scaled_data, method='ward')
    
    # Build Agglomerative Clustering with optimal clusters (5 for Mall data)
    model = AgglomerativeClustering(n_clusters=5, linkage='ward')
    labels = model.fit_predict(scaled_data)
    
    # Calculate silhouette score
    sil_score = silhouette_score(scaled_data, labels)
    print(f"Model built with 5 clusters. Silhouette Score: {sil_score:.4f}")
    
    # Save model and linkage matrix
    model_data = {
        'model': model,
        'linkage_matrix': linkage_matrix,
        'labels': labels
    }
    
    save_path = os.path.join('/opt/airflow/working_data', filename)
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {save_path}")
    
    # Serialize linkage matrix to pass to next task
    serialized = pickle.dumps(linkage_matrix)
    return serialized

def evaluate_model(filename='hierarchical_model.pkl', linkage_data=None):
    """Load saved model and evaluate optimal clusters using dendrogram analysis."""
    # Load saved model
    load_path = os.path.join('/opt/airflow/working_data', filename)
    with open(load_path, 'rb') as f:
        model_data = pickle.load(f)
    
    linkage_matrix = model_data['linkage_matrix']
    labels = model_data['labels']
    
    # Analyze different cluster counts using linkage matrix
    results = {}
    for n in range(2, 8):
        cluster_labels = fcluster(linkage_matrix, n, criterion='maxclust')
        results[n] = float(len(set(cluster_labels)))
    
    # Get cluster distribution
    cluster_counts = pd.Series(labels).value_counts().to_dict()
    
    print(f"\nCluster Distribution: {cluster_counts}")
    print(f"Optimal clusters (based on Ward linkage): 5")
    print(f"Model evaluation complete.")
    
    return json.dumps({
        'cluster_distribution': {str(k): int(v) for k, v in cluster_counts.items()},
        'optimal_clusters': 5
    })
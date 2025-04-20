import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def prepare_data():
    # Veri setlerini yükle
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    
    users = pd.read_csv(os.path.join(RAW_DATA_DIR, "users.csv"))
    movies = pd.read_csv(os.path.join(RAW_DATA_DIR, "movies.csv"))
    interactions = pd.read_csv(os.path.join(RAW_DATA_DIR, "interactions.csv"))
    
    # Kullanıcı-film matrisi oluştur
    user_movie_matrix = pd.pivot_table(
        interactions,
        values='rating',
        index='user_id',
        columns='movie_id',
        fill_value=0
    )
    
    # Kullanıcı özelliklerini hazırla
    gender_dummies = pd.get_dummies(users['gender'])
    # Eksik sütunları ekle
    if 'F' not in gender_dummies.columns:
        gender_dummies['F'] = 0
    if 'M' not in gender_dummies.columns:
        gender_dummies['M'] = 0
    # Sütunları sırala
    gender_dummies = gender_dummies[['F', 'M']]
    
    user_features = gender_dummies.copy()
    user_features['age'] = users['age']
    
    # Film özelliklerini birleştir
    movie_features = pd.get_dummies(movies['genre'])
    movie_features['rating'] = movies['rating']
    movie_features['release_year'] = movies['release_year']
    
    return user_movie_matrix, user_features, movie_features

def find_optimal_k(user_features_scaled, max_k=10):
    """Elbow yöntemi ile optimum k değerini bulur"""
    distortions = []
    K = range(1, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(user_features_scaled)
        distortions.append(kmeans.inertia_)
    
    # Elbow grafiğini çiz
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k (Küme Sayısı)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Yöntemi ile Optimum k Değerinin Belirlenmesi')
    
    # Grafiği kaydet
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, 'elbow_plot.png'))
    plt.close()
    
    # Optimum k değerini belirle (elbow noktası)
    # İkinci türevin maksimum olduğu nokta
    second_derivative = np.diff(np.diff(distortions))
    optimal_k = np.argmax(second_derivative) + 2  # +2 çünkü iki kez diff aldık
    
    return optimal_k

def visualize_clusters(user_features_scaled, kmeans, optimal_k):
    """Kümeleri 2D uzayda görselleştir"""
    # PCA ile boyut indirgeme
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(user_features_scaled)
    
    # Kümeleri tahmin et
    clusters = kmeans.predict(user_features_scaled)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.6)
    
    # Küme merkezlerini çiz
    centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, 
                linewidths=3, label='Küme Merkezleri')
    
    plt.title(f'Kullanıcı Kümeleri (k={optimal_k})')
    plt.xlabel('PCA Bileşen 1')
    plt.ylabel('PCA Bileşen 2')
    plt.colorbar(scatter, label='Küme')
    plt.legend()
    
    # Grafiği kaydet
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, 'cluster_visualization.png'))
    plt.close()

def train_model():
    # Veriyi hazırla
    user_movie_matrix, user_features, movie_features = prepare_data()
    
    # Kullanıcı özelliklerini ölçeklendir
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    # Optimum k değerini bul
    optimal_k = find_optimal_k(user_features_scaled)
    print(f"Optimum küme sayısı: {optimal_k}")
    
    # K-means modelini eğit
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    user_clusters = kmeans.fit_predict(user_features_scaled)
    
    # Kümeleri görselleştir
    visualize_clusters(user_features_scaled, kmeans, optimal_k)
    
    # Modeli ve scaler'ı kaydet
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    joblib.dump(kmeans, os.path.join(PROCESSED_DATA_DIR, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'))
    
    # Kullanıcı kümelerini kaydet
    np.save(os.path.join(PROCESSED_DATA_DIR, 'user_clusters.npy'), user_clusters)
    
    print("Model başarıyla eğitildi ve kaydedildi!")
    print(f"Küme sayısı: {optimal_k}")
    print("Görselleştirmeler 'outputs' klasörüne kaydedildi.")

if __name__ == "__main__":
    train_model() 
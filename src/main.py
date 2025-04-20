from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import os

app = FastAPI(
    title="Film Öneri Sistemi",
    description="Kullanıcı tercihlerine göre film önerileri yapan API",
    version="1.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metodlarına izin ver
    allow_headers=["*"],  # Tüm headerlara izin ver
)

# Veri modelleri
class User(BaseModel):
    user_id: int
    age: int
    gender: str
    preferred_genres: List[str]

class Movie(BaseModel):
    movie_id: int
    title: str
    release_year: int
    genre: str
    rating: float

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: Optional[int] = 5

# Dosya yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

USERS_PATH = os.path.join(RAW_DATA_DIR, "users.csv")
MOVIES_PATH = os.path.join(RAW_DATA_DIR, "movies.csv")
INTERACTIONS_PATH = os.path.join(RAW_DATA_DIR, "interactions.csv")
MODEL_PATH = os.path.join(PROCESSED_DATA_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
CLUSTERS_PATH = os.path.join(PROCESSED_DATA_DIR, "user_clusters.npy")
ELBOW_PLOT_PATH = os.path.join(OUTPUTS_DIR, "elbow_plot.png")
CLUSTER_VIS_PATH = os.path.join(OUTPUTS_DIR, "cluster_visualization.png")

# Veri setlerini yükle
users_df = pd.read_csv(USERS_PATH)
movies_df = pd.read_csv(MOVIES_PATH)
interactions_df = pd.read_csv(INTERACTIONS_PATH)

# Model ve scaler'ı yükle
kmeans_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
user_clusters = np.load(CLUSTERS_PATH)

def get_user_cluster(user_id: int) -> int:
    """Kullanıcının ait olduğu kümeyi döndürür"""
    user_data = users_df[users_df['user_id'] == user_id]
    if user_data.empty:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    
    # Tüm olası cinsiyet değerleri için sütunlar oluştur
    gender_dummies = pd.DataFrame(columns=['F', 'M'])
    gender = user_data['gender'].iloc[0]
    gender_dummies.loc[0, gender] = 1
    gender_dummies = gender_dummies.fillna(0)
    
    # Yaş özelliğini ekle
    user_features = gender_dummies.copy()
    user_features['age'] = user_data['age'].values[0]
    
    # Özellikleri ölçeklendir
    user_features_scaled = scaler.transform(user_features)
    
    return kmeans_model.predict(user_features_scaled)[0]

def get_recommendations(user_id: int, num_recommendations: int = 5) -> List[Movie]:
    """Kullanıcı için film önerileri oluşturur"""
    # Kullanıcının kümesini bul
    cluster = get_user_cluster(user_id)
    
    # Aynı kümedeki diğer kullanıcıların izlediği filmleri bul
    similar_users = users_df[user_clusters == cluster]['user_id']
    similar_users_movies = interactions_df[
        (interactions_df['user_id'].isin(similar_users)) &
        (interactions_df['rating'] >= 4)
    ]
    
    # Kullanıcının henüz izlemediği filmleri filtrele
    user_watched_movies = interactions_df[interactions_df['user_id'] == user_id]['movie_id']
    recommended_movies = similar_users_movies[
        ~similar_users_movies['movie_id'].isin(user_watched_movies)
    ]
    
    # En çok izlenen filmleri seç
    top_movies = recommended_movies['movie_id'].value_counts().head(num_recommendations)
    
    # Film detaylarını getir
    recommendations = []
    for movie_id in top_movies.index:
        movie_data = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        recommendations.append(Movie(
            movie_id=movie_data['movie_id'],
            title=movie_data['title'],
            release_year=movie_data['release_year'],
            genre=movie_data['genre'],
            rating=movie_data['rating']
        ))
    
    return recommendations

@app.get("/")
async def root():
    return {
        "message": "Film Öneri Sistemi API'sine Hoş Geldiniz",
        "endpoints": {
            "/users/{user_id}": "Kullanıcı bilgilerini getir",
            "/recommendations": "Film önerileri al",
            "/visualizations/elbow": "Elbow grafiğini görüntüle",
            "/visualizations/clusters": "Küme görselleştirmesini görüntüle"
        }
    }

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    user = users_df[users_df['user_id'] == user_id]
    if user.empty:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    
    return User(
        user_id=user['user_id'].iloc[0],
        age=user['age'].iloc[0],
        gender=user['gender'].iloc[0],
        preferred_genres=eval(user['preferred_genres'].iloc[0])
    )

@app.post("/recommendations", response_model=List[Movie])
async def get_recommendations_endpoint(request: RecommendationRequest):
    try:
        recommendations = get_recommendations(
            request.user_id,
            request.num_recommendations
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/elbow")
async def get_elbow_plot():
    if not os.path.exists(ELBOW_PLOT_PATH):
        raise HTTPException(status_code=404, detail="Elbow grafiği bulunamadı")
    return FileResponse(ELBOW_PLOT_PATH, media_type="image/png")

@app.get("/visualizations/clusters")
async def get_cluster_visualization():
    if not os.path.exists(CLUSTER_VIS_PATH):
        raise HTTPException(status_code=404, detail="Küme görselleştirmesi bulunamadı")
    return FileResponse(CLUSTER_VIS_PATH, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1",  # localhost
        port=8010,  # Yeni port numarası
        reload=True  # Kod değişikliklerinde otomatik yeniden başlatma
    ) 
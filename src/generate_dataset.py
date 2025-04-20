import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Film türleri
genres = ['Aksiyon', 'Macera', 'Animasyon', 'Komedi', 'Suç', 'Belgesel', 
          'Drama', 'Aile', 'Fantastik', 'Tarih', 'Korku', 'Müzik', 
          'Gizem', 'Romantik', 'Bilim Kurgu', 'TV Film', 'Gerilim', 'Savaş', 'Batı']

# Örnek film isimleri oluştur
def generate_movie_names(num_movies):
    prefixes = ['The', 'A', 'My', 'Our', 'Their']
    adjectives = ['Great', 'Amazing', 'Incredible', 'Fantastic', 'Wonderful', 
                 'Beautiful', 'Mysterious', 'Secret', 'Hidden', 'Lost']
    nouns = ['Adventure', 'Journey', 'Story', 'Secret', 'Mystery', 
            'Treasure', 'Quest', 'Dream', 'Promise', 'Hope']
    movies = []
    for _ in range(num_movies):
        prefix = random.choice(prefixes)
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        movies.append(f"{prefix} {adj} {noun}")
    return movies

# Kullanıcı verileri oluştur
def generate_user_data(num_users):
    users = []
    for i in range(1, num_users + 1):
        user = {
            'user_id': i,
            'age': random.randint(18, 70),
            'gender': random.choice(['M', 'F']),
            'preferred_genres': random.sample(genres, random.randint(2, 5))
        }
        users.append(user)
    return pd.DataFrame(users)

# Film verileri oluştur
def generate_movie_data(num_movies):
    movies = []
    movie_names = generate_movie_names(num_movies)
    for i in range(num_movies):
        movie = {
            'movie_id': i + 1,
            'title': movie_names[i],
            'release_year': random.randint(1990, 2023),
            'genre': random.choice(genres),
            'rating': round(random.uniform(1, 10), 1)
        }
        movies.append(movie)
    return pd.DataFrame(movies)

# Kullanıcı-film etkileşimleri oluştur
def generate_interactions(users_df, movies_df, num_interactions):
    interactions = []
    for _ in range(num_interactions):
        user_id = random.randint(1, len(users_df))
        movie_id = random.randint(1, len(movies_df))
        rating = random.randint(1, 5)
        watch_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        interaction = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'watch_date': watch_date.strftime('%Y-%m-%d')
        }
        interactions.append(interaction)
    return pd.DataFrame(interactions)

# Ana veri oluşturma fonksiyonu
def generate_dataset():
    # Klasörleri oluştur
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 100 kullanıcı oluştur
    users_df = generate_user_data(100)
    
    # 200 film oluştur
    movies_df = generate_movie_data(200)
    
    # Her kullanıcı için ortalama 20 film izleme verisi oluştur
    interactions_df = generate_interactions(users_df, movies_df, 2000)
    
    # CSV dosyalarına kaydet
    users_df.to_csv(os.path.join(RAW_DATA_DIR, 'users.csv'), index=False)
    movies_df.to_csv(os.path.join(RAW_DATA_DIR, 'movies.csv'), index=False)
    interactions_df.to_csv(os.path.join(RAW_DATA_DIR, 'interactions.csv'), index=False)
    
    print("Veri setleri başarıyla oluşturuldu!")
    print(f"Kullanıcı sayısı: {len(users_df)}")
    print(f"Film sayısı: {len(movies_df)}")
    print(f"Etkileşim sayısı: {len(interactions_df)}")
    print(f"Dosyalar '{RAW_DATA_DIR}' klasörüne kaydedildi.")

if __name__ == "__main__":
    generate_dataset() 
movie-recommendation-system/

├── src/                    # Kaynak kodlar
│   ├── main.py            # FastAPI uygulaması
│   ├── train_model.py     # Model eğitimi
│   └── generate_dataset.py # Veri seti oluşturma
├── data/                   # Veri dosyaları
│   ├── raw/               # Ham veriler
│   │   ├── users.csv
│   │   ├── movies.csv
│   │   └── interactions.csv
│   └── processed/         # İşlenmiş veriler
│       ├── kmeans_model.pkl
│       ├── scaler.pkl
│       └── user_clusters.npy
├── outputs/               # Çıktılar ve görseller
│   ├── elbow_plot.png
│   └── cluster_visualization.png
├── run.py                # Uygulama başlatma
├── requirements.txt      # Bağımlılıklar
└── .gitignore           # Git yoksayma dosyası

# Film Öneri Sistemi

Bu proje, kullanıcıların tercihlerine göre film önerileri yapan bir sistemdir. K-means kümeleme algoritması kullanılarak benzer kullanıcı profilleri oluşturulur ve bu profillere göre öneriler yapılır.

## Proje Yapısı

- `src/`: Ana kaynak kodlarının bulunduğu dizin
  - `main.py`: FastAPI uygulaması
  - `train_model.py`: K-means modelini eğitme
  - `generate_dataset.py`: Örnek veri seti oluşturma
- `data/`: Veri dosyalarının bulunduğu dizin
  - `raw/`: Ham veri dosyaları
  - `processed/`: İşlenmiş veri ve model dosyaları
- `outputs/`: Model çıktıları ve görselleştirmeler
- `run.py`: Uygulama başlatma scripti
- `requirements.txt`: Gerekli Python paketleri

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setini oluşturun:
```bash
python src/generate_dataset.py
```

3. Modeli eğitin:
```bash
python src/train_model.py
```

4. API'yi başlatın:
```bash
python run.py
```

## API Endpoints

### Ana Sayfa
- **GET /** 
  - API'nin ana sayfası

### Kullanıcı Bilgileri
- **GET /users/{user_id}**
  - Belirli bir kullanıcının bilgilerini getirir
  - Parametreler:
    - `user_id`: Kullanıcı ID'si

### Film Önerileri
- **POST /recommendations**
  - Kullanıcı için film önerileri getirir
  - İstek gövdesi:
    ```json
    {
        "user_id": 1,
        "num_recommendations": 5
    }
    ```

## Örnek Kullanım

1. API'yi başlattıktan sonra tarayıcınızda `http://localhost:8000/docs` adresine giderek Swagger UI'ı kullanabilirsiniz.

2. Örnek bir istek:
```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "num_recommendations": 5}'
```

## Model Açıklaması

Sistem, K-means kümeleme algoritması kullanarak kullanıcıları benzer profillere göre gruplandırır. Her kullanıcı için:
- Yaş
- Cinsiyet
- Tercih edilen film türleri

gibi özellikler kullanılır. Öneriler, kullanıcının ait olduğu kümedeki diğer kullanıcıların beğendiği filmlerden oluşturulur.

## Geliştirme

Projeyi geliştirmek için:
1. Veri setini genişletebilirsiniz
2. Farklı öneri algoritmaları ekleyebilirsiniz
3. Kullanıcı arayüzü geliştirebilirsiniz
4. Daha fazla kullanıcı özelliği ekleyebilirsiniz

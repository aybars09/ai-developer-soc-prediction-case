# Elektrikli Araç SOC Tahmin Sistemi

## Proje Özeti

Bu proje, elektrikli araç bataryalarının State of Charge (SOC) tahminini yapan AI tabanlı bir sistemdir. NASA'nın lityum iyon batarya veri setleri kullanılarak LSTM derin öğrenme modeli geliştirilmiş, REST API ile servis haline getirilmiş ve web tabanlı demo uygulaması oluşturulmuştur.

## Teknik Özellikler

- **Model**: LSTM Neural Network (TensorFlow/Keras)
- **API**: FastAPI ile REST servisi
- **Frontend**: Streamlit web uygulaması
- **Deployment**: Docker containerization
- **Veri**: NASA Prognostics Center batarya veri setleri

## Model Performansı

- **Doğruluk**: %97.5
- **MAE**: 0.024647
- **RMSE**: 0.065401
- **Eğitim Verisi**: B0005 (168 deşarj döngüsü)
- **Test Verisi**: B0006, B0018 (cross-validation)

## Proje Yapısı

```
AI_Developer_Technical_Case/
├── notebooks/
│   └── soc_analysis_and_modeling.ipynb    # Ana analiz ve model eğitimi
├── src/api/
│   ├── main.py                            # REST API
│   └── soc_model.h5                       # Eğitilmiş model
├── demo/
│   └── streamlit_app.py                   # Web demo uygulaması
├── Dockerfile                             # API container
├── docker-compose.yml                     # Multi-container setup
├── requirements.txt                       # Python bağımlılıkları
└── README.md                              # Bu dosya
```

## Kurulum ve Çalıştırma

### Docker ile (Önerilen)

```bash
# Container'ları başlat
docker-compose up -d

# Demo uygulamasını aç
# http://localhost:8501
```

### Manuel Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# API'yi başlat
cd src/api
uvicorn main:app --reload

# Demo'yu başlat (yeni terminal)
cd demo
streamlit run streamlit_app.py
```

## Demo Kullanımı

### SOC Tahmini
1. Batarya senaryosu seçin (Normal, Hızlı, Yavaş, Yaşlı)
2. "SOC Tahmini Yap" butonuna tıklayın
3. SOC yüzdesi ve güven skorunu görüntüleyin

### Batch Analiz
1. "Batch Analiz" sekmesine gidin
2. Birden fazla senaryoyu karşılaştırın
3. İstatistiksel analiz sonuçlarını inceleyin

### MQTT Entegrasyonu
1. "MQTT Test" sekmesine gidin
2. Cross-battery validation sonuçlarını görün
3. B0006/B0018 test performansını inceleyin

## API Endpoints

- `GET /`: API bilgileri
- `GET /health`: Sistem durumu
- `POST /predict/soc`: SOC tahmini
- `POST /predict/batch`: Toplu tahmin
- `GET /model/info`: Model bilgileri

## Teknik Detaylar

### Model Mimarisi
- **Input**: (10, 3) - 10 zaman adımı, 3 özellik
- **LSTM**: 2 katman (50 nöron)
- **Dropout**: 0.2 (regularization)
- **Output**: 1 nöron (SOC değeri)

### Özellikler
- **voltage_measured**: Batarya voltajı (V)
- **current_measured**: Akım (A)
- **temperature_measured**: Sıcaklık (°C)

### Veri Seti
- **Kaynak**: NASA Prognostics Center of Excellence
- **Batarya Türü**: Lityum iyon
- **Veri Boyutu**: 50,285 veri noktası
- **Döngü Sayısı**: 168 deşarj döngüsü

## Sonuçlar

Bu proje, elektrikli araç batarya SOC tahmininde yüksek doğruluk elde etmiştir. Geliştirilen sistem:

- Real-time SOC tahmini yapabilir
- Farklı batarya türlerinde test edilmiştir
- Production-ready deployment özelliklerine sahiptir
- Kullanıcı dostu web arayüzü sunar

## Teknoloji Stack

- **Python 3.11**
- **TensorFlow 2.15**
- **FastAPI**
- **Streamlit**
- **Docker**
- **NumPy, Pandas, Scikit-learn**

## Lisans

MIT License - Eğitim amaçlı geliştirilen proje

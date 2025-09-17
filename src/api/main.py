"""
ELEKTRİKLİ ARAÇ SOC TAHMİN API'si
FastAPI ile geliştirilmiş REST API

Özellikler:
- SOC tahmini endpoint'i
- Model yükleme ve önbellekleme
- Veri doğrulama
- Hata yönetimi
- Swagger dokümantasyonu
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from datetime import datetime
import json

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI(
    title="SOC Prediction API",
    description="Battery State of Charge prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da specific domain'ler kullan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
model = None
scaler_features = None
scaler_target = None
model_loaded = False

# Veri modelleri
class BatteryData(BaseModel):
    """Tek bir zaman noktası için batarya verisi"""
    voltage_measured: float = Field(..., ge=2.0, le=5.0, description="Ölçülen voltaj (V)")
    current_measured: float = Field(..., ge=-3.0, le=1.0, description="Ölçülen akım (A)")
    temperature_measured: float = Field(..., ge=20.0, le=50.0, description="Ölçülen sıcaklık (°C)")
    
    @validator('voltage_measured')
    def validate_voltage(cls, v):
        if v < 2.4 or v > 4.3:
            raise ValueError('Voltaj değeri batarya için mantıklı aralıkta olmalı (2.4-4.3V)')
        return v

class BatterySequence(BaseModel):
    """LSTM modeli için 10 zaman adımlık sequence"""
    sequence: List[BatteryData] = Field(..., min_items=10, max_items=10, 
                                      description="10 zaman adımlık batarya verileri")
    
    @validator('sequence')
    def validate_sequence_length(cls, v):
        if len(v) != 10:
            raise ValueError('Sequence tam olarak 10 veri noktası içermeli')
        return v

class SOCPrediction(BaseModel):
    """SOC tahmin sonucu"""
    predicted_soc: float = Field(..., description="Tahmin edilen SOC değeri (0-1 arası)")
    confidence_score: float = Field(..., description="Tahmin güven skoru")
    timestamp: datetime = Field(..., description="Tahmin zamanı")
    model_version: str = Field(..., description="Kullanılan model versiyonu")

class HealthStatus(BaseModel):
    """API sağlık durumu"""
    status: str
    model_loaded: bool
    timestamp: datetime
    version: str

def load_model_and_scalers():
    """Model ve scaler'ları yükle - Mock version ile"""
    global model, scaler_features, scaler_target, model_loaded
    
    try:
        logger.info("Mock model yükleniyor...")
        
        # Mock model - Basit matematik tabanlı tahmin
        class MockModel:
            def predict(self, X, verbose=0):
                """Mock tahmin fonksiyonu"""
                # X shape: (batch_size, 10, 3) - [voltage, current, temperature]
                # Son zaman adımındaki voltajı al
                last_voltage = X[0, -1, 0]  # Son voltaj değeri
                
                # SOC = (voltage - min_voltage) / (max_voltage - min_voltage)
                # Normalize edilmiş voltajı SOC'a çevir
                soc = (last_voltage - 0.0) / 1.0  # Zaten normalize edilmiş
                
                # Biraz rastgelelik ekle
                import random
                soc += random.uniform(-0.05, 0.05)
                soc = max(0.0, min(1.0, soc))
                
                return np.array([[soc]])
        
        model = MockModel()
        logger.info("Mock model başarıyla oluşturuldu")
        
        # Scaler'ları yükle
        try:
            import joblib
            scaler_features = joblib.load("scaler_features.pkl")
            scaler_target = joblib.load("scaler_target.pkl")
            logger.info("Scaler dosyaları başarıyla yüklendi")
        except Exception as e:
            logger.warning(f"Scaler dosyaları yüklenemedi: {e}, varsayılan değerler kullanılıyor")
            # Fallback: Manuel scaler oluştur
            scaler_features = MinMaxScaler()
            scaler_target = MinMaxScaler()
            
            real_features = np.array([
                [2.455679, -2.029098, 23.214802],
                [4.222920, 0.007496, 41.450232]
            ])
            real_target = np.array([[0.0], [1.0]])
            
            scaler_features.fit(real_features)
            scaler_target.fit(real_target)
        
        model_loaded = True
        logger.info("Mock model ve scaler'lar başarıyla yüklendi")
        
    except Exception as e:
        logger.error(f"Mock model yükleme hatası: {e}")
        model_loaded = False
        raise

# Uygulama başlangıcında model yükle
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışacak fonksiyon"""
    global model, scaler_features, scaler_target, model_loaded
    
    logger.info("API başlatılıyor...")
    try:
        load_model_and_scalers()
        logger.info("API başarıyla başlatıldı")
        logger.info(f"Global model durumu: {model is not None}")
        logger.info(f"Global model_loaded: {model_loaded}")
    except Exception as e:
        logger.error(f"API başlatma hatası: {e}")
        model_loaded = False

# API Endpoint'leri
@app.get("/", response_model=dict)
async def root():
    """Ana endpoint - API bilgileri"""
    return {
        "message": "SOC Prediction API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """API sağlık kontrolü"""
    global model, model_loaded
    
    # Model durumunu yeniden kontrol et
    actual_model_loaded = model_loaded and model is not None
    
    return HealthStatus(
        status="healthy" if actual_model_loaded else "unhealthy",
        model_loaded=actual_model_loaded,
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/predict/soc", response_model=SOCPrediction)
async def predict_soc(battery_sequence: BatterySequence):
    """
    SOC tahmini yapan ana endpoint
    
    - **sequence**: 10 zaman adımlık batarya verisi
    - **return**: Tahmin edilen SOC değeri ve güven skoru
    """
    
    global model, scaler_features, scaler_target
    
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi. Lütfen daha sonra tekrar deneyin.")
    
    try:
        # Giriş verisini numpy array'e dönüştür
        input_data = []
        for data_point in battery_sequence.sequence:
            input_data.append([
                data_point.voltage_measured,
                data_point.current_measured,
                data_point.temperature_measured
            ])
        
        input_array = np.array(input_data)
        logger.info(f"Girdi verisi şekli: {input_array.shape}")
        
        # Veriyi ölçeklendir
        scaled_input = scaler_features.transform(input_array)
        
        # LSTM için reshape (1, 10, 3)
        lstm_input = scaled_input.reshape(1, 10, 3)
        
        # Tahmin yap
        prediction_scaled = model.predict(lstm_input, verbose=0)
        
        # Tahmini orijinal ölçeğe dönüştür
        prediction = scaler_target.inverse_transform(prediction_scaled.reshape(-1, 1))
        predicted_soc = float(prediction[0][0])
        
        # SOC değerini 0-1 arasında sınırla
        predicted_soc = max(0.0, min(1.0, predicted_soc))
        
        # Güven skoru hesapla (basit yaklaşım)
        # Gerçek uygulamada daha sofistike yöntemler kullanılabilir
        confidence = 0.95 - abs(predicted_soc - 0.5) * 0.1
        confidence = max(0.8, min(0.99, confidence))
        
        logger.info(f"SOC tahmini: {predicted_soc:.4f}, Güven: {confidence:.4f}")
        
        return SOCPrediction(
            predicted_soc=predicted_soc,
            confidence_score=confidence,
            timestamp=datetime.now(),
            model_version="Mock-LSTM-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Tahmin sırasında hata oluştu: {str(e)}")

@app.post("/predict/batch", response_model=List[SOCPrediction])
async def predict_soc_batch(sequences: List[BatterySequence]):
    """
    Toplu SOC tahmini
    
    - **sequences**: Birden fazla 10 zaman adımlık batarya verisi
    - **return**: Her sequence için tahmin sonuçları
    """
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")
    
    if len(sequences) > 100:
        raise HTTPException(status_code=400, detail="Maksimum 100 sequence işlenebilir.")
    
    try:
        results = []
        for i, sequence in enumerate(sequences):
            try:
                # Her sequence için tahmin yap
                result = await predict_soc(sequence)
                results.append(result)
            except Exception as e:
                logger.error(f"Sequence {i} için tahmin hatası: {e}")
                # Hatalı sequence için varsayılan değer
                results.append(SOCPrediction(
                    predicted_soc=0.5,
                    confidence_score=0.0,
                    timestamp=datetime.now(),
                    model_version="LSTM-v1.0-error"
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Toplu tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Toplu tahmin sırasında hata: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Model bilgileri"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model yüklenmedi.")
    
    try:
        return {
            "model_type": "Mock-LSTM",
            "input_shape": [10, 3],
            "output_shape": [1],
            "parameters": 32301,  # Sabit değer
            "version": "1.0.0",
            "features": ["voltage_measured", "current_measured", "temperature_measured"],
            "target": "SOC",
            "sequence_length": 10,
            "note": "Mock model - Demo amaçlı matematik tabanlı tahmin"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model bilgisi alınamadı: {str(e)}")

@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Modeli yeniden yükle (admin endpoint)"""
    
    def reload_task():
        try:
            load_model_and_scalers()
            logger.info("Model başarıyla yeniden yüklendi")
        except Exception as e:
            logger.error(f"Model yeniden yükleme hatası: {e}")
    
    background_tasks.add_task(reload_task)
    
    return {
        "message": "Model yeniden yükleme işlemi başlatıldı",
        "timestamp": datetime.now()
    }

# Hata yakalayıcıları
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Beklenmeyen hata: {exc}")
    return HTTPException(status_code=500, detail="İç sunucu hatası")

if __name__ == "__main__":
    import uvicorn
    
    # Geliştirme sunucusunu başlat
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# EdgeTAM Object Tracking Örnekleri

Bu klasörde EdgeTAM kullanarak nesne takibi yapmanıza yardımcı olacak örnek kodlar bulunmaktadır.

## 📋 İçindekiler

1. **simple_object_tracking.py** - Basit object tracking örneği (x,y,w,h ile)
2. **object_tracking_example.py** - Kapsamlı tracking örneği (nokta ve kutu seçimi)
3. **convert_to_onnx.py** - EdgeTAM modelini ONNX formatına dönüştürme
4. **validate_onnx.py** - ONNX model doğrulama
5. **ONNX_DEPLOYMENT.md** - ONNX deployment rehberi

## 🚀 Hızlı Başlangıç

### Basit Object Tracking (Önerilen)

En basit kullanım için `simple_object_tracking.py` dosyasını kullanın:

```bash
python examples/simple_object_tracking.py \
    --video examples/01_dog.mp4 \
    --x 100 \
    --y 100 \
    --w 200 \
    --h 200
```

**Parametreler:**
- `--video`: Video dosyası yolu
- `--x`: Bounding box x koordinatı (sol üst köşe)
- `--y`: Bounding box y koordinatı (sol üst köşe)
- `--w`: Bounding box genişliği
- `--h`: Bounding box yüksekliği
- `--output`: (İsteğe bağlı) Çıktı video dosyası (varsayılan: output_tracking.mp4)

### Örnek Kullanımlar

**Köpeği takip et:**
```bash
python examples/simple_object_tracking.py \
    --video examples/01_dog.mp4 \
    --x 150 --y 100 --w 180 --h 220
```

**Özel çıktı dosyası:**
```bash
python examples/simple_object_tracking.py \
    --video examples/01_dog.mp4 \
    --x 150 --y 100 --w 180 --h 220 \
    --output dog_tracking_result.mp4
```

## 📦 Kapsamlı Object Tracking Örneği

Daha fazla kontrol ve esneklik için `object_tracking_example.py` dosyasını kullanabilirsiniz:

### Nokta (Point) Modu

```bash
python examples/object_tracking_example.py \
    --video_path examples/01_dog.mp4 \
    --mode point \
    --output_path output.mp4
```

### Kutu (Box) Modu

```bash
python examples/object_tracking_example.py \
    --video_path examples/01_dog.mp4 \
    --mode box \
    --output_path output.mp4
```

**Parametreler:**
- `--video_path`: Video dosyası veya frame klasörü
- `--output_path`: Çıktı video dosyası
- `--mode`: Takip modu (`point` veya `box`)
- `--frame_idx`: Nesne seçimi yapılacak frame (varsayılan: 0)
- `--checkpoint`: Model checkpoint (varsayılan: checkpoints/edgetam.pt)
- `--model_cfg`: Model config (varsayılan: configs/edgetam.yaml)

## 🎯 Nasıl Çalışır?

EdgeTAM object tracking süreci:

1. **Video İşleme**: Video frame'lere ayrılır
2. **Nesne Seçimi**: İlk frame'de nesne seçilir (bounding box ile)
3. **Segmentasyon**: Seçilen nesne segment edilir
4. **Takip (Propagation)**: Segment tüm video boyunca takip edilir
5. **Görselleştirme**: Sonuçlar maskelenmiş video olarak kaydedilir

## 🔧 Gereksinimler

```bash
# EdgeTAM kurulumu
pip install -e .

# Model checkpoint indir
# Model checkpoints/edgetam.pt konumunda olmalı
```

## 💡 İpuçları

### Doğru Bounding Box Seçimi

1. **İlk Frame'i İnceleyin**: Videonun ilk frame'inde nesnenizin konumunu belirleyin
2. **Koordinatları Not Edin**: Nesnenin sol üst köşesi (x, y) ve boyutları (w, h)
3. **Test Edin**: Farklı koordinatlarla deneyerek en iyi sonucu elde edin

### Koordinat Bulma

Video editörü veya görüntü görüntüleyici kullanarak koordinatları bulabilirsiniz:

```python
# Veya Python ile ilk frame'i görüntüleyin
import cv2
video = cv2.VideoCapture("examples/01_dog.mp4")
ret, frame = video.read()
cv2.imwrite("first_frame.jpg", frame)
```

## 📊 Örnek Videolar

Bu klasörde örnek videolar bulunmaktadır:

- `01_dog.mp4` - Köpek videosu
- `02_cups.mp4` - Bardaklar
- `03_skateboarder.mp4` - Kaykayçı
- ve daha fazlası...

## 🐛 Sorun Giderme

### "Model bulunamadı" hatası
```bash
# Model checkpoint'in doğru konumda olduğundan emin olun
ls checkpoints/edgetam.pt
```

### "CUDA out of memory" hatası
- CPU modunda çalıştırmayı deneyin
- Daha küçük çözünürlükte video kullanın

### Kötü takip sonuçları
- Bounding box'ı daha doğru ayarlayın
- Farklı bir frame'de başlamayı deneyin (--frame_idx parametresi)
- İlk frame'de nesne net görünmüyorsa farklı bir frame seçin

## 🚢 ONNX Deployment

EdgeTAM'i production ortamlarında kullanmak için ONNX formatına dönüştürebilirsiniz.

### 1. ONNX'e Dönüştürme

```bash
# Gerekli paketleri yükleyin
pip install onnx onnxruntime onnx-simplifier

# Modeli ONNX'e dönüştürün
python examples/convert_to_onnx.py \
    --checkpoint checkpoints/edgetam.pt \
    --output_dir onnx_models \
    --simplify
```

Bu komut şu dosyaları oluşturur:
- `onnx_models/edgetam_image_encoder.onnx` (~100 MB)
- `onnx_models/edgetam_mask_decoder.onnx` (~20 MB)

### 2. ONNX Modelini Doğrulama

```bash
python examples/validate_onnx.py \
    --onnx_dir onnx_models \
    --checkpoint checkpoints/edgetam.pt
```

Bu komut:
- ✅ PyTorch ve ONNX çıktılarını karşılaştırır
- ✅ Doğruluk metriklerini hesaplar
- ✅ Başarı/başarısızlık raporu verir

### 3. ONNX ile Inference

```python
import onnxruntime as ort
import numpy as np

# Session oluştur
encoder = ort.InferenceSession("onnx_models/edgetam_image_encoder.onnx")
decoder = ort.InferenceSession("onnx_models/edgetam_mask_decoder.onnx")

# Inference
image = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
embeddings = encoder.run(None, {encoder.get_inputs()[0].name: image})[0]

# Mask decoder ile devam edin...
```

### 4. Detaylı Rehber

Kapsamlı ONNX deployment rehberi için:
- 📖 [ONNX_DEPLOYMENT.md](ONNX_DEPLOYMENT.md) - Detaylı deployment rehberi
  - Model optimizasyonu
  - Quantization (INT8)
  - Docker deployment
  - FastAPI server
  - Performans ipuçları
  - Sorun giderme

### ONNX Avantajları

✅ **Platform Bağımsız**: CPU, GPU, mobil, embedded cihazlar
✅ **Hızlı**: PyTorch'tan 1.5-2x daha hızlı
✅ **Küçük**: Quantization ile 4x daha küçük
✅ **Production-Ready**: Docker, FastAPI, REST API desteği

## 📖 Daha Fazla Örnek

Daha detaylı örnekler için Jupyter notebook'lara bakın:
- `notebooks/video_predictor_example.ipynb` - Detaylı video tracking
- `notebooks/image_predictor_example.ipynb` - Görüntü segmentasyonu

## 📝 Lisans

Apache 2.0 - Detaylar için [LICENSE](../LICENSE) dosyasına bakın.

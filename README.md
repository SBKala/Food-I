# Food-I 🍽️

**TR** | [EN](#en)

---

## TR

Türk yemeklerini tanıyan bir Android uygulaması. Kamera ile çekilen yemek fotoğrafını analiz ederek yemeğin adını ve kalori bilgisini gösterir.

### Tanınan Yemekler

- Beyti
- Domates Çorbası
- Mercimek Çorbası
- Sarma
- Su Böreği
- Tiramisu

### Model Performansı

| Metrik | Değer |
|--------|-------|
| Test Accuracy | %91 |
| Val Accuracy | %88 |
| Mimari | CNN (3x Conv2D + Dense) |
| Görüntü Boyutu | 64x64 |
| Epoch | 30 (Early Stopping ile) |

### Kullanılan Teknolojiler

- **Python** — Model eğitimi
- **TensorFlow / Keras** — CNN mimarisi, Data Augmentation, Early Stopping
- **TensorFlow Lite** — Modeli mobil cihaza taşıma
- **Java / Android Studio** — Android uygulaması
- **Google Colab** — Model eğitim ortamı

### Proje Yapısı

```
├── android/          # Android Studio projesi (model.tflite dahil)
├── model/
│   ├── food_classifier.ipynb   # Eğitim notebook'u (Colab)
│   └── food_classifier.py      # Eğitim kodu (Python)
└── README.md
```

### Dataset

Dataset 6 sınıf için toplamda ~667 görüntüden oluşmaktadır. Görüntüler yalnızca akademik araştırma amacıyla web'den derlenmiş olup telif hakkı gerekçesiyle kamuya açık olarak dağıtılmamaktadır.

### Kısıtlar

- Veri dengesizliği model performansını etkilemektedir (Beyti/Sarma karışıklığı)
- Daha fazla veri ile model performansı artırılabilir
- "Tanınamadı" sonucu %70 altı güven skorunda döner

---

## EN

<a name="en"></a>

An Android application that recognizes Turkish dishes. It analyzes food photos taken with the camera and displays the dish name and calorie information.

### Recognized Foods

- Beyti
- Tomato Soup (Domates Çorbası)
- Lentil Soup (Mercimek Çorbası)
- Stuffed Grape Leaves (Sarma)
- Water Pastry (Su Böreği)
- Tiramisu

### Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 91% |
| Val Accuracy | 88% |
| Architecture | CNN (3x Conv2D + Dense) |
| Image Size | 64x64 |
| Epochs | 30 (with Early Stopping) |

### Technologies Used

- **Python** — Model training
- **TensorFlow / Keras** — CNN architecture, Data Augmentation, Early Stopping
- **TensorFlow Lite** — Deploying model to mobile
- **Java / Android Studio** — Android application
- **Google Colab** — Training environment

### Project Structure

```
├── android/          # Android Studio project (includes model.tflite)
├── model/
│   ├── food_classifier.ipynb   # Training notebook (Colab)
│   └── food_classifier.py      # Training script (Python)
└── README.md
```

### Dataset

The dataset consists of ~667 images across 6 classes. Images were collected from the web for academic research purposes only and are not redistributed publicly due to copyright.

### Limitations

- Class imbalance affects model performance (Beyti/Sarma confusion)
- Performance can be improved with more data
- Returns "Unrecognized" for confidence scores below 70%
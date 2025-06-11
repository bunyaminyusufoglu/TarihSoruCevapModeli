import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

df = pd.read_csv('tarih_soru_yil_1000.csv')
print(df.head())

# Girdi ve çıktı
X = df['soru'].values
y = df['yil'].values

# Yılları sayısal sınıflara dönüştür
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Tokenizer ile metinleri sayılara dönüştür
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Dizileri aynı uzunluğa getir (padding)
X_pad = pad_sequences(X_seq, padding='post')

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_encoded, test_size=0.2, random_state=42)

# Model oluştur
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=X_pad.shape[1]))
model.add(LSTM(64))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

# Modeli derle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Modeli eğit
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Tahmin fonksiyonu
def tahmin_et(soru):
    seq = tokenizer.texts_to_sequences([soru])
    padded = pad_sequences(seq, maxlen=X_pad.shape[1], padding='post')
    pred_class = np.argmax(model.predict(padded), axis=-1)
    return le.inverse_transform(pred_class)[0]

# Örnek testler
print(tahmin_et("İstanbul ne zaman fethedildi?"))
print(tahmin_et("Türkiye Cumhuriyeti ne zaman kuruldu?"))
print(tahmin_et("Çanakkale Savaşı hangi yılda gerçekleşti?"))
print(tahmin_et("İstanbul'un fethi ne zaman oldu?"))

# Kullanıcıdan giriş alma
def model_ile_soru_sor():
    print("Modelle tarih sorusu cevaplama başladı! Çıkmak için 'çık' yaz.")
    while True:
        soru = input("Soru: ")
        if soru.lower() == 'çık':
            print("Görüşürüz!")
            break
        tahmin = tahmin_et(soru)
        print(f"Tahmini yıl: {tahmin}\n")

# Çalıştır
model_ile_soru_sor()

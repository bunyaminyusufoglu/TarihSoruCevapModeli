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
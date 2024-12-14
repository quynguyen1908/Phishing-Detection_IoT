import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Embedding, Dropout
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec 
import pickle

# Đọc dữ liệu từ file CSV
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')

# Xóa các cột không cần thiết
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Đổi tên các cột
df = df.rename(columns={'v1': 'TARGET', 'v2': 'MESSAGE'})

# Xóa các dòng trùng lặp
df.drop_duplicates(keep='first', inplace=True)

# Mã hóa nhãn (label encoding)
encoder = LabelEncoder()
df['TARGET'] = encoder.fit_transform(df['TARGET'])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x = df['MESSAGE']
y = df['TARGET']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Token hóa theo từ
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
max_len = max([len(seq) for seq in x_train_seq])
x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)

# Tạo mô hình Word2Vec
sentences = [text.split() for text in x_train]
word2vec_model = Word2Vec(sentences, vector_size=128, window=5, min_count=1, workers=4)

# Tạo embedding matrix
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 128))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Xây dựng mô hình LSTM với embedding đã được huấn luyện
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train_pad, y_train, epochs=5, batch_size=32, validation_data=(x_test_pad, y_test))

# Lưu mô hình đã huấn luyện
model.save('word2vec_model.h5')

# Lưu tokenizer
with open('w2p_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
import joblib

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

# Token hóa và tạo đặc trưng Count Vectors theo ký tự
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
x_train_count = count_vectorizer.fit_transform(x_train)
x_test_count = count_vectorizer.transform(x_test)

# Chuyển đổi dữ liệu Count Vectors thành mảng numpy
x_train_count = x_train_count.toarray()
x_test_count = x_test_count.toarray()

# Xây dựng mô hình học sâu
model = Sequential()
model.add(Dense(512, input_shape=(x_train_count.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train_count, y_train, epochs=10, batch_size=32, validation_data=(x_test_count, y_test))

# Lưu mô hình đã huấn luyện và vectorizer
model.save('char_count_model.h5')
joblib.dump(count_vectorizer, 'char_count_vectorizer.pkl')
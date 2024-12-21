import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import pickle
import joblib
import re
import nltk

# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Dữ liệu mẫu
sample_texts = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
    "Nah I don't think he goes to usf, he lives around here though.",
    "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT! Your account has been compromised. Please reset your password immediately.",
    "Can you send me the report by end of day?",
    "Win a brand new car! Text WIN to 12345 to enter the contest.",
    "Don't forget about the meeting at 3 PM.",
    "Your subscription is about to expire. Renew now to continue enjoying our services.",
    "Hey, just checking in. How have you been?",
    "Limited time offer! Get 50% off on all products. Shop now at our website.",
    "Are you coming to the party this weekend?",
    "Your package has been shipped and will arrive in 3-5 business days.",
    "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
    "Congratulations on your promotion! Well deserved.",
    "You have a new message from John. Check your inbox.",
    "Don't miss out on our summer sale! Up to 70% off on selected items.",
    "Can we reschedule our meeting to next week?",
    "Your order has been confirmed. Thank you for shopping with us.",
    "Hey, I found this great article on machine learning. Thought you might like it."
]
# Đọc dữ liệu từ file CSV
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df = df.rename(columns={'v1': 'TARGET', 'v2': 'MESSAGE'})
df.drop_duplicates(keep='first', inplace=True)
df.dropna(inplace=True)

# Label encoding
encoder = LabelEncoder()
df['TARGET'] = encoder.fit_transform(df['TARGET'])

# Loại bỏ các ký tự đặc biệt và chuyển đổi văn bản thành chữ thường
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['MESSAGE'] = df['MESSAGE'].apply(clean_text)

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['MESSAGE'] = df['MESSAGE'].apply(lemmatize_text)

# Chia dữ liệu
x = df['MESSAGE']
y = df['TARGET']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Word Embedding: Word2Vec
def word2vec():
    # Tải lại tokenizer
    with open('Word Embedding/w2p_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Token hóa và padding
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    max_len = max([len(seq) for seq in x_train_seq])
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)

    # Tải mô hình
    model = load_model('Word Embedding/word2vec_model.h5')

    # Đánh giá mô hình
    loss, accuracy = model.evaluate(x_test_pad, y_test)
    print(f'Accuracy: {accuracy * 100}')

    # Tokenize and pad sample texts
    sample_seq = tokenizer.texts_to_sequences(sample_texts)
    sample_pad = pad_sequences(sample_seq, maxlen=max_len)

    # In kết quả dự đoán
    predictions = model.predict(sample_pad)
    for text, prediction in zip(sample_texts, predictions):
        print(f'Text: {text}\nPrediction: {"Spam" if prediction > 0.5 else "Ham"}\n')

# Character Level TF-IDF
def tfidf():
    # Tải lại vectorizer
    tfidf_vectorizer = joblib.load('Character-level TF-IDF/tfidf_vectorizer.pkl')

    # Token hóa và tạo đặc trưng TF-IDF
    x_train_tfidf = tfidf_vectorizer.transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    # Tải mô hình
    model = load_model('Character-level TF-IDF/tfidf_model.h5')

    # Đánh giá mô hình
    y_pred = model.predict(x_test_tfidf)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100}')

    # Tokenize and transform sample texts
    sample_tfidf = tfidf_vectorizer.transform(sample_texts)

    # In kết quả dự đoán
    predictions = model.predict(sample_tfidf)
    predictions = (predictions > 0.5).astype(int)
    for text, prediction in zip(sample_texts, predictions):
        print(f'Text: {text}\nPrediction: {"Spam" if prediction == 1 else "Ham"}\n')

# Character Level Count Vectors
def count_vectors():
    # Tải lại vectorizer
    count_vectorizer = joblib.load('Character-level Count Vectors/char_count_vectorizer.pkl')

    # Token hóa và tạo đặc trưng Count Vectors
    x_train_count = count_vectorizer.transform(x_train)
    x_test_count = count_vectorizer.transform(x_test)

    # Tải mô hình
    model = load_model('Character-level Count Vectors/char_count_model.h5')

    # Đánh giá mô hình
    y_pred = model.predict(x_test_count)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100}')

    # Tokenize and transform sample texts
    sample_count = count_vectorizer.transform(sample_texts)

    # In kết quả dự đoán
    predictions = model.predict(sample_count)
    predictions = (predictions > 0.5).astype(int)
    for text, prediction in zip(sample_texts, predictions):
        print(f'Text: {text}\nPrediction: {"Spam" if prediction == 1 else "Ham"}\n')

# Character Embedding
def char_embedding():
    # Tải lại tokenizer
    with open('Character Embedding/ce_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Token hóa và padding
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    max_len = max([len(seq) for seq in x_train_seq])
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)

    # Tải mô hình
    model = load_model('Character Embedding/char_embedding_model.h5')

    # Đánh giá mô hình
    loss, accuracy = model.evaluate(x_test_pad, y_test)
    print(f'Accuracy: {accuracy * 100}')

    # Tokenize and pad sample texts
    sample_seq = tokenizer.texts_to_sequences(sample_texts)
    sample_pad = pad_sequences(sample_seq, maxlen=max_len)

    # In kết quả dự đoán
    predictions = model.predict(sample_pad)
    for text, prediction in zip(sample_texts, predictions):
        print(f'Text: {text}\nPrediction: {"Spam" if prediction > 0.5 else "Ham"}\n')

if __name__ == "__main__":
    # word2vec()
    # tfidf()
    count_vectors()
    # char_embedding()
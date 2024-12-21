# Ứng dụng Xử lý ngôn ngữ tự nhiên (NLP) vào phân loại dữ liệu (phân loại tin nhắn spam)

## Nghiên cứu xây dựng đặc trưng dựa vào ngữ nghĩa.

- Sử dụng phương pháp Embedding để biểu diễn các đối tượng (từ ngữ, ký tự) dưới dạng các vector số học trong không gian nhiều chiều. Các vector này được thiết kế sao cho các đối tượng có ngữ nghĩa tương tự sẽ có các vector gần nhau trong không gian vector.
- Cấp độ từ ngữ: Word Embedding (Word2Pec)
- Cấp độ ký tự: Character Embedding

##  Tiền xử lý dữ liệu, thiết kế mô hình học sâu (DL).

- Tiền xử lý dữ liệu (spam.csv):
    + Loại bỏ cột không cần thiết và dữ liệu trùng lặp, xử lý các giá trị thiếu.
    + Mã hóa nhãn (Label Encoding) để chuyển nhãn văn bản thành số.
    + Loại bỏ các ký tự đặc biệt và chuyển đổi văn bản thành chữ thường.
    + Chuẩn hóa các từ về dạng gốc của chúng, loại bỏ các từ dừng (stop words) không mang nhiều ý nghĩa.
    + Chuyển đổi văn bản thành các token (từ hoặc ký tự).
    + Đảm bảo tất cả các chuỗi có cùng độ dài.
    + Chuyển đổi các token thành các vector số.
- Thiết kế mô hình học sâu: 
    + Đồ án thiết kế 2 mô hình học sâu (Deep Learning): LSTM dùng cho vector đặc trưng tuần tự (Embedding) và Dense Network dùng cho vector đặc trưng không tuần tự (TF-IDF, Count Vectorizer).
    + LSTM (Long Short-Term Memory) là một loại mạng nơ-ron hồi quy (Recurrent Neural Network - RNN) được thiết kế để xử lý và dự đoán dữ liệu tuần tự.
    + Dense Network, còn được gọi là Fully Connected Network (FCN), là một loại mạng nơ-ron nhân tạo trong đó mỗi nơ-ron trong một lớp được kết nối với tất cả các nơ-ron trong lớp tiếp theo. Dense Network phù hợp cho các bài toán mà dữ liệu đầu vào không có tính tuần tự.
    + Cấu trúc mô hình của LSTM:
        * Dữ liệu đầu vào: Dữ liệu đầu vào là các đặc trưng được mã hóa dưới dạng vector số.
        * Embedding Layer: Biểu diễn từ dưới dạng vector ngữ nghĩa với kích thước cố định. Ma trận embedding được học trong quá trình huấn luyện.
        * LSTM Layers: Sử dụng 2 lớp LSTM có 128 tế bào. Lớp 1 học đặc trưng tuần tự, trả về chuỗi trạng thái, lớp 2 tổng hợp thông tin, trả về trạng thái cuối cùng.
        * Dropout Layers: Dùng để tắt ngẫu nhiên 50% nơ-ron trong quá trình huấn luyện, giúp giảm thiểu overfitting.
        * Dense Layer: Nhận các đặc trưng từ lớp LSTM và chuyển chúng thành xác suất thông qua hàm sigmoid, quyết định xem tin nhắn là spam hay không spam.
    + Cấu trúc mô hình của Dense Network:
        * Dữ liệu đầu vào: Dữ liệu đầu vào là các đặc trưng được mã hóa dưới dạng vector số.
        * Dense Layer 1: Lớp Dense đầu tiên có 512 neuron và sử dụng hàm kích hoạt ReLU để học các đặc trưng phi tuyến tính từ dữ liệu đầu vào.
        * Dense Layer 2: Lớp Dense thứ hai có 256 neuron và tiếp tục trích xuất các đặc trưng sâu hơn.
        * Dropout Layers: Dùng để tắt ngẫu nhiên 50% nơ-ron trong quá trình huấn luyện, giúp giảm thiểu overfitting.
        * Dense Layer cuối: Lớp đầu ra có 1 neuron và sử dụng hàm sigmoid để dự đoán xác suất (phân loại nhị phân).
    + Biên dịch mô hình:
        * Loss: Dùng hàm mất mát binary_crossentropy (cho phân loại nhị phân) để đo lường sự khác biệt giữa nhãn thực tế và dự đoán của mô hình.
        * Optimizer: Dùng thuật toán adam để điều chỉnh trọng số trong quá trình huấn luyện, tối ưu hóa hiệu quả mô hình.
        * Metrics: Đánh giá thông qua độ chính xác (accuracy) để theo dõi hiệu suất mô hình trong suốt quá trình huấn luyện.
    + Huấn luyện mô hình:
        * Mô hình sẽ được huấn luyện qua nhiều lần (epochs=10) với tất cả dữ liệu huấn luyện.
        * Dữ liệu huấn luyện sẽ được chia thành các batch nhỏ. Mỗi batch sẽ chứa 32 mẫu, và mô hình sẽ cập nhật trọng số sau mỗi 32 mẫu.
        * Trong suốt quá trình huấn luyện, sau mỗi epoch, mô hình sẽ được đánh giá trên tập kiểm tra.

## Tìm hiểu và ứng dụng vào mô hình phát hiện: character embedding, character level TF-IDF và character level count vectors.

- Character-level Embedding:
    + Là một phương pháp chuyển đổi các ký tự thành các vector có độ dài cố định.
    + Character-level Embedding giúp mô hình học các mối quan hệ giữa các ký tự trong một từ. Mỗi ký tự có thể được ánh xạ vào một vector với các giá trị số, và mô hình học được sự liên kết của các ký tự trong một từ.
- Character-level TF-IDF:
    + Phương pháp này đo lường sự quan trọng của một ký tự trong một chuỗi văn bản dựa trên tần suất xuất hiện của nó trong văn bản và trong toàn bộ tập dữ liệu.
    + Character-level TF-IDF giúp nắm bắt thông tin chi tiết hơn, đánh giá tầm quan trọng của từng ký tự trong văn bản, đặc biệt khi có các từ ngữ viết tắt hoặc lỗi chính tả.
- Character-level Count Vectors:
    + Là một phương pháp biểu diễn văn bản dựa trên tần suất xuất hiện của các ký tự trong văn bản, tương tự như TF-IDF, nhưng không tính đến trọng số của các ký tự.
    + Character-level Count Vectors hữu ích trong các tình huống mà các ký tự có tầm quan trọng riêng biệt, như phát hiện từ viết tắt, tên riêng, hoặc các ký tự bất thường trong văn bản.

## Thực nghiệm đối tượng thử nghiệm đúng theo số lượng nhãn có trong bộ dữ liệu

- Sử dụng bộ dữ liệu gồm các tin nhắn có nội dung và nhãn phân biệt: 'spam' là lừa đảo hoặc quảng cáo, khuyến mã, 'ham' là bình thường.
- Dữ liệu mẫu dùng để dự đoán:
(Spam)  Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
(Ham)   Nah I don't think he goes to usf, he lives around here though.
(Spam)  Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.
(Ham)   Hey, are we still meeting for lunch tomorrow?
(Spam)  URGENT! Your account has been compromised. Please reset your password immediately.
(Ham)   Can you send me the report by end of day?
(Spam)  Win a brand new car! Text WIN to 12345 to enter the contest.
(Ham)   Don't forget about the meeting at 3 PM.
(Spam)  Your subscription is about to expire. Renew now to continue enjoying our services.
(Ham)   Hey, just checking in. How have you been?
(Spam)  Limited time offer! Get 50% off on all products. Shop now at our website.
(Ham)   Are you coming to the party this weekend?
(Ham)   Your package has been shipped and will arrive in 3-5 business days.
(Ham)   Reminder: Your appointment is scheduled for tomorrow at 10 AM.
(Ham)   Congratulations on your promotion! Well deserved.
(Ham)   You have a new message from John. Check your inbox.
(Spam)  Don't miss out on our summer sale! Up to 70% off on selected items.
(Ham)   Can we reschedule our meeting to next week?
(Ham)   Your order has been confirmed. Thank you for shopping with us.
(Ham)   Hey, I found this great article on machine learning. Thought you might like it.
- Kết quả dự đoán:
    + Word Embedding (Word2Pec): 
    Accuracy: 88.97485733032227
    
    Text: Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
    Prediction: Ham

    Text: Nah I don't think he goes to usf, he lives around here though.
    Prediction: Ham

    Text: Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.
    Prediction: Ham

    Text: Hey, are we still meeting for lunch tomorrow?
    Prediction: Ham

    Text: URGENT! Your account has been compromised. Please reset your password immediately.  
    Prediction: Ham

    Text: Can you send me the report by end of day?
    Prediction: Ham

    Text: Win a brand new car! Text WIN to 12345 to enter the contest.
    Prediction: Ham

    Text: Don't forget about the meeting at 3 PM.
    Prediction: Ham

    Text: Your subscription is about to expire. Renew now to continue enjoying our services.  
    Prediction: Ham

    Text: Hey, just checking in. How have you been?
    Prediction: Ham

    Text: Limited time offer! Get 50% off on all products. Shop now at our website.
    Prediction: Ham

    Text: Are you coming to the party this weekend?
    Prediction: Ham

    Text: Your package has been shipped and will arrive in 3-5 business days.
    Prediction: Ham

    Text: Reminder: Your appointment is scheduled for tomorrow at 10 AM.
    Prediction: Ham

    Text: Congratulations on your promotion! Well deserved.
    Prediction: Ham

    Text: You have a new message from John. Check your inbox.
    Prediction: Ham

    Text: Don't miss out on our summer sale! Up to 70% off on selected items.
    Prediction: Ham

    Text: Can we reschedule our meeting to next week?
    Prediction: Ham

    Text: Your order has been confirmed. Thank you for shopping with us.
    Prediction: Ham

    Text: Hey, I found this great article on machine learning. Thought you might like it.     
    Prediction: Ham

    + Character Embedding: 
    Accuracy: 96.90521955490112

    Text: Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
    Prediction: Spam

    Text: Nah I don't think he goes to usf, he lives around here though.
    Prediction: Ham

    Text: Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.
    Prediction: Spam

    Text: Hey, are we still meeting for lunch tomorrow?
    Prediction: Ham

    Text: URGENT! Your account has been compromised. Please reset your password immediately.
    Prediction: Ham

    Text: Can you send me the report by end of day?   
    Prediction: Ham

    Text: Win a brand new car! Text WIN to 12345 to enter the contest.
    Prediction: Ham

    Text: Don't forget about the meeting at 3 PM.     
    Prediction: Ham

    Text: Your subscription is about to expire. Renew now to continue enjoying our services.
    Prediction: Ham

    Text: Hey, just checking in. How have you been?
    Prediction: Ham

    Text: Limited time offer! Get 50% off on all products. Shop now at our website.
    Prediction: Ham

    Text: Are you coming to the party this weekend?   
    Prediction: Ham

    Text: Your package has been shipped and will arrive in 3-5 business days.
    Prediction: Ham

    Text: Reminder: Your appointment is scheduled for tomorrow at 10 AM.
    Prediction: Ham

    Text: Congratulations on your promotion! Well deserved.
    Prediction: Ham

    Text: You have a new message from John. Check your inbox.
    Prediction: Ham

    Text: Don't miss out on our summer sale! Up to 70% off on selected items.
    Prediction: Ham

    Text: Can we reschedule our meeting to next week? 
    Prediction: Ham

    Text: Your order has been confirmed. Thank you for shopping with us.
    Prediction: Ham

    Text: Hey, I found this great article on machine learning. Thought you might like it.
    Prediction: Ham

    + Character-level TF-IDF:
    Accuracy: 98.45261121856866

    Text: Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
    Prediction: Spam

    Text: Nah I don't think he goes to usf, he lives around here though.
    Prediction: Ham

    Text: Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.
    Prediction: Spam

    Text: Hey, are we still meeting for lunch tomorrow?
    Prediction: Ham

    Text: URGENT! Your account has been compromised. Please reset your password immediately.
    Prediction: Ham

    Text: Can you send me the report by end of day?   
    Prediction: Ham

    Text: Win a brand new car! Text WIN to 12345 to enter the contest.
    Prediction: Spam

    Text: Don't forget about the meeting at 3 PM.     
    Prediction: Ham

    Text: Your subscription is about to expire. Renew now to continue enjoying our services.
    Prediction: Spam

    Text: Hey, just checking in. How have you been?   
    Prediction: Ham

    Text: Limited time offer! Get 50% off on all products. Shop now at our website.
    Prediction: Ham

    Text: Are you coming to the party this weekend?   
    Prediction: Ham

    Text: Your package has been shipped and will arrive in 3-5 business days.
    Prediction: Ham

    Text: Reminder: Your appointment is scheduled for tomorrow at 10 AM.
    Prediction: Ham

    Text: Congratulations on your promotion! Well deserved.
    Prediction: Ham

    Text: You have a new message from John. Check your inbox.
    Prediction: Ham

    Text: Don't miss out on our summer sale! Up to 70% off on selected items.
    Prediction: Ham

    Text: Can we reschedule our meeting to next week? 
    Prediction: Ham

    Text: Your order has been confirmed. Thank you for shopping with us.
    Prediction: Ham

    Text: Hey, I found this great article on machine learning. Thought you might like it.
    Prediction: Ham

    + Character-level Count Vectors:
    Accuracy: 98.16247582205028

    Text: Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
    Prediction: Spam

    Text: Nah I don't think he goes to usf, he lives around here though.
    Prediction: Ham

    Text: Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.
    Prediction: Spam

    Text: Hey, are we still meeting for lunch tomorrow?
    Prediction: Ham

    Text: URGENT! Your account has been compromised. Please reset your password immediately.
    Prediction: Ham

    Text: Can you send me the report by end of day?   
    Prediction: Ham

    Text: Win a brand new car! Text WIN to 12345 to enter the contest.
    Prediction: Spam

    Text: Don't forget about the meeting at 3 PM.     
    Prediction: Ham

    Text: Your subscription is about to expire. Renew now to continue enjoying our services.
    Prediction: Spam

    Text: Hey, just checking in. How have you been?   
    Prediction: Ham

    Text: Limited time offer! Get 50% off on all products. Shop now at our website.
    Prediction: Ham

    Text: Are you coming to the party this weekend?   
    Prediction: Ham

    Text: Your package has been shipped and will arrive in 3-5 business days.
    Prediction: Ham

    Text: Reminder: Your appointment is scheduled for tomorrow at 10 AM.
    Prediction: Ham

    Text: Congratulations on your promotion! Well deserved.
    Prediction: Ham

    Text: You have a new message from John. Check your inbox.
    Prediction: Ham

    Text: Don't miss out on our summer sale! Up to 70% off on selected items.
    Prediction: Ham

    Text: Can we reschedule our meeting to next week? 
    Prediction: Ham

    Text: Your order has been confirmed. Thank you for shopping with us.
    Prediction: Ham

    Text: Hey, I found this great article on machine learning. Thought you might like it.
    Prediction: Ham

- Character-level TF-IDF và Character-level Count Vectors có độ chinh xác cao và kết quả dự đoán khả quan hơn Word Embedding và Character-level Embedding.
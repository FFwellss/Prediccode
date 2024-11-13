import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import streamlit as st
import re

# Tạo giao diện Streamlit
st.title("Lấy ProductCode vs Segment")

# Huấn luyện mô hình Random Forest
train_file_path_1 = "Product Code.xlsx"  # Đường dẫn đến tệp huấn luyện đầu tiên
df1 = pd.read_excel(train_file_path_1)

# Chuẩn bị dữ liệu cho mô hình 1
df1['Mo_ta'] = df1['Mo_ta'].astype(str)
df1['Product_code'] = df1['Product_code'].astype(str)
df1['HScode'] = df1['HScode'].astype(str)

df1['Combined_Features'] = df1['HScode'] + ' ' + df1['Mo_ta']

vectorizer1 = TfidfVectorizer()
X1 = vectorizer1.fit_transform(df1['Combined_Features'])
y1 = df1['Product_code']

model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X1, y1)

# Huấn luyện mô hình Logistic Regression
train_file_path_2 = "Segment Dectect.xlsx"  # Đường dẫn đến tệp huấn luyện thứ hai
df2 = pd.read_excel(train_file_path_2)
dflean = df2[['Mo_ta', 'Segment']]

# Tiền xử lý dữ liệu cho mô hình 2
dflean['Mo_ta'] = dflean['Mo_ta'].str.lower().str.replace('[^a-zA-Z0-9\s]', '', regex=True)

X2 = dflean['Mo_ta']
y2 = dflean['Segment']

vectorizer2 = TfidfVectorizer()
X2_vec = vectorizer2.fit_transform(X2)

model2 = LogisticRegression()
model2.fit(X2_vec, y2)

# Tạo phần tải lên tệp Excel từ người dùng
uploaded_file = st.file_uploader("Tải lên tệp Excel mới với 2 trường HSCODE vs Mô Tả", type=["xlsx"])

if uploaded_file is not None:
    # Đọc tệp Excel mới từ người dùng
    new_df = pd.read_excel(uploaded_file)

    # Dự đoán Product Code từ mô hình 1
    def find_product_code(description):
        description_vec = vectorizer1.transform([description])
        predicted_product_code = model1.predict(description_vec)[0]
        return predicted_product_code

    # Dự đoán Segment từ mô hình 2
    def predict_segment(row):
        product_description = row['Mo_ta']
        if pd.notna(product_description):
            product_description = product_description.lower()
            product_description = re.sub(r'[^a-zA-Z0-9\s]', '', product_description)
            input_vec = vectorizer2.transform([product_description])
            return model2.predict(input_vec)[0]
        return None

    # Áp dụng hàm dự đoán cho cột 'Mo_ta' trong DataFrame mới
    new_df['Product_code'] = new_df['Mo_ta'].apply(find_product_code)
    new_df['Segment'] = new_df.apply(predict_segment, axis=1)

    # Lưu DataFrame mới với các cột dự đoán vào một file Excel mới
    output_file_path = 'Product_Segment.xlsx'
    new_df.to_excel(output_file_path, index=False)

    st.success(f"Dữ liệu đã được lưu vào {output_file_path}.")
    st.download_button("Tải xuống file Excel", data=open(output_file_path, 'rb'), file_name=output_file_path)

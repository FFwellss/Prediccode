import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re
import streamlit as st

# Đọc dữ liệu từ file Excel
df = pd.read_excel('Segment Dectect.xlsx') 
dflean = df[['Mo_ta', 'Segment']]

# Tiền xử lý dữ liệu
dflean['Mo_ta'] = dflean['Mo_ta'].str.lower().str.replace('[^a-zA-Z0-9\s]', '', regex=True)

# Chia dữ liệu và huấn luyện mô hình
X = dflean['Mo_ta']
y = dflean['Segment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Lưu mô hình và vectorizer
joblib.dump(model, 'trained_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Tạo giao diện Streamlit
st.title("Xuất thông tin Segement")
uploaded_file = st.file_uploader("Tải lên file Excel", type=["xlsx"])

if uploaded_file is not None:
    df_new = pd.read_excel(uploaded_file)
    df_new['Code'] = ''

    # Dự đoán và cập nhật 'Code'
    def predict_and_update_code(row):
        product_description = row['Mo_ta']
        if pd.notna(product_description):
            product_description = product_description.lower()
            product_description = re.sub(r'[^a-zA-Z0-9\s]', '', product_description)
            input_vec = vectorizer.transform([product_description])
            return model.predict(input_vec)[0]
        return None

    df_new['Code'] = df_new.apply(predict_and_update_code, axis=1)

    # Lưu và tải xuống file đã cập nhật
    output_file_path = 'updated_data.xlsx'
    df_new.to_excel(output_file_path, index=False)
    
    st.success("Dự đoán hoàn tất! Tải xuống file đã cập nhật:")
    st.download_button("Tải xuống file Excel", data=open(output_file_path, 'rb'), file_name='updated_data.xlsx')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Tạo giao diện Streamlit
st.title("Dự đoán Product Code từ mô tả sản phẩm")

# Đọc file huấn luyện từ tệp Excel đã có sẵn
train_file_path = "Product Code.xlsx"  # Đường dẫn đến tệp huấn luyện
df = pd.read_excel(train_file_path)

# Chuẩn bị dữ liệu
df['Mo_ta'] = df['Mo_ta'].astype(str)  # Đảm bảo cột 'Mo_ta' là kiểu chuỗi
df['Product_code'] = df['Product_code'].astype(str)
df['HScode'] = df['HScode'].astype(str)  # Đảm bảo cột 'HScode' là kiểu chuỗi

# Tạo một cột mới kết hợp HScode và Mo_ta
df['Combined_Features'] = df['HScode'] + ' ' + df['Mo_ta']

# Huấn luyện mô hình Random Forest
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Combined_Features'])
y = df['Product_code']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Tạo phần tải lên tệp Excel từ người dùng
uploaded_file = st.file_uploader("Tải lên tệp Excel mới", type=["xlsx"])

if uploaded_file is not None:
    # Đọc tệp Excel mới từ người dùng
    new_df = pd.read_excel(uploaded_file)

    # In ra DataFrame mới để kiểm tra
    st.write("Dữ liệu mới:")
    st.dataframe(new_df.head())

    # Sử dụng mô hình đã huấn luyện để tìm Product_code trong file mới
    def find_product_code(description):
        description_vec = vectorizer.transform([description])
        predicted_product_code = model.predict(description_vec)[0]
        return predicted_product_code

    # Áp dụng hàm find_product_code cho cột 'Mo_ta' trong DataFrame mới
    new_df['Predicted_Product_code'] = new_df['Mo_ta'].apply(find_product_code)

    # In ra DataFrame mới với cột 'Predicted_Product_code'
    st.write("Dữ liệu mới với mã sản phẩm dự đoán:")
    st.dataframe(new_df)

    # Lưu DataFrame mới với cột 'Predicted_Product_code' vào một file Excel mới
    output_file_path = 'New_Product_Codes_with_Predictions.xlsx'
    new_df.to_excel(output_file_path, index=False)

    st.success(f"Dữ liệu đã được lưu vào {output_file_path}.")
    st.download_button("Tải xuống file Excel đã dự đoán", data=open(output_file_path, 'rb'), file_name=output_file_path)

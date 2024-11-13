# prompt: gôm gọn đoạn code bên trên lại giúp tôi

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import joblib


df = pd.read_excel("SegmentDectect.xlsx") 
dflean = df[['Mo_ta', 'Segment']]

# Preprocess data
dflean['Mo_ta'] = dflean['Mo_ta'].str.lower().str.replace('[^a-zA-Z0-9\s]', '', regex=True)

# Split data and train model
X = dflean['Mo_ta']
y = dflean['Segment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save the trained model and vectorizer using joblib
joblib.dump(model, 'trained_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Upload Excel file
uploaded = files.upload()
for fn in uploaded.keys():
  df_new = pd.read_excel(fn)
  df_new['Code'] = ''

# Predict and update 'Code'
def predict_and_update_code(row):
  product_description = row['Mo_ta']
  if pd.notna(product_description):
    product_description = product_description.lower()
    product_description = re.sub(r'[^a-zA-Z0-9\s]', '', product_description)
    input_vec = vectorizer.transform([product_description])
    predicted_segment = model.predict(input_vec)[0]
    return predicted_segment
  else:
    return None

df_new['Code'] = df_new.apply(predict_and_update_code, axis=1)

# Save and download
df_new.to_excel('updated_data.xlsx', index=False)
files.download('updated_data.xlsx')

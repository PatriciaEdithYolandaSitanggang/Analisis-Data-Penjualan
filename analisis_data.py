# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify

# 1. Pengumpulan Data
df = pd.read_csv('data_penjualan.csv')

# 2. Data Cleaning
df['tanggal'] = pd.to_datetime(df['tanggal'])
df.dropna(inplace=True)

# 3. Data Transformation
df['bulan'] = df['tanggal'].dt.to_period('M')

# 4. Exploratory Data Analysis (EDA)
print("Deskripsi Data:")
print(df.describe())

# Visualisasi hubungan jumlah dan pendapatan
sns.pairplot(df)
plt.show()

# Visualisasi pendapatan per produk
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='produk', y='pendapatan')
plt.title('Pendapatan per Produk')
plt.show()

# 5. Modelling Data
# Menggunakan jumlah penjualan untuk memprediksi pendapatan
X = df[['jumlah']]
y = df['pendapatan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 6. Validasi dan Tuning Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 7. Interpretasi dan Penyajian Hasil
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Data Aktual')
plt.plot(X_test, y_pred, color='red', label='Prediksi')
plt.title('Prediksi Pendapatan')
plt.xlabel('Jumlah')
plt.ylabel('Pendapatan')
plt.legend()
plt.show()

# 8. Deployment dan Monitoring
# Menyediakan API sederhana menggunakan Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    jumlah = data['jumlah']
    prediction = model.predict([[jumlah]])
    return jsonify({'pendapatan': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

# 9. Maintenance dan Iterasi
# Ini adalah langkah berkelanjutan yang memerlukan pemantauan dan pembaruan model secara berkala.
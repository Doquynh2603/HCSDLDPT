import pymysql
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler

from dac_trung.pitch import pitch_function
from dac_trung.mfcc import mfcc_function
from dac_trung.rmse import rmse_function
from dac_trung.spectral import spectral_centroid_function
from dac_trung.zcr import zcr_function
def extract_features(input_file):
    # Kết nối MySQL
    conn = pymysql.connect(
        host='localhost',      # hoặc địa chỉ IP MySQL server
        user='root',
        password='123456789',
        database='hcsdldpt',
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    # # Đường dẫn gốc của thư mục Dataset
    # audio_dir = "D:/NAM4/KY2/HCSDLDPT/HCSDLDPT/src/dac_trung/Dataset/"

    # # Lấy tất cả đặc trưng từ CSDL
    cursor.execute("SELECT file_name, mfcc, pitch, spectral_centroid, zcr, rmse FROM female_voices")
    db_features = []
    file_names = []
    for row in cursor.fetchall():
        file_name = row[0]  # Lấy file_name (ví dụ: F01/F01-5001.wav)
        mfcc = np.array(json.loads(row[1]))  # Chuyển JSON thành mảng NumPy
        pitch = row[2]
        spectral_centroid = row[3]
        zcr = row[4]
        rmse = row[5]
        feature_vector = np.concatenate([mfcc, [pitch], [spectral_centroid], [zcr], [rmse]])
        db_features.append(feature_vector)
        file_names.append(file_name)

    db_features = np.array(db_features)

    # Chuẩn hóa đặc trưng
    scaler = MinMaxScaler()
    db_features_normalized = scaler.fit_transform(db_features)

    # Trích xuất đặc trưng từ file đầu vào
    # input_file = "D:\\NAM4\\KY2\\HCSDLDPT\\HCSDLDPT\\src\\TestData\\VIVOSDEV01_R002.wav"
    mfcc = mfcc_function(input_file)
    pitch = pitch_function(input_file)
    spectral_centroid = spectral_centroid_function(input_file)
    zcr = zcr_function(input_file)
    rmse = rmse_function(input_file)
    input_features = np.concatenate([mfcc, [pitch], [spectral_centroid], [zcr], [rmse]])
    input_features_normalized = scaler.transform([input_features])[0]

    # Định nghĩa trọng số
    weights = np.array([0.5/13] * 13 + [0.25, 0.1, 0.05, 0.1])

    # So sánh với CSDL
    distances = []
    for i, features in enumerate(db_features_normalized):
        weighted_diff = weights * (input_features_normalized - features) ** 2
        distance = np.sqrt(np.sum(weighted_diff))
        distances.append((file_names[i], distance))

    # Sắp xếp và trả về 3 file giống nhất
    distances.sort(key=lambda x: x[1])
    top_3 = distances[:3]
    for i, (file_name, distance) in enumerate(top_3, 1):
        similarity = 1 / (1 + distance)
        print(f"Top {i}: {file_name}, Độ tương đồng: {similarity:.4f}")
    results = [
                    {"file_name": file_name, "similarity": round(1 / (1 + distance), 4),"audio_path": f"/dataset/{file_name}"}
                    for file_name, distance in top_3
                ]
    # Đóng kết nối
    cursor.close()
    conn.close()
    return results

import pymysql
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from dac_trung.pitch import pitch_function
from dac_trung.mfcc import mfcc_function
from dac_trung.rmse import rmse_function
from dac_trung.spectral import spectral_centroid_function
from dac_trung.zcr import zcr_function
from dac_trung.speech_rate import speach_rate_function

def extract_features_and_compare(input_file):
    if not os.path.exists(input_file):
        print(f"Lỗi: File {input_file} không tồn tại")
        return []

    try:
        # Kết nối MySQL
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='123456789',
            database='hcsdldpt',
            charset='utf8mb4'
        )
        cursor = conn.cursor()

        # Lấy đặc trưng từ CSDL
        cursor.execute("SELECT file_name, mfcc, pitch, spectral_centroid, zcr, rmse, speech_rate FROM female_voices")
        db_features = []
        file_names = []
        for row in cursor.fetchall():
            file_name = row[0]
            mfcc = np.array(json.loads(row[1]))
            pitch = row[2]
            spectral_centroid = row[3]
            zcr = row[4]
            rmse = row[5]
            speech_rate = row[6]
            feature_vector = np.concatenate([mfcc, [pitch], [spectral_centroid], [zcr], [rmse], [speech_rate]])
            db_features.append(feature_vector)
            file_names.append(file_name)

        db_features = np.array(db_features)

        # Trích xuất đặc trưng từ file đầu vào
        mfcc = mfcc_function(input_file)
        pitch = pitch_function(input_file)
        spectral_centroid = spectral_centroid_function(input_file)
        zcr = zcr_function(input_file)
        rmse = rmse_function(input_file)
        speech_rate = speach_rate_function(input_file)

        # Tải tham số chuẩn hóa
        with open("norm_params.json", "r") as f:
            norm_params = json.load(f)
        mfcc_mean = np.array(norm_params["mfcc_mean"])
        mfcc_std = np.array(norm_params["mfcc_std"])
        pitch_mean = norm_params["pitch_mean"]
        pitch_std = norm_params["pitch_std"]
        sc_mean = norm_params["sc_mean"]
        sc_std = norm_params["sc_std"]
        zcr_mean = norm_params["zcr_mean"]
        zcr_std = norm_params["zcr_std"]
        rmse_mean = norm_params["rmse_mean"]
        rmse_std = norm_params["rmse_std"]
        sr_mean = norm_params["sr_mean"]
        sr_std = norm_params["sr_std"]

        # Chuẩn hóa Z-score
        mfcc_norm = (mfcc - mfcc_mean) / mfcc_std
        pitch_norm = (pitch - pitch_mean) / pitch_std
        sc_norm = (spectral_centroid - sc_mean) / sc_std
        zcr_norm = (zcr - zcr_mean) / zcr_std
        rmse_norm = (rmse - rmse_mean) / rmse_std
        sr_norm = (speech_rate - sr_mean) / sr_std
        input_features = np.concatenate([mfcc_norm, [pitch_norm], [sc_norm], [zcr_norm], [rmse_norm], [sr_norm]])

        # Tính khoảng cách và tìm ba file tương đồng
        weights = np.array([0.5/13] * 13 + [0.15, 0.1, 0.05, 0.1, 0.1])
        distances = []
        for i, features in enumerate(db_features):
            weighted_diff = weights * (input_features - features) ** 2
            distance = np.sqrt(np.sum(weighted_diff))
            distances.append((file_names[i], distance))

        distances.sort(key=lambda x: x[1])
        top_3 = distances[:3]
        results = [
            {"file_name": file_name, "similarity": round(1 / (1 + distance), 4), "audio_path": f"/dataset/{file_name}"}
            for file_name, distance in top_3
        ]
        print("Ba file tương đồng nhất:")
        for i, result in enumerate(results, 1):
            print(f"Top {i}: {result['file_name']}, Độ tương đồng: {result['similarity']}")

        # Lấy đặc trưng đã chuẩn hóa của ba file tương đồng
        top_3_filenames = [file_name for file_name, _ in top_3]
        top_3_features = []
        top_3_mfcc = []
        for file_name in top_3_filenames:
            cursor.execute("""
            SELECT mfcc, pitch, spectral_centroid, zcr, rmse, speech_rate
            FROM female_voices
            WHERE file_name = %s
            """, (file_name,))
            result = cursor.fetchone()
            if result:
                mfcc = json.loads(result[0])
                top_3_mfcc.append(mfcc)
                top_3_features.append({
                    "pitch": result[1],
                    "spectral_centroid": result[2],
                    "zcr": result[3],
                    "rmse": result[4],
                    "speech_rate": result[5]
                })

        # Kết hợp đặc trưng: file đầu vào + ba file tương đồng
        input_filename = os.path.basename(input_file)
        all_features = [{
            "pitch": pitch_norm,
            "spectral_centroid": sc_norm,
            "zcr": zcr_norm,
            "rmse": rmse_norm,
            "speech_rate": sr_norm
        }] + top_3_features
        all_mfcc = [mfcc_norm.tolist()] + top_3_mfcc
        all_filenames = [input_filename] + top_3_filenames  # Sử dụng tên file thực tế

        # Tạo DataFrame
        df_features = pd.DataFrame(all_features, index=all_filenames)
        df_mfcc = pd.DataFrame(all_mfcc, columns=[f"mfcc_{i+1}" for i in range(13)], index=all_filenames)

        # Trực quan hóa
        fig, ax = plt.subplots(2, 1, figsize=(18, 12))  # Tăng kích thước chiều rộng để tránh chồng lấn

        if not df_mfcc.empty:
            df_mfcc.plot(kind='bar', ax=ax[0])
            ax[0].set_title("So sánh Các Đặc Trưng MFCC (Đã Chuẩn Hóa)")
            ax[0].set_xlabel("File")
            ax[0].set_ylabel("Giá trị MFCC")
            ax[0].tick_params(axis='x', rotation=0)  # Nhãn nằm ngang
            ax[0].tick_params(axis='x', labelsize=8)  # Giảm kích thước chữ nếu cần

        df_features.plot(kind='bar', ax=ax[1])
        ax[1].set_title("So sánh Các Đặc Trưng Âm Thanh (Đã Chuẩn Hóa)")
        ax[1].set_xlabel("File")
        ax[1].set_ylabel("Giá trị đặc trưng")
        ax[1].tick_params(axis='x', rotation=0)  # Nhãn nằm ngang
        ax[1].tick_params(axis='x', labelsize=8)  # Giảm kích thước chữ nếu cần

        # Điều chỉnh khoảng cách để tránh chồng lấn
        plt.tight_layout()
        plt.show()

        cursor.close()
        conn.close()
        return results
    except Exception as e:
        print(f"Lỗi khi xử lý file: {e}")
        return []

# Gọi hàm
input_file = "D:\\NAM4\\KY2\\HCSDLDPT\\HCSDLDPT\\src\\TestData\\VIVOSSPK01_R021.wav"
results = extract_features_and_compare(input_file)
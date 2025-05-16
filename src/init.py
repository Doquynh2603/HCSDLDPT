from dac_trung.pitch import pitch_function
from dac_trung.mfcc import mfcc_function
from dac_trung.rmse import rmse_function
from dac_trung.spectral import spectral_centroid_function
from dac_trung.zcr import zcr_function
from dac_trung.speech_rate import speach_rate_function
import os
import pymysql
import json
import numpy as np
# Thông tin kết nối MySQL
conn = pymysql.connect(
    host='localhost',      # hoặc địa chỉ IP MySQL server
    user='root',
    password='123456789',
    database='hcsdldpt',
    charset='utf8mb4'
)
cursor = conn.cursor()
# Thư mục chứa các file âm thanh
audio_dir = "Dataset\\"
data = [] #lưu list các đặc trưng tạm thời để chuẩn hóa
# Trích xuất và lưu đặc trưng
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):  # Chỉ xử lý file WAV
            file_path = os.path.join(root, file)
            # Tạo file_name để lưu vào CSDL (bao gồm thư mục con, ví dụ: F01/F01-5001.wav)
            relative_path = os.path.relpath(file_path, audio_dir).replace("\\", "/")
    
            try:
                # Trích xuất các đặc trưng
                mfcc = mfcc_function(file_path)
                pitch = pitch_function(file_path)
                spectral_centroid = spectral_centroid_function(file_path)
                zcr = zcr_function(file_path)
                rmse = rmse_function(file_path)
                speech_rate = speach_rate_function(file_path)
                # Chuyển MFCC thành chuỗi JSON
                mfcc_json = json.dumps(mfcc.tolist())
                
                data.append({
                    "file" : relative_path,
                    "mfcc" : mfcc,
                    "pitch" : pitch,
                    "spectral_centroid" : spectral_centroid,
                    "zcr": zcr,
                    "rmse": rmse,
                    "speech_rate": speech_rate
                })
            except Exception as e: 
                print(f"Lỗi khi xử lý")
#tính giá trị trung bình và độ lệch chuẩn cho toàn bộ file
def z_score(x, mean, std):
    return (x-mean) / std if std != 0 else 0.0

# Lấy giá trị mảng cho từng đặc trưng scalar
pitch_list = [d["pitch"] for d in data]
sc_list = [d["spectral_centroid"] for d in data]
zcr_list = [d["zcr"] for d in data]
rmse_list = [d["rmse"] for d in data]
sr_list = [d["speech_rate"] for d in data]

# Tính mean và std
pitch_mean, pitch_std = np.mean(pitch_list), np.std(pitch_list)
sc_mean, sc_std = np.mean(sc_list), np.std(sc_list)
zcr_mean, zcr_std = np.mean(zcr_list), np.std(zcr_list)
rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
sr_mean, sr_std = np.mean(sr_list), np.std(sr_list)

#Chuẩn hóa scalar và chuẩn hóa từng chiều MFCC
mfcc_matrix = np.array([d["mfcc"] for d in data])  # shape (n_samples, n_mfcc)
mfcc_mean = np.mean(mfcc_matrix, axis=0)
mfcc_std = np.std(mfcc_matrix, axis=0)

norm_params = {
    "mfcc_mean": mfcc_mean.tolist(),
    "mfcc_std": mfcc_std.tolist(),
    "pitch_mean": float(pitch_mean),
    "pitch_std": float(pitch_std),
    "sc_mean": float(sc_mean),
    "sc_std": float(sc_std),
    "zcr_mean": float(zcr_mean),
    "zcr_std": float(zcr_std),
    "rmse_mean": float(rmse_mean),
    "rmse_std": float(rmse_std),
    "sr_mean": float(sr_mean),
    "sr_std": float(sr_std)
}


with open("norm_params.json", "w") as f:
    json.dump(norm_params, f)

#Lưu vào DB
for d in data:
    mfcc = (d["mfcc"] - mfcc_mean) / mfcc_std
    pitch = z_score(d["pitch"], pitch_mean, pitch_std)
    sc = z_score(d["spectral_centroid"], sc_mean, sc_std)
    zcr = z_score(d["zcr"], zcr_mean, zcr_std)
    rmse = z_score(d["rmse"], rmse_mean, rmse_std)
    sr = z_score(d["speech_rate"], sr_mean, sr_std)

    mfcc_json = json.dumps(mfcc.tolist())

    try:
        query = """
        INSERT INTO female_voices (file_name, mfcc, pitch, spectral_centroid, zcr, rmse, speech_rate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (d["file"], mfcc_json, pitch, sc, zcr, rmse, sr)
        cursor.execute(query, values)
        print(f"lưu file {d['file']} vào database thành công")
    except Exception as e:
        print(f"Lỗi khi lưu {d['file']}: {e}")


# Xác nhận thay đổi
conn.commit()

# Đóng kết nối
cursor.close()
conn.close()
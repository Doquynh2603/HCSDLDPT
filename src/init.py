from dac_trung.pitch import pitch_function
from dac_trung.mfcc import mfcc_function
from dac_trung.rmse import rmse_function
from dac_trung.spectral import spectral_centroid_function
from dac_trung.zcr import zcr_function
import os
import pymysql
import json
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
                
                # Chuyển MFCC thành chuỗi JSON
                mfcc_json = json.dumps(mfcc.tolist())
                
                # Chèn dữ liệu vào bảng
                query = """
                INSERT INTO female_voices (file_name, mfcc, pitch, spectral_centroid, zcr, rmse)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = (relative_path, mfcc_json, pitch, spectral_centroid, zcr, rmse)
                cursor.execute(query, values)
            #     query = """
            #     UPDATE female_voices SET pitch = %s WHERE file_name = %s
            #     """
            #     values = (pitch,relative_path)
            #     cursor.execute(query, values)
                
            #     print(f"Đã sửa đặc trưng pitch cho {relative_path}")
                
            except Exception as e:
                print(f"Lỗi khi xử lý {relative_path}: {e}")

# Xác nhận thay đổi
conn.commit()

# Đóng kết nối
cursor.close()
conn.close()
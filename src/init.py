from pitch import pitch_function
from mfcc import mffc_function
from mel import mel_function
from spectral import spectral_function
from zcr import zcr_function
from formats import formats_function
import pandas as pd
import os
import pymysql
# Thông tin kết nối MySQL
conn = pymysql.connect(
    host='localhost',      # hoặc địa chỉ IP MySQL server
    user='root',
    password='123456789',
    database='hcsdldpt',
    charset='utf8mb4'
)
cursor = conn.cursor()
# get subpath
listsubpath = []
directory = 'Dataset/'
for x in os.walk(directory):
    listsubpath.append(x[0].replace("\\", "/"))
listsubpath.pop(0)

# get files
allpath = []
for subpath in listsubpath:
    f = []
    for (dirpath, dirnames, filenames) in os.walk(subpath):
        f.extend(filenames)
        break
    for namefile in f:
        allpath.append(subpath + "/" + namefile)

# Danh sách lưu kết quả
data_list = []

# Header
header = ['Path', 'MFCC', 'Pitch', 'Formats', 'Mel-Spectrogram', 'Spectral Centroid', 'Zero-Crossing Rate']

for path in allpath:
    try:
        mfcc = mffc_function(path)
        pitch = pitch_function(path)
        formats = formats_function(path)
        mel = mel_function(path)
        spectral = spectral_function(path)
        zrc = zcr_function(path)

        sql = """
        INSERT INTO audio_features
        (audio_path, mfcc, pitch, formats, mel_spectrogram, spectral_centroid, zero_crossing_rate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            path, str(mfcc),str(pitch), str(formats), str(mel), float(spectral), float(zrc)
        ))
        print(f"Đã lưu: {path}")
    except Exception as e:
        print("======" + path)
        print("Have exception:", str(e))
conn.commit()
cursor.close()
conn.close()
print("Lưu dữ liệu vào MySQL thành công.")

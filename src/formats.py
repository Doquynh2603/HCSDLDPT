import os
import numpy as np
import parselmouth  # Cài đặt pypraat

def formats_function(file_path):
    # Đọc file âm thanh bằng Parselmouth
    sound = parselmouth.Sound(file_path)
    formants = sound.to_formant_burg()

    duration = sound.get_total_duration()
    time_step = 0.01  # mỗi 10ms
    times = np.arange(0, duration, time_step)

    f1_list, f2_list, f3_list = [], [], []

    for t in times:
        try:
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)

            if f1 and not np.isnan(f1): f1_list.append(f1)
            if f2 and not np.isnan(f2): f2_list.append(f2)
            if f3 and not np.isnan(f3): f3_list.append(f3)
        except:
            continue

    # Tính trung bình
    f1_avg = np.mean(f1_list) if f1_list else 0
    f2_avg = np.mean(f2_list) if f2_list else 0
    f3_avg = np.mean(f3_list) if f3_list else 0

    return np.array([f1_avg, f2_avg, f3_avg])

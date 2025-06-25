
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import get_window

def pitch_function(path):
    # Đọc file âm thanh và lấy sample rate
    sr, signal = wav.read(path)

    # Nếu là âm thanh stereo (2 kênh), chuyển thành mono bằng cách lấy trung bình 2 kênh
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Chuẩn hóa tín hiệu về khoảng [-1, 1] nếu chưa phải dạng float32
    if signal.dtype != np.float32:
        signal = signal / np.max(np.abs(signal))

    # Thiết lập tham số phân khung (25ms mỗi khung, bước nhảy 10ms)
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    window = get_window("hamming", frame_length)

    # Chia tín hiệu thành nhiều khung
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    pitches = []

    # Duyệt qua từng khung để tính pitch
    for i in range(num_frames):
        # Cắt khung và nhân với cửa sổ Hamming để giảm nhiễu biên
        frame = signal[i * hop_length:i * hop_length + frame_length] * window

        # Tính tự tương quan (autocorrelation) của khung
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]  # chỉ giữ nửa phía sau (dương)

        # Xác định khoảng lag hợp lệ tương ứng với dải tần (165Hz - 784Hz)
        min_lag = int(sr / 784)   # Lag tương ứng với tần số cao nhất (G5)
        max_lag = int(sr / 165)   # Lag tương ứng với tần số thấp nhất (E3)

        # Tìm chỉ số có giá trị autocorr cao nhất trong vùng hợp lệ
        peak_index = np.argmax(corr[min_lag:max_lag]) + min_lag
        r = corr[peak_index]

        # Nếu đỉnh có giá trị đủ lớn (ngưỡng đơn giản), tính pitch
        if r > 0.1:  # Ngưỡng năng lượng để lọc nhiễu
            pitch = sr / peak_index  # Công thức chuyển từ độ trễ sang tần số
            pitches.append(pitch)

    # Trả về giá trị pitch trung bình trên toàn bộ tín hiệu (nếu có)
    if len(pitches) > 0:
        return float(np.mean(pitches))
    else:
        return 0.0  # Không tìm được pitch hợp lệ
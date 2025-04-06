## **Đề tài BTL:**

Xây dựng hệ CSDL lưu trữ và tìm kiếm giọng nói phụ nữ.
1. Hãy xây dựng/sưu tầm một bộ dữ liệu gồm ít nhất 200 files âm thanh về giọng nói phụ nữ, các file có cùng độ dài (SV tùy chọn định dạng file âm thanh).
2. Hãy xây dựng một bộ thuộc tính để nhận diện giọng nói phụ nữ của các file âm thanh khác nhau từ bộ dữ liệu đã thu thập. Trình bày cụ thể về lý do lựa chọn và giá trị thông tin của các thuộc tính này.
3. Xây dựng hệ thống tìm kiếm âm thanh phụ nữ với đầu vào là một file âm thanh mới của một người nào đó (đối tượng đã có và không có trong dữ liệu), đầu ra là 3 files giống nhất, xếp thứ tự giảm dần về độ tương đồng giọng nói với âm thanh đầu vào.
        a.Trình bày sơ đồ khối của hệ thống và quy trình thực hiện yêu cầu của đề bài.
        b.Trình bày quá trình trích rút, lưu trữ và sử dụng các thuộc tính để tìm kiếm giọng nói phụ nữ trong hệ thống.
4. Demo hệ thống và đánh giá kết quả đã đạt được.

## **Các bước làm**

- Chuẩn bị dữ liệu ( 200 file giọng nói phụ nữ)
- Trích rút đặc trưng (có tool sẵn)
    - Sử dụng thư viện `librosa` hoặc `pypraat` để trích MFCC, pitch, spectral centroid,...
    - Các đặc trưng lựa chọn:
        - MFCC (Mel-Frequency Cepstral Coefficients):librosa
        - Pitch (Tần số cơ bản - F0):librosa
        - Formant Frequencies (F1, F2, F3):pypraat
        - Spectral Centroid:librosa
        - Zero-Crossing Rate (ZCR):librosa
        - Mel-Spectrogram:librosa
- Lưu vào cơ sở dữ liệu: sử dụng MySQL
- làm giao diện cơ bản + api : sử dụng flask trong python

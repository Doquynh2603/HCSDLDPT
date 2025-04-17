# pitch_reduced = np.array([
# #     [np.mean(p), np.std(p), np.min(p), np.max(p), np.median(p)]
# #     for p in pitch_data
# # ])

# # # Chuẩn hóa Pitch giảm chiều
# # scaler_pitch = StandardScaler()
# # pitch_reduced_scaled = scaler_pitch.fit_transform(pitch_reduced)

# # # Tạo DataFrame cho Pitch giảm chiều
# # pitch_columns = ['Pitch_mean', 'Pitch_std', 'Pitch_min', 'Pitch_max', 'Pitch_median']
# # pitch_reduced_df = pd.DataFrame(pitch_reduced_scaled, columns=pitch_columns)

# # print(f"Pitch reduced from {len(pitch_data[0])} to {pitch_reduced.shape[1]} dimensions")

# # # --- Giảm chiều Mel-Spectrogram ---
# # # Chuyển Mel-Spectrogram thành ma trận
# # mel_data = [parse_list(x) for x in df['Mel-Spectrogram']]
# # mel_data_array = np.array(mel_data)

# # # Chuẩn hóa trước khi PCA
# # scaler_mel = StandardScaler()
# # mel_scaled = scaler_mel.fit_transform(mel_data_array)

# # # Áp dụng PCA, giữ 95% phương sai
# # pca_mel = PCA(n_components=0.95)
# # mel_reduced = pca_mel.fit_transform(mel_scaled)

# # # Tạo DataFrame cho Mel-Spectrogram giảm chiều
# # mel_columns = [f'Mel_PC{i+1}' for i in range(mel_reduced.shape[1])]
# # mel_reduced_df = pd.DataFrame(mel_reduced, columns=mel_columns)

# # print(f"Mel-Spectrogram reduced from {mel_data_array.shape[1]} to {mel_reduced.shape[1]} dimensions")
# # print(f"Explained variance ratio: {sum(pca_mel.explained_variance_ratio_):.4f}")

# # # --- Giữ nguyên MFCC ---
# # mfcc_data = [parse_list(x) for x in df['MFCC']]
# # mfcc_data_array = np.array(mfcc_data)

# # # Chuẩn hóa MFCC
# # scaler_mfcc = StandardScaler()
# # mfcc_scaled = scaler_mfcc.fit_transform(mfcc_data_array)

# # # Tạo DataFrame cho MFCC
# # mfcc_columns = [f'MFCC_{i+1}' for i in range(mfcc_data_array.shape[1])]
# # mfcc_df = pd.DataFrame(mfcc_scaled, columns=mfcc_columns)

# # print(f"MFCC kept at {mfcc_data_array.shape[1]} dimensions")

# # # --- Giữ nguyên các đặc trưng khác ---
# # # Formants
# # formants_data = [parse_list(x) for x in df['Formats']]
# # formants_data_array = np.array(formants_data)
# # scaler_formants = StandardScaler()
# # formants_scaled = scaler_formants.fit_transform(formants_data_array)
# # formants_df = pd.DataFrame(formants_scaled, columns=['F1', 'F2', 'F3'])

# # # Spectral Centroid
# # centroid_data = df['Spectral Centroid'].values.reshape(-1, 1)
# # scaler_centroid = StandardScaler()
# # centroid_scaled = scaler_centroid.fit_transform(centroid_data)
# # centroid_df = pd.DataFrame(centroid_scaled, columns=['Spectral_Centroid'])

# # # Zero-Crossing Rate
# # zcr_data = df['Zero-Crossing Rate'].values.reshape(-1, 1)
# # scaler_zcr = StandardScaler()
# # zcr_scaled = scaler_zcr.fit_transform(zcr_data)
# # zcr_df = pd.DataFrame(zcr_scaled, columns=['Zero_Crossing_Rate'])

# # # --- Kết hợp dữ liệu ---
# # result_df = pd.concat([
# #     df[['Path']].reset_index(drop=True),
# #     mfcc_df.reset_index(drop=True),
# #     pitch_reduced_df.reset_index(drop=True),
# #     formants_df.reset_index(drop=True),
# #     mel_reduced_df.reset_index(drop=True),
# #     centroid_df.reset_index(drop=True),
# #     zcr_df.reset_index(drop=True)
# # ], axis=1)

# # # Lưu kết quả vào file mới
# # result_df.to_excel('CSDLDPT_reduced.xlsx', index=False)
# # print("Saved reduced data to 'CSDLDPT_reduced.xlsx'")

# # # --- Kiểm tra một vài mẫu ---
# # print("\nSample data (first 2 rows):")
# print(result_df.head(2))
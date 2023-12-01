import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# CSV 파일 읽어오기
file_path = r'C:\Users\KYH\Desktop\모바일 및 유비쿼터스\20170858_dataset'

File_list = [[0] * 3 for _ in range(7)]
for i in range(7) :
    File_list[i][0] = pd.read_csv(fr'{file_path}\{i}\linear.csv', sep=",", header=None)
    File_list[i][1] = pd.read_csv(fr'{file_path}\{i}\gyroscope.csv', sep=",", header=None)
    File_list[i][2] = pd.read_csv(fr'{file_path}\{i}\gravity.csv', sep=",", header=None)

# Accel data를 Linear accel data로
for i in range(7) :
    File_list[i][0].iloc[:, 2:5] = File_list[i][0].iloc[:, 2:5].values - File_list[i][2].iloc[:, 2:5].values.copy()

# Window size, Stride
window_size = 300
stride = 10

# feature 추출
all_combined_label = []
all_combined_data = []
for i in range(7) :
    Data_accel = File_list[i][0]
    Data_gyro = File_list[i][1]
    Data_gravity = File_list[i][2]

    accel_raw_data_x = Data_accel.iloc[:, 2]
    accel_raw_data_y = Data_accel.iloc[:, 3]
    accel_raw_data_z = Data_accel.iloc[:, 4]

    gyro_raw_data_x = Data_gyro.iloc[:, 2]
    gyro_raw_data_y = Data_gyro.iloc[:, 3]
    gyro_raw_data_z = Data_gyro.iloc[:, 4]

    gravity_raw_data_x = Data_gravity.iloc[:, 2]
    gravity_raw_data_y = Data_gravity.iloc[:, 3]
    gravity_raw_data_z = Data_gravity.iloc[:, 4]

    num_windows = (len(accel_raw_data_x) - window_size) // stride + 1
    labels = pd.DataFrame(np.full((num_windows, 1), Data_accel.iloc[:, 0][0]))

    accel_mean_x = pd.concat([pd.Series([accel_raw_data_x.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    accel_var_x = pd.concat([pd.Series([accel_raw_data_x.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    accel_mean_y = pd.concat([pd.Series([accel_raw_data_y.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    accel_var_y = pd.concat([pd.Series([accel_raw_data_y.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    accel_mean_z = pd.concat([pd.Series([accel_raw_data_z.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    accel_var_z = pd.concat([pd.Series([accel_raw_data_z.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    accel_corr_xy = pd.concat([pd.Series([accel_raw_data_x.iloc[i * stride:i * stride + window_size].corr(accel_raw_data_y.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T
    accel_corr_yz = pd.concat([pd.Series([accel_raw_data_y.iloc[i * stride:i * stride + window_size].corr(accel_raw_data_z.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T
    accel_corr_xz = pd.concat([pd.Series([accel_raw_data_x.iloc[i * stride:i * stride + window_size].corr(accel_raw_data_z.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T

    gyro_mean_x = pd.concat([pd.Series([gyro_raw_data_x.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    gyro_var_x = pd.concat([pd.Series([gyro_raw_data_x.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    gyro_mean_y = pd.concat([pd.Series([gyro_raw_data_y.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    gyro_var_y = pd.concat([pd.Series([gyro_raw_data_y.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    gyro_mean_z = pd.concat([pd.Series([gyro_raw_data_z.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    gyro_var_z = pd.concat([pd.Series([gyro_raw_data_z.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    gyro_corr_xy = pd.concat([pd.Series([gyro_raw_data_x.iloc[i * stride:i * stride + window_size].corr(gyro_raw_data_y.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T
    gyro_corr_yz = pd.concat([pd.Series([gyro_raw_data_y.iloc[i * stride:i * stride + window_size].corr(gyro_raw_data_z.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T
    gyro_corr_xz = pd.concat([pd.Series([gyro_raw_data_x.iloc[i * stride:i * stride + window_size].corr(gyro_raw_data_z.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T

    gravity_mean_x = pd.concat([pd.Series([gravity_raw_data_x.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    gravity_var_x = pd.concat([pd.Series([gravity_raw_data_x.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    gravity_mean_y = pd.concat([pd.Series([gravity_raw_data_y.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    gravity_var_y = pd.concat([pd.Series([gravity_raw_data_y.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    gravity_mean_z = pd.concat([pd.Series([gravity_raw_data_z.iloc[i * stride:i * stride + window_size].mean()]) for i in range(num_windows)], axis=0).T
    gravity_var_z = pd.concat([pd.Series([gravity_raw_data_z.iloc[i * stride:i * stride + window_size].var()]) for i in range(num_windows)], axis=0).T
    gravity_corr_xy = pd.concat([pd.Series([gravity_raw_data_x.iloc[i * stride:i * stride + window_size].corr(gravity_raw_data_y.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T
    gravity_corr_yz = pd.concat([pd.Series([gravity_raw_data_y.iloc[i * stride:i * stride + window_size].corr(gravity_raw_data_z.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T
    gravity_corr_xz = pd.concat([pd.Series([gravity_raw_data_x.iloc[i * stride:i * stride + window_size].corr(gravity_raw_data_z.iloc[i * stride:i * stride + window_size])]) for i in range(num_windows)], axis=0).T

    # 모든 feature를 하나로 묶기
    combined_data = pd.concat([accel_mean_x, accel_var_x, accel_mean_y, accel_var_y, accel_mean_z, accel_var_z, accel_corr_xy, accel_corr_yz, accel_corr_xz,
                               gyro_mean_x, gyro_var_x, gyro_mean_y, gyro_var_y, gyro_mean_z, gyro_var_z, gyro_corr_xy, gyro_corr_yz, gyro_corr_xz, 
                               gravity_mean_x, gravity_var_x, gravity_mean_y, gravity_var_y, gravity_mean_z, gravity_var_z, gravity_corr_xy, gravity_corr_yz, gravity_corr_xz], axis=1)
    
    all_combined_label.append(labels)
    all_combined_data.append(combined_data)

final_label = pd.concat(all_combined_label, axis=0, ignore_index=True)
final_data = pd.concat(all_combined_data, axis=0, ignore_index=True)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(final_data, final_label, test_size=0.2, random_state=42)


# 표준화 (Standardization)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 모델 훈련
svm_model = SVC(kernel='rbf', random_state=42, C=10, gamma=0.1)
svm_model.fit(X_train_scaled, y_train.values.ravel())  # ravel()을 사용하여 1차원 배열로 변환

# 훈련된 모델을 사용하여 테스트 데이터 예측
y_pred = svm_model.predict(X_test_scaled)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
rounded_normalized_conf_matrix = np.round(normalized_conf_matrix, 3)

print(f"Total Accuracy: {accuracy}")
print(rounded_normalized_conf_matrix)

#모델 저장, 스케일러 저장
joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# 각 feature의 최소값
min_values = scaler.data_min_
# 각 feature의 최대값
max_values = scaler.data_max_
# 스케일링에 사용된 값
scaling_values = scaler.scale_
# 최소값에 대한 정보
min_info = scaler.min_
# 스케일러 값 출력
print(min_values)
print(max_values)
print(scaling_values)
print(min_info)
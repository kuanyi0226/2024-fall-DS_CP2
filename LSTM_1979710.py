import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import zipfile

# 訓練集和模型存放的資料夾
train_folder = "DS_Dataset"
output_file = "upload.csv"
zip_file = "upload.zip"

# 存儲模型的字典
models = {}

# 定義一些輔助函數，用於 LSTM 輸入格式的處理
def create_lstm_dataset(df, features, look_back=1):
    """
    創建 LSTM 模型的訓練資料集，根據 look_back 來建立時間窗口。
    """
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[features].iloc[i:i + look_back].values)
        y.append(df["Power(mW)"].iloc[i + look_back])
    return np.array(X), np.array(y)

# 訓練每個裝置的模型
for location in range(1, 18):  # 共 17 台裝置
    train_file = os.path.join(train_folder, f"L{location}_train.csv")
    train_df = pd.read_csv(train_file)
    
    # 處理時間特徵
    train_df["DateTime"] = pd.to_datetime(train_df["DateTime"])
    train_df["Hour"] = train_df["DateTime"].dt.hour
    train_df["Minute"] = train_df["DateTime"].dt.minute
    train_df["DayOfWeek"] = train_df["DateTime"].dt.dayofweek
    
    # 處理天氣數據（加入互動項）
    train_df["WindTemp"] = train_df["WindSpeed(m/s)"] * train_df["Temperature(°C)"]
    
    # 定義特徵和標籤
    features = ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", 
                "Sunlight(Lux)", "Hour", "Minute", "DayOfWeek", "WindTemp"]
    
    # 創建 LSTM 資料集，這裡使用 5 小時的時間窗口
    X, y = create_lstm_dataset(train_df, features, look_back=5)
    
    # 分割數據集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 建立 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # 訓練模型
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # 驗證模型
    val_predictions = model.predict(X_val)
    print(f"Location {location} MAE: {mean_absolute_error(y_val, val_predictions):.2f}")
    
    # 保存模型
    models[location] = model

# 預測 upload.csv 中的數據
upload_path = os.path.join(train_folder, "upload.csv")
upload_df = pd.read_csv(upload_path, header=0, names=["序號", "答案"])

# 確保序號列為字符串
upload_df["序號"] = upload_df["序號"].astype(str)
upload_df["DateTime"] = pd.to_datetime(upload_df["序號"].str[:12], format="%Y%m%d%H%M")
upload_df["LocationCode"] = upload_df["序號"].str[-2:].astype(int)
upload_df["Hour"] = upload_df["DateTime"].dt.hour
upload_df["Minute"] = upload_df["DateTime"].dt.minute
upload_df["DayOfWeek"] = upload_df["DateTime"].dt.dayofweek

# 生成預測值
predictions = []
for _, row in upload_df.iterrows():
    location = row["LocationCode"]
    model = models[location]
    
    # 獲取特徵數據，這裡使用前5小時的資料來預測
    features = ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", 
                "Sunlight(Lux)", "Hour", "Minute", "DayOfWeek", "WindTemp"]
    pred_features = pd.DataFrame([row], columns=features).fillna(0)
    
    # 將數據轉換為 LSTM 輸入格式
    pred_features_window = np.array([pred_features.values] * 5).reshape((1, 5, len(features)))  # 使用5小時的滯後特徵
    
    # 預測
    prediction = model.predict(pred_features_window)[0][0]
    predictions.append(f"{max(0, round(prediction, 2)):.2f}")

# 保存預測結果
upload_df["答案"] = predictions
upload_df[["序號", "答案"]].to_csv(output_file, index=False, header=True, encoding='utf-8-sig')

# 壓縮為 zip 文件
with zipfile.ZipFile(zip_file, 'w') as zf:
    zf.write(output_file)

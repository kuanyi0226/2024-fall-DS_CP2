import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import zipfile

# 訓練集和模型存放的資料夾
train_folder = "DS_Dataset"
output_file = "upload.csv"
zip_file = "upload.zip"

# 存儲模型的字典
models = {}

# 訓練每個裝置的模型
for location in range(1, 18):  # 共 17 台裝置
    train_file = os.path.join(train_folder, f"L{location}_train.csv")
    train_df = pd.read_csv(train_file)
    
    # 處理時間特徵
    train_df["DateTime"] = pd.to_datetime(train_df["DateTime"])
    train_df["Hour"] = train_df["DateTime"].dt.hour
    train_df["Minute"] = train_df["DateTime"].dt.minute
    train_df["DayOfWeek"] = train_df["DateTime"].dt.dayofweek
    
    # 滯後特徵
    for lag in range(1, 6):  # 使用 1 到 5 小時的滯後數據
        train_df[f"Lag_{lag}_Power"] = train_df["Power(mW)"].shift(lag)
    
    # 處理天氣數據（加入互動項）
    train_df["WindTemp"] = train_df["WindSpeed(m/s)"] * train_df["Temperature(°C)"]
    
    # 定義特徵和標籤
    features = ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", 
                "Sunlight(Lux)", "Hour", "Minute", "DayOfWeek", "WindTemp"] + \
               [f"Lag_{lag}_Power" for lag in range(1, 6)]
    X = train_df[features].fillna(0)  # 用 0 填充缺失值
    y = train_df["Power(mW)"]
    
    # 分割數據集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用 XGBoost 訓練模型
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)
    
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
    
    # 獲取特徵數據
    features = ["WindSpeed(m/s)", "Pressure(hpa)", "Temperature(°C)", "Humidity(%)", 
                "Sunlight(Lux)", "Hour", "Minute", "DayOfWeek", "WindTemp"] + \
               [f"Lag_{lag}_Power" for lag in range(1, 6)]
    pred_features = pd.DataFrame([row], columns=features).fillna(0)
    
    # 預測
    prediction = model.predict(pred_features.values)[0]
    predictions.append(f"{max(0, round(prediction, 2)):.2f}")

# 保存預測結果
upload_df["答案"] = predictions
upload_df[["序號", "答案"]].to_csv(output_file, index=False, header=True, encoding='utf-8-sig')

# 壓縮為 zip 文件
with zipfile.ZipFile(zip_file, 'w') as zf:
    zf.write(output_file)

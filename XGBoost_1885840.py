import zipfile
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ------------------- 數據路徑和參數 -------------------
train_folder = "DS_Dataset"
output_file = "upload.csv"
zip_file = "upload.zip"
num_questions = 200  # 題目數量
num_time_points = 48  # 每個題目的時間點

# ------------------- 全局模型設定 -------------------
model_global = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42
)
models_per_device = {}
models_per_question = []

# ------------------- 特徵工程函數 -------------------
def create_features(df):
    """對輸入的 DataFrame 增加特徵"""
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek

    # 歷史特徵
    df["Prev_Hour_Power"] = df["Power(mW)"].shift(1)
    df["Prev_Day_Avg_Power"] = df["Power(mW)"].rolling(window=24).mean()
    
    # 新增交互特徵
    df["Wind_Temp_Interaction"] = df["WindSpeed(m/s)"] * df["Temperature(\u00b0C)"]
    
    # 填充缺失值
    df = df.fillna(0)
    return df

# ------------------- 數據讀取與模型訓練 -------------------
for location in range(1, 18):
    train_file = os.path.join(train_folder, f"L{location}_train.csv")
    train_df = pd.read_csv(train_file)
    
    # 日期處理
    train_df["DateTime"] = pd.to_datetime(train_df["DateTime"])
    train_df = create_features(train_df)

    # 特徵與標籤
    features = [
        "WindSpeed(m/s)", "Pressure(hpa)", "Temperature(\u00b0C)", "Humidity(%)", "Sunlight(Lux)",
        "Hour", "DayOfWeek", "Prev_Hour_Power", "Prev_Day_Avg_Power", "Wind_Temp_Interaction"
    ]
    X = train_df[features]
    y = train_df["Power(mW)"]

    # 數據標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 切分數據集
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 訓練裝置專屬模型
    model_device = XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=10, subsample=0.8, random_state=42
    )
    model_device.fit(X_train, y_train)
    models_per_device[location] = model_device

    # 全局模型訓練數據
    if location == 1:
        X_global, y_global = X_scaled, y
    else:
        X_global = np.vstack((X_global, X_scaled))
        y_global = np.hstack((y_global, y))

# 訓練全局模型
model_global.fit(X_global, y_global)

# ------------------- 處理 upload.csv -------------------
upload_path = os.path.join(train_folder, "upload.csv")
upload_df = pd.read_csv(upload_path, header=0, names=["\u5e8f\u865f", "\u7b54\u6848"])

# 確保編號列是字符串
upload_df["\u5e8f\u865f"] = upload_df["\u5e8f\u865f"].astype(str)

# 解析日期時間和位置編碼
upload_df["DateTime"] = pd.to_datetime(upload_df["\u5e8f\u865f"].str[:12], format="%Y%m%d%H%M")
upload_df["LocationCode"] = upload_df["\u5e8f\u865f"].str[-2:].astype(int)

predictions = []
for _, row in upload_df.iterrows():
    location = row["LocationCode"]
    model = models_per_device[location]  # 使用裝置專屬模型

    # 製作特徵
    X_pred = {
        "WindSpeed(m/s)": 0, "Pressure(hpa)": 0, "Temperature(\u00b0C)": 0,
        "Humidity(%)": 0, "Sunlight(Lux)": 0,
        "Hour": row["DateTime"].hour, "DayOfWeek": row["DateTime"].dayofweek,
        "Prev_Hour_Power": 0, "Prev_Day_Avg_Power": 0, "Wind_Temp_Interaction": 0
    }

    # 預測
    pred = model.predict(np.array([list(X_pred.values())]))[0]
    predictions.append(f"{max(0, round(pred, 2)):.2f}")

# 保存結果
upload_df["\u7b54\u6848"] = predictions
upload_df[["\u5e8f\u865f", "\u7b54\u6848"]].to_csv(output_file, index=False, header=True, encoding='utf-8-sig')

# 壓縮為 zip 文件
with zipfile.ZipFile(zip_file, 'w') as zf:
    zf.write(output_file)

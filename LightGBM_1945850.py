import zipfile
import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from joblib import Parallel, delayed

# ------------------- 數據路徑和參數 -------------------
train_folder = "DS_Dataset"
output_file = "upload.csv"
zip_file = "upload.zip"
num_questions = 200  # 題目數量
num_time_points = 48  # 每個題目的時間點

# ------------------- 特徵工程函數 -------------------
def create_features(df):
    """對輸入的 DataFrame 增加特徵 (僅保留與日照相關的特徵)"""
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["DayOfYear"] = df["DateTime"].dt.dayofyear

    # 只保留日照相關特徵和交互特徵
    df["Wind_Temp_Interaction"] = df["WindSpeed(m/s)"] * df["Temperature(\u00b0C)"]
    df["Humidity_Temp_Interaction"] = df["Humidity(%)"] * df["Temperature(\u00b0C)"]
    
    # 時間聚合特徵（僅與日照相關）
    df["Hour_Mean_Sunlight"] = df.groupby("Hour")["Sunlight(Lux)"].transform("mean")
    df["DayOfWeek_Mean_Sunlight"] = df.groupby("DayOfWeek")["Sunlight(Lux)"].transform("mean")
    
    # 填充缺失值
    df = df.fillna(0)
    return df

# ------------------- 模型訓練函數 -------------------
def train_device_model(location):
    """Train a specific device model (僅使用日照相關特徵)"""
    train_file = os.path.join(train_folder, f"L{location}_train.csv")
    train_df = pd.read_csv(train_file)
    
    # Process dates and create features
    train_df["DateTime"] = pd.to_datetime(train_df["DateTime"])
    train_df = create_features(train_df)

    # Features and labels (只選擇與日照相關的特徵)
    features = [
        "Sunlight(Lux)", "WindSpeed(m/s)", "Temperature(\u00b0C)", "Humidity(%)", 
        "Hour", "DayOfWeek", "DayOfYear", 
        "Wind_Temp_Interaction", "Humidity_Temp_Interaction", 
        "Hour_Mean_Sunlight", "DayOfWeek_Mean_Sunlight"
    ]
    X = train_df[features]
    y = train_df["Power(mW)"]

    # Feature scaling and selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(f_regression, k=5)  # Select top 5 features
    X_selected = selector.fit_transform(X_scaled, y)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Train LightGBM model with early stopping
    model = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(50)],
    )
    return location, model, selector, scaler

# ------------------- 並行模型訓練 -------------------
results = Parallel(n_jobs=-1)(
    delayed(train_device_model)(location) for location in range(1, 18)
)
models_per_device = {loc: model for loc, model, _, _ in results}
selectors_per_device = {loc: selector for loc, _, selector, _ in results}
scalers_per_device = {loc: scaler for loc, _, _, scaler in results}

# ------------------- 處理 upload.csv -------------------
upload_path = os.path.join(train_folder, "upload.csv")
upload_df = pd.read_csv(upload_path, header=0, names=["\u5e8f\u865f", "\u7b54\u6848"])

# 確保編號列是字符串
upload_df["\u5e8f\u865f"] = upload_df["\u5e8f\u865f"].astype(str)

# 解析日期時間和位置編碼
upload_df["DateTime"] = pd.to_datetime(upload_df["\u5e8f\u865f"].str[:12], format="%Y%m%d%H%M")
upload_df["LocationCode"] = upload_df["\u5e8f\u865f"].str[-2:].astype(int)

# 預測
predictions = []
for _, row in upload_df.iterrows():
    location = row["LocationCode"]
    model = models_per_device[location]
    selector = selectors_per_device[location]
    scaler = scalers_per_device[location]

    # 製作特徵
    X_pred = {
        "Sunlight(Lux)": 0, "WindSpeed(m/s)": 0, "Temperature(\u00b0C)": 0,
        "Humidity(%)": 0,
        "Hour": row["DateTime"].hour, "DayOfWeek": row["DateTime"].dayofweek, 
        "DayOfYear": row["DateTime"].timetuple().tm_yday,
        "Wind_Temp_Interaction": 0, "Humidity_Temp_Interaction": 0,
        "Hour_Mean_Sunlight": 0, "DayOfWeek_Mean_Sunlight": 0
    }
    X_pred_scaled = scaler.transform([list(X_pred.values())])
    X_pred_selected = selector.transform(X_pred_scaled)
    pred = model.predict(X_pred_selected)[0]
    predictions.append(f"{max(0, round(pred, 2)):.2f}")

# 保存結果
upload_df["\u7b54\u6848"] = predictions
upload_df[["\u5e8f\u865f", "\u7b54\u6848"]].to_csv(output_file, index=False, header=True, encoding='utf-8-sig')

# 壓縮為 zip 文件
with zipfile.ZipFile(zip_file, 'w') as zf:
    zf.write(output_file)

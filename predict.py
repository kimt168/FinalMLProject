import joblib
import numpy as np
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
import pandas as pd
import os

def predict_from_input(tv_budget: float, radio_budget: float, newspaper_budget: float) -> float:
    try:
        # 1. Tạo mảng dữ liệu từ các giá trị đầu vào
        input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    
        # 2. Load tập dữ liệu huấn luyện (để lấy thông tin cột)
        train_X = pd.read_csv('./Data/Train_X_std.csv')
        
        # Tạo DataFrame từ input_data với các cột tương ứng với train_X
        input_data_df = pd.DataFrame(input_data, columns=train_X.columns)

        # 3. Load mô hình stacking đã huấn luyện từ file .pkl
        stacking_model = joblib.load('stacked_meta_model.pkl')

        # 4. Dự đoán kết quả
        prediction = stacking_model.predict(input_data_df)

        print("Dự đoán kết quả:", prediction)

    except FileNotFoundError as e:
        # Trả về lỗi nếu file không tồn tại
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        # Trả về lỗi khác
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    # Trả về kết quả dự đoán (lấy giá trị đầu tiên từ mảng dự đoán)
    return float(prediction[0])

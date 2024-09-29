import joblib
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
def predict_from_input(tv_budget: float, radio_budget: float, newspaper_budget: float) -> float:
    # 1. Load các mô hình đã train
    meta_model = joblib.load('stacked_meta_model_best_alpha.pkl')
    linear_model = joblib.load('Data/linear_regression_model.pkl')
    mlp_model = joblib.load('mlp_regression1_model.pkl')
    ridge_model = joblib.load('ridge_regression_model_new2.pkl')

    # 2. Tạo mảng dữ liệu từ các giá trị đầu vào
    # Ví dụ: giá trị đầu vào là một danh sách [tv_budget, radio_budget, newspaper_budget]
    input_data_df = np.array([[tv_budget, radio_budget, newspaper_budget]])

    

    # 3. Dự đoán từ các mô hình cơ sở (Linear, MLP, Ridge)
    pred_linear = linear_model.predict(input_data_df)
    pred_mlp = mlp_model.predict(input_data_df)
    pred_ridge = ridge_model.predict(input_data_df)

    # 4. Stack các dự đoán từ mô hình cơ sở
    meta_input = np.column_stack((pred_linear, pred_mlp, pred_ridge))

    # 5. Dự đoán với mô hình meta đã huấn luyện
    prediction = meta_model.predict(meta_input)

    # 6. Trả về kết quả dự đoán (lấy giá trị đầu tiên từ mảng dự đoán)
    return float(prediction[0])

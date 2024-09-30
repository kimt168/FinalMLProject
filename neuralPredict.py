import joblib
import numpy as np

def neuralpredict(tv_budget: float, radio_budget: float, newspaper_budget: float) -> float:
    # Tải mô hình từ tệp pkl bằng joblib
    model = joblib.load('mlp_regression2_model.pkl')
    
    # Tạo mảng dữ liệu từ các giá trị đầu vào
    input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    
    # Sử dụng mô hình để dự đoán
    prediction = model.predict(input_data)
    
    # Trả về kết quả dự đoán
    return prediction[0]
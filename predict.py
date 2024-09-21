import joblib
import pandas as pd

# Tải mô hình và scaler từ file
stacking_model = joblib.load('stacking_model.pkl')
scaler = joblib.load('scaler.pkl')

# Hàm dự đoán với các tham số truyền vào
def predict_from_input(tv_budget, radio_budget, newspaper_budget):
    # Chuyển đổi dữ liệu nhập vào thành DataFrame
    input_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], 
                              columns=['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)'])
    
    # Chuyển đổi dữ liệu nhập vào thành dạng đã được chuẩn hóa
    input_data_scaled = scaler.transform(input_data)
    
    # Dự đoán kết quả
    prediction = stacking_model.predict(input_data_scaled)
    
    return f"{prediction[0]:.2f}"

# Gọi hàm dự đoán với các tham số cụ thể
#result = predict_from_input(230.1, 37.8, 69.2)
#print(result)

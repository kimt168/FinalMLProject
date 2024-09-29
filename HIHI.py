from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV

# Step 2: Load pre-trained models
linear_model = joblib.load('/content/drive/MyDrive/sale/Data/linear_regression_model.pkl')
mlp_model = joblib.load('/content/drive/MyDrive/sale/Data/mlp_regression_model.pkl')
ridge_model = joblib.load('/content/drive/MyDrive/sale/Data/ridge_regression_model.pkl')

# Bước 3: Tạo dự đoán từ các mô hình cơ sở bằng Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Tạo các dự đoán cho từng lớp mô hình cơ sở
train_pred_linear_cv = cross_val_predict(linear_model, Train_X_std, Train_Y, cv=kf)
train_pred_mlp_cv = cross_val_predict(mlp_model, Train_X_std, Train_Y, cv=kf)
train_pred_ridge_cv = cross_val_predict(ridge_model, Train_X_std, Train_Y, cv=kf)

# Ghép các dự đoán thành ma trận đặc trưng cho mô hình meta
train_meta_X_cv = np.column_stack((train_pred_linear_cv, train_pred_mlp_cv, train_pred_ridge_cv))

# Bước 4: Tìm kiếm alpha tốt nhất cho mô hình meta bằng GridSearchCV
# Khởi tạo mô hình Ridge
meta_model = Ridge()

# Tạo lưới tham số để tìm kiếm alpha tốt nhất
param_grid = {'alpha': np.logspace(-3, 3, 50)}  # Các giá trị alpha từ 0.001 đến 1000

# Sử dụng GridSearchCV để tìm alpha tốt nhất
grid_search = GridSearchCV(meta_model, param_grid, cv=kf, scoring='r2')
grid_search.fit(train_meta_X_cv, Train_Y)

# Lấy ra alpha tốt nhất
best_alpha = grid_search.best_params_['alpha']
print(f'Giá trị alpha tốt nhất: {best_alpha}')

# Bước 5: Huấn luyện lại mô hình meta với alpha tốt nhất
meta_model_opt = Ridge(alpha=best_alpha)
meta_model_opt.fit(train_meta_X_cv, Train_Y)

# Bước 6: Đánh giá mô hình trên tập huấn luyện, xác thực và kiểm tra
# Dự đoán trên tập huấn luyện bằng các mô hình cơ sở đã huấn luyện
train_pred_linear = linear_model.predict(Train_X_std)
train_pred_mlp = mlp_model.predict(Train_X_std)
train_pred_ridge = ridge_model.predict(Train_X_std)

# Tạo ma trận đặc trưng cho tập huấn luyện
train_meta_X = np.column_stack((train_pred_linear, train_pred_mlp, train_pred_ridge))

# Dự đoán và đánh giá trên tập huấn luyện
train_meta_pred = meta_model_opt.predict(train_meta_X)

# Tính toán các chỉ số hiệu suất trên tập huấn luyện
train_mse = mean_squared_error(Train_Y, train_meta_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(Train_Y, train_meta_pred)

# Dự đoán trên tập xác thực và kiểm tra bằng các mô hình đã huấn luyện
val_pred_linear = linear_model.predict(Val_X_std)
val_pred_mlp = mlp_model.predict(Val_X_std)
val_pred_ridge = ridge_model.predict(Val_X_std)

test_pred_linear = linear_model.predict(Test_X_std)
test_pred_mlp = mlp_model.predict(Test_X_std)
test_pred_ridge = ridge_model.predict(Test_X_std)

# Tạo ma trận đặc trưng cho tập xác thực và kiểm tra
val_meta_X = np.column_stack((val_pred_linear, val_pred_mlp, val_pred_ridge))
test_meta_X = np.column_stack((test_pred_linear, test_pred_mlp, test_pred_ridge))

# Dự đoán và đánh giá trên tập xác thực và kiểm tra
val_meta_pred = meta_model_opt.predict(val_meta_X)
test_meta_pred = meta_model_opt.predict(test_meta_X)

# Tính toán các chỉ số hiệu suất trên tập xác thực và tập kiểm tra
val_mse = mean_squared_error(Val_Y, val_meta_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(Val_Y, val_meta_pred)

test_mse = mean_squared_error(Test_Y, test_meta_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(Test_Y, test_meta_pred)

# In các chỉ số hiệu suất trên tập huấn luyện, xác thực và kiểm tra
print(f'Training R²: {train_r2:.4f}')
print(f'Validation R²: {val_r2:.4f}')
print(f'Test R²: {test_r2:.4f}')
print(f'Training MSE: {train_mse:.4f}')
print(f'Validation MSE: {val_mse:.4f}')
print(f'Test MSE: {test_mse:.4f}')
print(f'Training RMSE: {train_rmse:.4f}')
print(f'Validation RMSE: {val_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')




train_meta_pred = meta_model_opt.predict(train_meta_X_cv)
val_meta_pred = meta_model_opt.predict(val_meta_X)
test_meta_pred = meta_model_opt.predict(test_meta_X)

# Bước 2: Tạo một figure với 3 subplot cho train, validation và test
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Đồ thị cho tập huấn luyện
axs[0].scatter(Train_Y, train_meta_pred, color='blue', label='Dự đoán')
axs[0].plot([Train_Y.min(), Train_Y.max()], [Train_Y.min(), Train_Y.max()], color='red', lw=2, label='Giá trị thực')
axs[0].set_title('Tập huấn luyện: Giá trị thực vs Dự đoán')
axs[0].set_xlabel('Giá trị thực')
axs[0].set_ylabel('Dự đoán')
axs[0].legend()
axs[0].grid(True)

# Đồ thị cho tập xác thực
axs[1].scatter(Val_Y, val_meta_pred, color='blue', label='Dự đoán')
axs[1].plot([Val_Y.min(), Val_Y.max()], [Val_Y.min(), Val_Y.max()], color='red', lw=2, label='Giá trị thực')
axs[1].set_title('Tập xác thực: Giá trị thực vs Dự đoán')
axs[1].set_xlabel('Giá trị thực')
axs[1].set_ylabel('Dự đoán')
axs[1].legend()
axs[1].grid(True)

# Đồ thị cho tập kiểm tra
axs[2].scatter(Test_Y, test_meta_pred, color='blue', label='Dự đoán')
axs[2].plot([Test_Y.min(), Test_Y.max()], [Test_Y.min(), Test_Y.max()], color='red', lw=2, label='Giá trị thực')
axs[2].set_title('Tập kiểm tra: Giá trị thực vs Dự đoán')
axs[2].set_xlabel('Giá trị thực')
axs[2].set_ylabel('Dự đoán')
axs[2].legend()
axs[2].grid(True)

# Bước 3: Hiển thị đồ thị
plt.tight_layout()
plt.show()

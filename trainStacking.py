import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
train_X = pd.read_csv('data/Train_X_std.csv')
train_Y = pd.read_csv('data/Train_Y.csv')
val_X = pd.read_csv('data/Val_X_std.csv')
val_Y = pd.read_csv('data/Val_Y.csv')
test_X = pd.read_csv('data/Test_X_std.csv')
test_Y = pd.read_csv('data/Test_Y.csv')

# Step 2: Load pre-trained models
linear_model = joblib.load('linear_regression_model.pkl')
mlp_model = joblib.load('mlp_regression_model.pkl')
ridge_model = joblib.load('ridge_regression_model2.pkl')

# Step 3: Create first-level predictions
train_pred_linear = linear_model.predict(train_X)
train_pred_mlp = mlp_model.predict(train_X)
train_pred_ridge = ridge_model.predict(train_X)

val_pred_linear = linear_model.predict(val_X)
val_pred_mlp = mlp_model.predict(val_X)
val_pred_ridge = ridge_model.predict(val_X)

# Stack predictions for meta-model
train_meta_X = np.column_stack((train_pred_linear, train_pred_mlp, train_pred_ridge))
val_meta_X = np.column_stack((val_pred_linear, val_pred_mlp, val_pred_ridge))



# Step 4: Train the meta-model (Ridge Regression)
meta_model = Ridge(alpha=1.0)  # You can tune the alpha value
meta_model.fit(train_meta_X, train_Y)

# Step 5: Evaluate the meta-model on validation set
val_meta_pred = meta_model.predict(val_meta_X)
val_mse = mean_squared_error(val_Y, val_meta_pred)
val_r2 = r2_score(val_Y, val_meta_pred)

print(f'Validation MSE of Stacked Model: {val_mse}')
print(f'Validation R-squared of Stacked Model: {val_r2}')

# Step 6: Make final predictions on the test set
test_meta_pred = meta_model.predict(test_meta_X)
test_mse = mean_squared_error(test_Y, test_meta_pred)
test_r2 = r2_score(test_Y, test_meta_pred)

print(f'Test MSE of Stacked Model: {test_mse}')
print(f'Test R-squared of Stacked Model: {test_r2}')

# Save the optimized model
joblib.dump(meta_model, 'stacked_meta_model_optimized.pkl')

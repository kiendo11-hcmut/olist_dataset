import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

#  Đọc dữ liệu
df = pd.read_csv("data_cleaned0_olist.csv")

features = [
    "order_item_count", "delivery_days", "execution_days",
    "estimated_days", "avg_review_score", "total_freight_value",
    "total_payment", "payment_installments"
]
target = "total_price"


# Phân tích tương quan
correlations = df[features + [target]].corr()[target].sort_values(ascending=False)
print("\n Correlation with total_price:")
print(correlations)

# Chọn các feature có |corr| > 0.1 (có tương quan đáng kể)
selected_features = [f for f in features if abs(correlations[f]) > 0.1]
print("\n Selected features:", selected_features)

X_vif = df[selected_features].copy()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\n VIF (kiểm tra trùng thông tin giữa các feature):")
print(vif_data)

# Tạo stratification labels
# Tạo bins cho target để stratify
df['target_bin'] = pd.qcut(df[target], q=5, labels=False, duplicates='drop')

# tạo composite stratification key từ nhiều features quan trọng nhất
# lấy top 2-3 features có correlation cao nhất
top_features = correlations[selected_features].abs().nlargest(min(3, len(selected_features))).index.tolist()

# Tạo bins cho từng top feature
for feat in top_features:
    df[f'{feat}_bin'] = pd.qcut(df[feat], q=3, labels=False, duplicates='drop')

# Kết hợp các bins thành một stratification key
strat_cols = ['target_bin'] + [f'{feat}_bin' for feat in top_features]
df['strat_key'] = df[strat_cols].astype(str).agg('_'.join, axis=1)

# Chia dữ liệu Train/Test với Stratification
X = df[selected_features]
y = df[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['strat_key']
    )
    print("\n Sử dụng stratified sampling")
except ValueError:
    # Nếu có nhóm quá nhỏ, fallback về random split
    print("\n Một số nhóm quá nhỏ, sử dụng random sampling")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# So sánh phân phối giữa train và test
print("\n So sánh phân phối Train vs Test:")
print("\nTarget (total_price):")
print(f"Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, Min: {y_train.min():.2f}, Max: {y_train.max():.2f}")
print(f"Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}, Min: {y_test.min():.2f}, Max: {y_test.max():.2f}")

print("\nTop Features:")
for feat in top_features[:3]:
    print(f"\n{feat}:")
    print(f"Train - Mean: {X_train[feat].mean():.2f}, Std: {X_train[feat].std():.2f}")
    print(f"Test  - Mean: {X_test[feat].mean():.2f}, Std: {X_test[feat].std():.2f}")

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Evaluation:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Xem trọng số (hệ số hồi quy)
coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n Feature Importance (Linear Regression Coefficients):")
print(coef_df)

# Xuất tập train và test ra file
# ===============================
train_df = X_train.copy()
train_df[target] = y_train
train_df.to_csv("train_set0.csv", index=False)

# Tạo DataFrame test (có cả giá trị thực và dự đoán)
test_df = X_test.copy()
test_df[target] = y_test
test_df['predicted_' + target] = y_pred
test_df['prediction_error'] = y_test.values - y_pred
test_df.to_csv("test_set0.csv", index=False)


# --- PHÂN TÍCH CÁC MẪU NGOẠI LAI (OUTLIERS) ---
print("\n" + "="*60)

# 1. TÍNH TOÁN SAI SỐ TUYỆT ĐỐI VÀ NGƯỠNG
df_results = pd.DataFrame({
    'Actual_Revenue': y_test,
    'Predicted_Revenue': y_pred
})
df_results['Residual'] = df_results['Actual_Revenue'] - df_results['Predicted_Revenue']
df_results['Abs_Residual'] = df_results['Residual'].abs()

# đặt ngưỡng: 2 lần độ lệch chuẩn của sai số tuyệt đối
std_abs_res = df_results['Abs_Residual'].std()
threshold = 2 * std_abs_res

print(f"độ lệch chuẩn của Sai số Tuyệt đối (Std): {std_abs_res:.2f}")
print(f"ngưỡng Sai số Tuyệt đối (Threshold > 2*Std): {threshold:.2f}")


# LỌC CÁC MẪU NẰM NGOÀI ĐƯỜNG CHÉO CHÍNH
outlier_results = df_results[df_results['Abs_Residual'] > threshold]
print(f"\nTổng số mẫu nằm ngoài đường chéo chính (Abs_Residual > {threshold:.2f}): {len(outlier_results)} mẫu")

if not outlier_results.empty:
    # Lấy các Features gốc của các mẫu ngoại lai
    outlier_indices = outlier_results.index
    outlier_features = X_test.loc[outlier_indices]

    outlier_data = pd.concat([outlier_results.sort_values(by='Abs_Residual', ascending=False), outlier_features], axis=1)

    file_name = "outlier_samples_analysis.csv"
    outlier_data.to_csv(file_name, index=True) 

else:
    print("Không tìm thấy mẫu nào vượt quá ngưỡng 2*Std. Mô hình rất chính xác!")

print("="*60 + "\n")


import matplotlib.pyplot as plt

non_outlier_results = df_results[df_results['Abs_Residual'] <= threshold]

plt.figure(figsize=(10, 8)) 

plt.scatter(non_outlier_results['Actual_Revenue'], non_outlier_results['Predicted_Revenue'],
            label='Mẫu dự đoán chính xác', alpha=0.6, s=20) 

if not outlier_results.empty:
    plt.scatter(outlier_results['Actual_Revenue'], outlier_results['Predicted_Revenue'],
                color='red', label=f'Mẫu ngoại lai (> {threshold:.2f} sai số)', alpha=0.8, s=40) 

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', label='Dự đoán hoà hảo (Y=X)')


plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue (Outliers Highlighted)", fontsize=16)
plt.legend() # Hiển thị chú giải
plt.grid(True, linestyle='--', alpha=0.7) # Thêm lưới để dễ đọc
plt.tight_layout() # Điều chỉnh layout để tránh chồng chéo
plt.show()


print("\n" + "="*60)
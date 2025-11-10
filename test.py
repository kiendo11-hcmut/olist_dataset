import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

#  Đọc dữ liệu
df = pd.read_csv("data_cleaning0.csv")

# Giữ lại các cột liên quan
features = [
  "delivery_days", "execution_days",
    "estimated_days", "avg_review_score", "total_freight_value",
    "total_payment", "payment_installments"
]
target = "total_price"

# Kiểm tra có thiếu dữ liệu không
print("Missing values per column:")
print(df[features + [target]].isnull().sum())

# Loại bỏ dòng thiếu dữ liệu
df = df.dropna(subset=features + [target])

# Phân tích tương quan
correlations = df[features + [target]].corr()[target].sort_values(ascending=False)
print("\n Correlation with total_price:")
print(correlations)

# Chọn các feature có |corr| > 0.1 (có tương quan đáng kể)
selected_features = [f for f in features if abs(correlations[f]) > 0.1]
print("\n Selected features:", selected_features)

# Kiểm tra đa cộng tuyến (VIF)
X_vif = df[selected_features].copy()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\n VIF (kiểm tra trùng thông tin giữa các feature):")
print(vif_data)

# Tạo stratification labels
# Tạo bins cho target để stratify
df['target_bin'] = pd.qcut(df[target], q=5, labels=False, duplicates='drop')

# Tạo composite stratification key từ nhiều features quan trọng nhất
# Lấy top 3 features có correlation cao nhất
top_features = correlations[selected_features].abs().nlargest(min(3, len(selected_features))).index.tolist()

# Tạo bins cho từng top feature
for feat in top_features:
    df[f'{feat}_bin'] = pd.qcut(df[feat], q=3, labels=False, duplicates='drop')

# Kết hợp các bins thành một stratification key
strat_cols = ['target_bin'] + [f'{feat}_bin' for feat in top_features]
df['strat_key'] = df[strat_cols].astype(str).agg('_'.join, axis=1)

strat_counts = df['strat_key'].value_counts()
small_groups = strat_counts[strat_counts < 5].index.tolist()

if not small_groups:
    print("Không tìm thấy nhóm quá nhỏ nào.")
else:

    # Tạo danh sách các nhóm hợp lệ (đủ lớn)
    valid_groups = strat_counts[strat_counts >= 5].index.tolist()

    for small_key in small_groups:
        # 1. Tách ra target_bin của nhóm lỗi
        target_bin_of_small_group = small_key.split('_')[0]

        # 2. Tìm các nhóm hợp lệ (valid_groups) có cùng target_bin
        #    Sử dụng 'startswith' để lọc các key
        candidates = [
            key for key in valid_groups 
            if key.startswith(target_bin_of_small_group + '_')
        ]
        
        if candidates:
            # 3. Chọn nhóm lớn nhất trong số các ứng cử viên
            best_merge_key = max(candidates, key=lambda k: strat_counts[k])
            
            # Gán lại các mẫu lỗi vào nhóm tối ưu
            df.loc[df['strat_key'] == small_key, 'strat_key'] = best_merge_key
        else:
            largest_group_key = strat_counts.idxmax()
            df.loc[df['strat_key'] == small_key, 'strat_key'] = largest_group_key

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

# ===============================
# 6️⃣ Huấn luyện mô hình hồi quy tuyến tính
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n Evaluation:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ===============================
# 8️⃣ Xem trọng số (hệ số hồi quy)
# ===============================
coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n🔍 Feature Importance (Linear Regression Coefficients):")
print(coef_df)

# ===============================
# 9️⃣ Xuất tập train và test ra file
# ===============================
# Tạo DataFrame train
train_df = X_train.copy()
train_df[target] = y_train
train_df.to_csv("train_set1.csv", index=False)

# Tạo DataFrame test (có cả giá trị thực và dự đoán)
test_df = X_test.copy()
test_df[target] = y_test
test_df['predicted_' + target] = y_pred
test_df['prediction_error'] = y_test.values - y_pred
test_df.to_csv("test_set1.csv", index=False)


import matplotlib.pyplot as plt

plt.scatter(y_test, model.predict(X_test))
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()


print("\n" + "="*60)
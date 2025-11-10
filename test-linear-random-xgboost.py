import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor

#  Đọc dữ liệu
df = pd.read_csv("data_cleaning0.csv")

features = [
    "order_item_count", "delivery_days", "execution_days",
    "estimated_days", "avg_review_score", "total_freight_value",
    "total_payment", "payment_installments"
]
target = "total_price"

df = df.dropna(subset=features + [target])

#  Phân tích tương quan
correlations = df[features + [target]].corr()[target].sort_values(ascending=False)
print("\n Correlation with total_price:")
print(correlations)

selected_features = [f for f in features if abs(correlations[f]) > 0.1]
print("\n Selected features:", selected_features)

#  Kiểm tra VIF
X_vif = df[selected_features].copy()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\n VIF:")
print(vif_data)

#  Stratified sampling
df['target_bin'] = pd.qcut(df[target], q=5, labels=False, duplicates='drop')
top_features = correlations[selected_features].abs().nlargest(min(3, len(selected_features))).index.tolist()

for feat in top_features:
    df[f'{feat}_bin'] = pd.qcut(df[feat], q=3, labels=False, duplicates='drop')

strat_cols = ['target_bin'] + [f'{feat}_bin' for feat in top_features]
df['strat_key'] = df[strat_cols].astype(str).agg('_'.join, axis=1)

strat_counts = df['strat_key'].value_counts()
small_groups = strat_counts[strat_counts < 5].index.tolist()

if not small_groups:
    print("Không tìm thấy nhóm quá nhỏ nào.")
else:
    print(f"Tìm thấy {len(small_groups)} nhóm quá nhỏ (< 5 mẫu). Đang tiến hành gộp tối ưu...")

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

#  Chia dữ liệu
X = df[selected_features]
y = df[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['strat_key']
    )
    print("\n Sử dụng stratified sampling")
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\n Sử dụng random sampling")

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ===============================
#  Huấn luyện NHIỀU mô hình
# ===============================
models = {}
predictions = {}
results = []

print("\n" + "="*70)
print("🚀 BẮT ĐẦU HUẤN LUYỆN CÁC MÔ HÌNH")
print("="*70)

# Model 1: Linear Regression
print("\n  Huấn luyện Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
models['Linear Regression'] = lr
predictions['Linear Regression'] = lr.predict(X_test)

# Model 2: Random Forest
print("  Huấn luyện Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train) 
models['Random Forest'] = rf
predictions['Random Forest'] = rf.predict(X_test)

# Model 3: XGBoost hoặc GradientBoosting
print(" Huấn luyện XGBoost...")
xgb = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
models['XGBoost'] = xgb
predictions['XGBoost'] = xgb.predict(X_test)

#  So sánh kết quả
print("\n" + "="*70)
print(" KẾT QUẢ SO SÁNH CÁC MÔ HÌNH")
print("="*70)

for model_name, y_pred in predictions.items():
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': model_name,
        'R²': r2,
    })
    
    print(f"\n {model_name}:")
    print(f"   R²: {r2:.4f}")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print("\n" + "="*70)
print(" BẢNG XẾP HẠNG MÔ HÌNH")
print("="*70)
print(results_df.to_string(index=False))

# Chọn mô hình tốt nhất
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\n Mô hình tốt nhất: {best_model_name}")

# Feature Importance của mô hình tốt nhất
print("\n" + "="*70)
print(f" FEATURE IMPORTANCE - {best_model_name}")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance_df.to_string(index=False))
elif hasattr(best_model, 'coef_'):
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': best_model.coef_
    }).sort_values('Coefficient', ascending=False)
    print(importance_df.to_string(index=False))
    
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title(f'Feature Importance - {best_model_name}')
plt.tight_layout()
plt.show()


# Train set
train_df = X_train.copy()
train_df[target] = y_train
train_df.to_csv("train_set_all_models.csv", index=False)

# Test set với predictions của tất cả models
test_df = X_test.copy()
test_df[target] = y_test
for model_name, y_pred in predictions.items():
    test_df[f'pred_{model_name.replace(" ", "_")}'] = y_pred
    test_df[f'error_{model_name.replace(" ", "_")}'] = y_test.values - y_pred

test_df.to_csv("test_set_all_models.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_predictions, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Giá trị thực tế (Actual Total Price)")
plt.ylabel("Giá trị dự đoán (Predicted Total Price)")
plt.title(f"Actual vs Predicted Revenue - {best_model_name}")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(y, bins=50, kde=True)
plt.title("Phân bố tổng doanh thu (total_price)")
plt.show()

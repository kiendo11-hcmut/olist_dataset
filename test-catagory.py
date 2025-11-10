import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

df = pd.read_csv("data_cleaning0.csv")

# Kiểm tra tên cột ngành hàng (có thể là product_category, category_name, etc.)
print("📋 Các cột trong dữ liệu:")
print(df.columns.tolist())

category_col = 'product_category_name'


category_english_col = 'product_category_name_english'


# Target: doanh thu (có thể là total_price, revenue, sales, etc.)
target_col = 'total_price'  # Thay đổi nếu cột doanh thu có tên khác

# ===============================
# 2️⃣ Phân tích doanh thu theo ngành hàng
# ===============================
print("\n" + "="*70)
print("📊 PHÂN TÍCH DOANH THU THEO NGÀNH HÀNG")
print("="*70)

# Loại bỏ dữ liệu thiếu
df_clean = df[[category_col, target_col]].dropna()

# Tạo mapping tên tiếng Anh nếu có
if category_english_col and category_english_col in df.columns:
    # Tạo dictionary mapping từ tên gốc sang tên tiếng Anh
    category_name_map = df[[category_col, category_english_col]].drop_duplicates().set_index(category_col)[category_english_col].to_dict()
else:
    category_name_map = None

# Tổng hợp doanh thu theo ngành
category_stats = df_clean.groupby(category_col).agg({
    target_col: ['count', 'sum', 'mean', 'std', 'min', 'max']
}).round(2)

category_stats.columns = ['Số đơn', 'Tổng doanh thu', 'TB doanh thu', 'Độ lệch chuẩn', 'Min', 'Max']
category_stats = category_stats.sort_values('Tổng doanh thu', ascending=False)

print(f"\n🔝 Top 10 ngành hàng có doanh thu cao nhất:")
print(category_stats.head(10).to_string())

# Lưu thống kê
category_stats.to_csv("revenue_by_category_stats.csv")

# ===============================
# 3️⃣ Chuẩn bị features cho mô hình dự đoán
# ===============================
print("\n" + "="*70)
print("🎯 XÂY DỰNG MÔ HÌNH DỰ ĐOÁN DOANH THU")
print("="*70)

# Chọn features
features = [
    category_col,
    "order_item_count", 
    "delivery_days", 
    "execution_days",
    "estimated_days", 
    "avg_review_score", 
    "total_freight_value",
    "total_payment", 
    "payment_installments"
]

# Kiểm tra features có tồn tại không
available_features = [f for f in features if f in df.columns]
missing_features = [f for f in features if f not in df.columns]

if missing_features:
    print(f"\n⚠️  Các features không tồn tại: {missing_features}")
    print(f"✅ Sử dụng features: {available_features}")
    features = available_features

# Loại bỏ dữ liệu thiếu
df_model = df[features + [target_col]].dropna()
print(f"\n📊 Số lượng dữ liệu sau khi làm sạch: {len(df_model)}")

# Encode category về dạng số
le = LabelEncoder()
df_model[category_col + '_encoded'] = le.fit_transform(df_model[category_col])

# Lưu mapping để decode sau này
category_mapping = pd.DataFrame({
    'category': le.classes_,
    'encoded_value': range(len(le.classes_))
})
category_mapping.to_csv("category_encoding_mapping.csv", index=False)

# Chuẩn bị X, y
numeric_features = [f for f in features if f != category_col]
X = df_model[numeric_features + [category_col + '_encoded']]
y = df_model[target_col]

# Chia train/test theo stratify category
# ===============================
try:
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, df_model[category_col], 
        test_size=0.2, 
        random_state=42,
        stratify=df_model[category_col]
    )
    print("\n✅ Chia dữ liệu theo stratify category")
except ValueError:
    # Nếu có category có quá ít samples
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, df_model[category_col], 
        test_size=0.2, 
        random_state=42
    )
    print("\n⚠️ Một số category có quá ít mẫu, sử dụng random split")

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Kiểm tra phân phối category
print("\n Phân phối ngành hàng trong Train vs Test:")
train_dist = cat_train.value_counts(normalize=True).head(10) * 100
test_dist = cat_test.value_counts(normalize=True).head(10) * 100
dist_compare = pd.DataFrame({
    'Train (%)': train_dist,
    'Test (%)': test_dist
}).round(2)
print(dist_compare)

# ===============================
#  Huấn luyện các mô hình
# ===============================
models = {}
predictions = {}
results = []

print("\n" + "="*70)
print("🚀 HUẤN LUYỆN MÔ HÌNH")
print("="*70)

# Random Forest
print("\n1️⃣  Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
models['Random Forest'] = rf
predictions['Random Forest'] = rf.predict(X_test)

print("2️⃣  XGBoost...")
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
models['XGBoost'] = xgb
predictions['XGBoost'] = xgb.predict(X_test)

# Đánh giá mô hình

print("\n" + "="*70)
print("📊 KẾT QUẢ ĐÁNH GIÁ")
print("="*70)

for model_name, y_pred in predictions.items():
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    
    results.append({
        'Model': model_name,
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape
    })
    
    print(f"\n{model_name}:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)

print("\n" + "="*70)
print(" XẾP HẠNG MÔ HÌNH")
print("="*70)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

#  Feature Importance

print("\n" + "="*70)
print(f"🔍 FEATURE IMPORTANCE - {best_model_name}")
print("="*70)

feature_names = X.columns.tolist()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Thay tên category_encoded về tên gốc
importance_df['Feature'] = importance_df['Feature'].str.replace('_encoded', ' (Category)')
print(importance_df.to_string(index=False))

# ===============================
# Dự đoán theo từng ngành hàng
# ===============================
print("\n" + "="*70)
print("🎯 DỰ ĐOÁN DOANH THU THEO NGÀNH HÀNG")
print("="*70)

# Tạo DataFrame kết quả test
test_results = X_test.copy()
test_results['actual_revenue'] = y_test.values
test_results['predicted_revenue'] = best_predictions
test_results['error'] = y_test.values - best_predictions
test_results['abs_error'] = abs(test_results['error'])
test_results['category'] = cat_test.values

# Tổng hợp theo ngành hàng
category_performance = test_results.groupby('category').agg({
    'actual_revenue': ['count', 'sum', 'mean'],
    'predicted_revenue': ['sum', 'mean'],
    'abs_error': 'mean'
}).round(2)

category_performance.columns = ['Sample Count', 'Actual Revenue', 'Actual Avg', 
                                'Predicted Revenue', 'Predicted Avg', 'Avg Error']
category_performance['Accuracy (%)'] = (100 - (category_performance['Avg Error'] / 
                                        category_performance['Actual Avg'] * 100)).round(2)
category_performance = category_performance.sort_values('Actual Revenue', ascending=False)

print("\n Dự đoán doanh thu theo ngành hàng (Top 10):")
print(category_performance.head(10).to_string())

# Lưu dự đoán theo ngành hàng
category_performance.to_csv("revenue_prediction_by_category.csv")

# Lưu chi tiết test set
test_results.to_csv("test_predictions_detail.csv", index=False)

#  Tìm ngành hàng dự đoán tốt/kém
print("\n" + "="*70)
print("------ Top 5 ngành hàng dự đoán CHÍNH XÁC nhất:")
print("="*70)
best_categories = category_performance.nlargest(5, 'Accuracy (%)')
print(best_categories[['Sample Count', 'Actual Avg', 'Predicted Avg', 'Accuracy (%)']].to_string())

print("\n" + "="*70)
print("------ Top 5 ngành hàng dự đoán KÉM nhất:")
print("="*70)
worst_categories = category_performance.nsmallest(5, 'Accuracy (%)')
print(worst_categories[['Sample Count', 'Actual Avg', 'Predicted Avg', 'Accuracy (%)']].to_string())


# Lấy Top 15 ngành có doanh thu cao nhất để vẽ
top_categories = category_performance.head(10)

# Map tên sang tiếng Anh nếu có
if category_name_map:
    top_categories_display = top_categories.copy()
    top_categories_display.index = top_categories_display.index.map(
        lambda x: category_name_map.get(x, x)[:30]  # Giới hạn 30 ký tự
    )
    display_names = top_categories_display.index
else:
    display_names = top_categories.index

# 1. Biểu đồ so sánh Doanh thu Thực tế vs Dự đoán (Bar Chart)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Phân tích Dự đoán Doanh thu theo Ngành hàng - Model: {best_model_name}', 
             fontsize=16, fontweight='bold')

# Chart 1: Doanh thu trung bình - Thực tế vs Dự đoán
ax1 = axes[0, 0]
x_pos = np.arange(len(top_categories))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, top_categories['Actual Avg'], width, 
                label='Thực tế', alpha=0.8, color='#2E86AB')
bars2 = ax1.bar(x_pos + width/2, top_categories['Predicted Avg'], width,
                label='Dự đoán', alpha=0.8, color='#A23B72')

ax1.set_xlabel('Ngành hàng', fontsize=11, fontweight='bold')
ax1.set_ylabel('Doanh thu trung bình', fontsize=11, fontweight='bold')
ax1.set_title('So sánh Doanh thu TB: Thực tế vs Dự đoán (Top 15)', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Thêm giá trị lên bar
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)

# Chart 2: Tổng doanh thu - Thực tế vs Dự đoán
ax2 = axes[0, 1]
bars3 = ax2.bar(x_pos - width/2, top_categories['Actual Revenue'], width,
                label='Thực tế', alpha=0.8, color='#06A77D')
bars4 = ax2.bar(x_pos + width/2, top_categories['Predicted Revenue'], width,
                label='Dự đoán', alpha=0.8, color='#F18F01')

ax2.set_xlabel('Ngành hàng', fontsize=11, fontweight='bold')
ax2.set_ylabel('Tổng doanh thu', fontsize=11, fontweight='bold')
ax2.set_title('So sánh Tổng doanh thu: Thực tế vs Dự đoán (Top 15)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)


# Chart 4: Độ chính xác theo ngành
ax3 = axes[1, 1]
colors_accuracy = ['#06A77D' if x >= 90 else '#F18F01' if x >= 80 else '#D62828' 
                   for x in top_categories['Accuracy (%)']]
bars5 = ax3.barh(range(len(top_categories)), top_categories['Accuracy (%)'], 
                 color=colors_accuracy, alpha=0.8)

ax3.set_yticks(range(len(top_categories)))
ax3.set_yticklabels(display_names, fontsize=9)
ax3.set_xlabel('Độ chính xác (%)', fontsize=11, fontweight='bold')
ax3.set_title('Độ chính xác dự đoán theo Ngành hàng (Top 15)', 
              fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='Mục tiêu: 90%')
ax3.legend()

# Thêm giá trị
for i, (bar, val) in enumerate(zip(bars5, top_categories['Accuracy (%)'])):
    ax3.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.show()

# 2. Biểu đồ chi tiết sai số theo ngành
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Phân tích Sai số Dự đoán theo Ngành hàng', fontsize=16, fontweight='bold')

# Chart 5: Sai số tuyệt đối trung bình
ax5 = axes2[0]
top_error = category_performance.nlargest(15, 'Avg Error')

# Map tên tiếng Anh
if category_name_map:
    top_error_display = top_error.copy()
    top_error_display.index = top_error_display.index.map(
        lambda x: category_name_map.get(x, x)[:30]
    )
    display_names_error = top_error_display.index
else:
    display_names_error = top_error.index

colors_error = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_error)))
bars6 = ax5.barh(range(len(top_error)), top_error['Avg Error'], color=colors_error, alpha=0.8)

ax5.set_yticks(range(len(top_error)))
ax5.set_yticklabels(display_names_error, fontsize=9)
ax5.set_xlabel('Sai số tuyệt đối TB', fontsize=11, fontweight='bold')
ax5.set_title('Top 15 Ngành có Sai số Lớn nhất', fontsize=12, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars6, top_error['Avg Error'])):
    ax5.text(val + val*0.02, i, f'{val:.0f}', va='center', fontsize=8)

# Chart 6: % Sai số so với doanh thu thực
ax6 = axes2[1]
top_categories_err = category_performance.head(15).copy()

# Map tên tiếng Anh
if category_name_map:
    display_names_pct = [category_name_map.get(x, x)[:30] for x in top_categories_err.index]
else:
    display_names_pct = top_categories_err.index

error_pct = (top_categories_err['Avg Error'] / top_categories_err['Actual Avg'] * 100)
colors_pct = ['#06A77D' if x <= 10 else '#F18F01' if x <= 20 else '#D62828' for x in error_pct]
bars7 = ax6.bar(range(len(top_categories_err)), error_pct, color=colors_pct, alpha=0.8)

ax6.set_xticks(range(len(top_categories_err)))
ax6.set_xticklabels(display_names_pct, rotation=45, ha='right', fontsize=9)
ax6.set_ylabel('% Sai số', fontsize=11, fontweight='bold')
ax6.set_title('% Sai số so với Doanh thu thực (Top 15)', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
ax6.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Mục tiêu: ≤10%')
ax6.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Chấp nhận: ≤20%')
ax6.legend()

for bar, val in zip(bars7, error_pct):
    ax6.text(bar.get_x() + bar.get_width()/2., val + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


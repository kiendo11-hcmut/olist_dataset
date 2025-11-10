import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu test set
test_df = pd.read_csv("test_set_all_models.csv")

# Lấy danh sách các model
model_names = ['Linear_Regression', 'Random_Forest', 'XGBoost']

# Danh sách features để phân tích
features = ['order_item_count', 'delivery_days', 'execution_days',
            'estimated_days', 'avg_review_score', 'total_freight_value',
            'total_payment', 'payment_installments']

print("="*100)
print("📊 PHÂN TÍCH CHI TIẾT: BEST vs WORST vs TRUNG BÌNH")
print("="*100)

# Tính giá trị trung bình của toàn bộ test set
mean_values = {}
for feat in features:
    if feat in test_df.columns:
        mean_values[feat] = test_df[feat].mean()
mean_values['total_price'] = test_df['total_price'].mean()

print("\n📈 GIÁ TRỊ TRUNG BÌNH CỦA TOÀN BỘ TEST SET:")
print("-" * 100)
for feat, val in mean_values.items():
    print(f"   {feat:25s}: {val:>15,.2f}")

# Phân tích cho từng model
all_comparisons = {}

for model_name in model_names:
    pred_col = f'pred_{model_name}'
    error_col = f'error_{model_name}'
    
    # Tính absolute error
    test_df[f'abs_error_{model_name}'] = abs(test_df[error_col])
    
    # Tìm dự đoán chính xác nhất
    best_idx = test_df[f'abs_error_{model_name}'].idxmin()
    best_row = test_df.loc[best_idx]
    
    # Tìm dự đoán sai lệch nhất
    worst_idx = test_df[f'abs_error_{model_name}'].idxmax()
    worst_row = test_df.loc[worst_idx]
    
    # Lưu để so sánh
    all_comparisons[model_name] = {
        'best': best_row,
        'worst': worst_row
    }
    
    # In kết quả chi tiết
    print(f"\n{'='*100}")
    print(f"🤖 MODEL: {model_name.replace('_', ' ').upper()}")
    print(f"{'='*100}")
    
    # Tạo bảng so sánh
    comparison_data = []
    
    print(f"\n{'Feature':<25} {'BEST':<20} {'WORST':<20} {'TRUNG BÌNH':<20} {'So sánh Best':<25}")
    print("-" * 100)
    
    # Total Price
    best_actual = best_row['total_price']
    best_pred = best_row[pred_col]
    best_error = best_row[error_col]
    best_pct = (abs(best_error) / best_actual) * 100
    
    worst_actual = worst_row['total_price']
    worst_pred = worst_row[pred_col]
    worst_error = worst_row[error_col]
    worst_pct = (abs(worst_error) / worst_actual) * 100
    
    print(f"\n{'TOTAL_PRICE (ACTUAL)':<25} {best_actual:<20,.2f} {worst_actual:<20,.2f} {mean_values['total_price']:<20,.2f}")
    print(f"{'TOTAL_PRICE (PREDICTED)':<25} {best_pred:<20,.2f} {worst_pred:<20,.2f} {'-':<20}")
    print(f"{'ERROR':<25} {best_error:<20,.2f} {worst_error:<20,.2f} {'-':<20}")
    print(f"{'ERROR %':<25} {best_pct:<20.2f}% {worst_pct:<20.2f}% {'-':<20}")
    
    print(f"\n{'--- FEATURES ---':<25}")
    print("-" * 100)
    
    comparison_data = []
    for feat in features:
        if feat in test_df.columns:
            best_val = best_row[feat]
            worst_val = worst_row[feat]
            mean_val = mean_values[feat]
            
            # So sánh với trung bình
            best_vs_mean = ((best_val - mean_val) / mean_val * 100) if mean_val != 0 else 0
            worst_vs_mean = ((worst_val - mean_val) / mean_val * 100) if mean_val != 0 else 0
            
            if abs(best_vs_mean) < 10:
                comparison = "≈ Gần trung bình"
            elif best_vs_mean > 0:
                comparison = f"↑ Cao hơn {best_vs_mean:.1f}%"
            else:
                comparison = f"↓ Thấp hơn {abs(best_vs_mean):.1f}%"
            
            print(f"{feat:<25} {best_val:<20,.2f} {worst_val:<20,.2f} {mean_val:<20,.2f} {comparison:<25}")
            
            comparison_data.append({
                'Feature': feat,
                'Best': best_val,
                'Worst': worst_val,
                'Mean': mean_val,
                'Best_vs_Mean_%': best_vs_mean,
                'Worst_vs_Mean_%': worst_vs_mean
            })
    
    # PHÂN TÍCH NGUYÊN NHÂN
    print(f"\n{'='*100}")
    print(f"🔍 PHÂN TÍCH NGUYÊN NHÂN - {model_name.replace('_', ' ').upper()}")
    print(f"{'='*100}")
    
    print(f"\n✅ TẠI SAO DỰ ĐOÁN TỐT (Sai số chỉ {best_pct:.2f}%):")
    print("-" * 100)
    near_mean_count = sum(1 for item in comparison_data if abs(item['Best_vs_Mean_%']) < 20)
    print(f"   • {near_mean_count}/{len(comparison_data)} features gần với giá trị trung bình (±20%)")
    print(f"   • Giá trị total_price = {best_actual:,.0f} đ (Trung bình: {mean_values['total_price']:,.0f} đ)")
    
    # Tìm features gần trung bình nhất
    near_mean_features = [item for item in comparison_data if abs(item['Best_vs_Mean_%']) < 20]
    if near_mean_features:
        print(f"   • Features nằm trong vùng 'an toàn' của model:")
        for item in near_mean_features[:3]:
            print(f"      - {item['Feature']}: {item['Best']:.2f} (Gần {item['Mean']:.2f})")
    
    print(f"\n❌ TẠI SAO DỰ ĐOÁN TỆ (Sai số lên tới {worst_pct:.2f}%):")
    print("-" * 100)
    
    # Kiểm tra outlier
    if worst_actual > test_df['total_price'].quantile(0.95):
        print(f"   • ⚠️  Giá trị total_price = {worst_actual:,.0f} đ là OUTLIER (cao hơn 95% mẫu)")
    elif worst_actual < test_df['total_price'].quantile(0.05):
        print(f"   • ⚠️  Giá trị total_price = {worst_actual:,.0f} đ là OUTLIER (thấp hơn 95% mẫu)")
    
    # Tìm features sai lệch nhiều
    outlier_features = [item for item in comparison_data if abs(item['Worst_vs_Mean_%']) > 50]
    if outlier_features:
        print(f"   • ⚠️  {len(outlier_features)} features bị lệch nhiều so với trung bình:")
        for item in outlier_features:
            direction = "cao hơn" if item['Worst_vs_Mean_%'] > 0 else "thấp hơn"
            print(f"      - {item['Feature']}: {item['Worst']:.2f} ({direction} {abs(item['Worst_vs_Mean_%']):.1f}% so với TB)")
    
    # Phân tích theo từng model
    print(f"\n   • Đặc điểm của {model_name.replace('_', ' ')}:")
    if model_name == 'Linear_Regression':
        print(f"      - Giả định mối quan hệ TUYẾN TÍNH giữa features và target")
        print(f"      - Không xử lý tốt với OUTLIERS và tương tác PHI TUYẾN")
        print(f"      - Hoạt động tốt khi dữ liệu gần với vùng đã học (trung bình)")
    elif model_name == 'Random_Forest':
        print(f"      - Dự đoán bằng cách lấy trung bình của nhiều decision trees")
        print(f"      - KHÓ NGOẠI SUY: Khó dự đoán giá trị nằm ngoài phạm vi training")
        print(f"      - Có thể bị OVERFITTING nếu max_depth quá lớn")
    elif model_name == 'XGBoost':
        print(f"      - Học dần bằng cách sửa lỗi của các cây trước")
        print(f"      - Nhạy cảm với OUTLIERS nếu không có regularization đủ")
        print(f"      - Cần điều chỉnh learning_rate và max_depth phù hợp")

# ==========================================
# VISUALIZATIONS
# ==========================================
print(f"\n{'='*100}")
print("📊 TẠO CÁC BIỂU ĐỒ TRỰC QUAN")
print(f"{'='*100}\n")

# 1. So sánh Best vs Worst vs Mean cho từng model
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('SO SÁNH FEATURES: BEST vs WORST vs TRUNG BÌNH', 
             fontsize=16, fontweight='bold', y=0.995)

for model_idx, model_name in enumerate(model_names):
    best_row = all_comparisons[model_name]['best']
    worst_row = all_comparisons[model_name]['worst']
    
    # Chọn 3 features quan trọng nhất để vẽ
    important_features = ['total_payment', 'total_freight_value', 'order_item_count']
    
    for feat_idx, feat in enumerate(important_features):
        if feat in test_df.columns:
            ax = axes[model_idx, feat_idx]
            
            categories = ['BEST\nPrediction', 'WORST\nPrediction', 'MEAN\n(Test Set)']
            values = [best_row[feat], worst_row[feat], mean_values[feat]]
            colors = ['#2ecc71', '#e74c3c', '#3498db']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Thêm giá trị lên đầu cột
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{model_name.replace("_", " ")}\n{feat}', 
                        fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('features_comparison_best_worst_mean.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu: features_comparison_best_worst_mean.png")
plt.show()

# 2. Radar Chart so sánh tất cả features (chuẩn hóa)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('RADAR CHART: So sánh tất cả Features (Chuẩn hóa 0-100)', 
             fontsize=16, fontweight='bold')

for model_idx, model_name in enumerate(model_names):
    ax = axes[model_idx]
    
    best_row = all_comparisons[model_name]['best']
    worst_row = all_comparisons[model_name]['worst']
    
    # Chuẩn hóa features về 0-100
    normalized_features = []
    feature_names_short = []
    
    for feat in features[:6]:  # Chỉ lấy 6 features để radar chart dễ nhìn
        if feat in test_df.columns:
            min_val = test_df[feat].min()
            max_val = test_df[feat].max()
            
            if max_val - min_val > 0:
                best_norm = (best_row[feat] - min_val) / (max_val - min_val) * 100
                worst_norm = (worst_row[feat] - min_val) / (max_val - min_val) * 100
                mean_norm = (mean_values[feat] - min_val) / (max_val - min_val) * 100
                
                normalized_features.append({
                    'best': best_norm,
                    'worst': worst_norm,
                    'mean': mean_norm
                })
                # Rút ngắn tên feature
                short_name = feat.replace('_', ' ').title()[:15]
                feature_names_short.append(short_name)
    
    # Vẽ radar chart
    angles = np.linspace(0, 2 * np.pi, len(normalized_features), endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng tròn
    
    best_values = [f['best'] for f in normalized_features]
    worst_values = [f['worst'] for f in normalized_features]
    mean_values_norm = [f['mean'] for f in normalized_features]
    
    best_values += best_values[:1]
    worst_values += worst_values[:1]
    mean_values_norm += mean_values_norm[:1]
    
    ax = plt.subplot(1, 3, model_idx + 1, projection='polar')
    
    ax.plot(angles, best_values, 'o-', linewidth=2, label='Best', color='#2ecc71')
    ax.fill(angles, best_values, alpha=0.15, color='#2ecc71')
    
    ax.plot(angles, worst_values, 'o-', linewidth=2, label='Worst', color='#e74c3c')
    ax.fill(angles, worst_values, alpha=0.15, color='#e74c3c')
    
    ax.plot(angles, mean_values_norm, 'o-', linewidth=2, label='Mean', color='#3498db')
    ax.fill(angles, mean_values_norm, alpha=0.15, color='#3498db')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names_short, size=8)
    ax.set_ylim(0, 100)
    ax.set_title(f'{model_name.replace("_", " ")}', size=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

plt.tight_layout()
plt.savefig('radar_chart_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu: radar_chart_comparison.png")
plt.show()

# 3. Heatmap: % chênh lệch so với trung bình
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for comparison_type, ax in zip(['Best', 'Worst'], axes):
    heatmap_data = []
    
    for model_name in model_names:
        row_data = []
        comp_row = all_comparisons[model_name]['best' if comparison_type == 'Best' else 'worst']
        
        for feat in features:
            if feat in test_df.columns:
                val = comp_row[feat]
                mean_val = mean_values[feat]
                pct_diff = ((val - mean_val) / mean_val * 100) if mean_val != 0 else 0
                row_data.append(pct_diff)
            else:
                row_data.append(0)
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(
        heatmap_data,
        columns=[f.replace('_', ' ').title()[:15] for f in features],
        index=[m.replace('_', ' ') for m in model_names]
    )
    
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': '% Chênh lệch so với TB'},
                linewidths=0.5, ax=ax, vmin=-100, vmax=100)
    
    ax.set_title(f'{comparison_type} Predictions: % Chênh lệch so với Trung bình', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Model', fontsize=11)

plt.tight_layout()
plt.savefig('heatmap_deviation_from_mean.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu: heatmap_deviation_from_mean.png")
plt.show()

# 4. Biểu đồ tổng hợp: Tỷ lệ % error
fig, ax = plt.subplots(figsize=(12, 6))

model_labels = [m.replace('_', ' ') for m in model_names]
best_errors = []
worst_errors = []

for model_name in model_names:
    best_row = all_comparisons[model_name]['best']
    worst_row = all_comparisons[model_name]['worst']
    
    pred_col = f'pred_{model_name}'
    
    best_error_pct = abs(best_row['total_price'] - best_row[pred_col]) / best_row['total_price'] * 100
    worst_error_pct = abs(worst_row['total_price'] - worst_row[pred_col]) / worst_row['total_price'] * 100
    
    best_errors.append(best_error_pct)
    worst_errors.append(worst_error_pct)

x = np.arange(len(model_labels))
width = 0.35

bars1 = ax.bar(x - width/2, best_errors, width, label='Best Prediction Error %', 
               color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, worst_errors, width, label='Worst Prediction Error %', 
               color='#e74c3c', alpha=0.8, edgecolor='black')

# Thêm giá trị lên đầu cột
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Prediction Error (%)', fontsize=12, fontweight='bold')
ax.set_title('So sánh Tỷ lệ % Sai số: Best vs Worst Predictions', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('error_percentage_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu: error_percentage_comparison.png")
plt.show()

print(f"\n{'='*100}")
print("✅ HOÀN THÀNH! Đã tạo 4 biểu đồ phân tích chi tiết.")
print(f"{'='*100}\n")

print("📋 TÓM TẮT KẾT LUẬN:")
print("-" * 100)
print("1. ✅ Dự đoán TỐT khi: Features gần với giá trị trung bình của training set")
print("2. ❌ Dự đoán TỆ khi: Gặp outliers hoặc giá trị bất thường")
print("3. 🎯 Model tốt nhất: Xem biểu đồ % error để chọn model có worst error thấp nhất")
print("4. 💡 Cải thiện: Thu thập thêm dữ liệu outliers hoặc xử lý outliers trước khi train")
print("-" * 100)
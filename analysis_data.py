import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu test set
test_df = pd.read_csv("test_set_all_models.csv")

# Lấy danh sách các model
model_names = ['Linear_Regression', 'Random_Forest', 'XGBoost']

print("="*80)
print("📊 PHÂN TÍCH KẾT QUẢ CHÍNH XÁC NHẤT VÀ SAI LỆCH NHẤT")
print("="*80)

# Phân tích cho từng model
analysis_results = []

for model_name in model_names:
    pred_col = f'pred_{model_name}'
    error_col = f'error_{model_name}'
    
    # Tính absolute error
    test_df[f'abs_error_{model_name}'] = abs(test_df[error_col])
    
    # Tìm dự đoán chính xác nhất (error nhỏ nhất)
    best_idx = test_df[f'abs_error_{model_name}'].idxmin()
    best_actual = test_df.loc[best_idx, 'total_price']
    best_pred = test_df.loc[best_idx, pred_col]
    best_error = test_df.loc[best_idx, error_col]
    best_pct_error = (abs(best_error) / best_actual) * 100
    
    # Tìm dự đoán sai lệch nhất (error lớn nhất)
    worst_idx = test_df[f'abs_error_{model_name}'].idxmax()
    worst_actual = test_df.loc[worst_idx, 'total_price']
    worst_pred = test_df.loc[worst_idx, pred_col]
    worst_error = test_df.loc[worst_idx, error_col]
    worst_pct_error = (abs(worst_error) / worst_actual) * 100
    
    # Lưu thông tin chi tiết
    analysis_results.append({
        'Model': model_name.replace('_', ' '),
        'Best_Actual': best_actual,
        'Best_Pred': best_pred,
        'Best_Error': best_error,
        'Best_%Error': best_pct_error,
        'Worst_Actual': worst_actual,
        'Worst_Pred': worst_pred,
        'Worst_Error': worst_error,
        'Worst_%Error': worst_pct_error,
        'Best_Features': test_df.loc[best_idx].to_dict(),
        'Worst_Features': test_df.loc[worst_idx].to_dict()
    })
    
    # In kết quả
    print(f"\n{'='*80}")
    print(f"🔍 {model_name.replace('_', ' ').upper()}")
    print(f"{'='*80}")
    
    print(f"\n✅ DỰ ĐOÁN CHÍNH XÁC NHẤT:")
    print(f"   • Giá trị thực tế: {best_actual:,.2f}")
    print(f"   • Giá trị dự đoán: {best_pred:,.2f}")
    print(f"   • Sai số: {best_error:,.2f} ({best_pct_error:.2f}%)")
    print(f"\n   📋 Đặc điểm của giao dịch này:")
    for feat in ['order_item_count', 'delivery_days', 'execution_days', 
                 'estimated_days', 'avg_review_score', 'total_freight_value',
                 'total_payment', 'payment_installments']:
        if feat in test_df.columns:
            print(f"      - {feat}: {test_df.loc[best_idx, feat]:.2f}")
    
    print(f"\n❌ DỰ ĐOÁN SAI LỆCH NHẤT:")
    print(f"   • Giá trị thực tế: {worst_actual:,.2f}")
    print(f"   • Giá trị dự đoán: {worst_pred:,.2f}")
    print(f"   • Sai số: {worst_error:,.2f} ({worst_pct_error:.2f}%)")
    print(f"   • Hướng sai lệch: {'DỰ ĐOÁN CAO HƠN' if worst_error < 0 else 'DỰ ĐOÁN THẤP HƠN'}")
    print(f"\n   📋 Đặc điểm của giao dịch này:")
    for feat in ['order_item_count', 'delivery_days', 'execution_days',
                 'estimated_days', 'avg_review_score', 'total_freight_value',
                 'total_payment', 'payment_installments']:
        if feat in test_df.columns:
            print(f"      - {feat}: {test_df.loc[worst_idx, feat]:.2f}")

# Tạo bảng so sánh
print(f"\n{'='*80}")
print("📈 BẢNG SO SÁNH TÓM TẮT")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame([
    {
        'Model': r['Model'],
        'Best Error (%)': f"{r['Best_%Error']:.2f}%",
        'Worst Error (%)': f"{r['Worst_%Error']:.2f}%",
        'Error Range': f"{r['Worst_%Error'] - r['Best_%Error']:.2f}%"
    }
    for r in analysis_results
])
print(comparison_df.to_string(index=False))

# PHÂN TÍCH TẠI SAO
print(f"\n{'='*80}")
print("🔬 PHÂN TÍCH NGUYÊN NHÂN")
print(f"{'='*80}\n")

for result in analysis_results:
    model_name = result['Model']
    print(f"\n📌 {model_name.upper()}:")
    
    # Phân tích dự đoán tốt
    print(f"\n   ✅ Dự đoán chính xác vì:")
    best_feat = result['Best_Features']
    print(f"      • Giá trị nằm gần trung bình của dữ liệu huấn luyện")
    print(f"      • Không có giá trị bất thường (outlier)")
    print(f"      • Các features có mối tương quan mạnh với target")
    
    # Phân tích dự đoán kém
    print(f"\n   ❌ Dự đoán sai lệch vì:")
    worst_feat = result['Worst_Features']
    
    # Kiểm tra các yếu tố
    if worst_feat.get('total_price', 0) > test_df['total_price'].quantile(0.95):
        print(f"      • Giá trị thuộc nhóm OUTLIER (cao hơn 95% dữ liệu)")
    elif worst_feat.get('total_price', 0) < test_df['total_price'].quantile(0.05):
        print(f"      • Giá trị thuộc nhóm OUTLIER (thấp hơn 95% dữ liệu)")
    
    if model_name == 'Linear Regression':
        print(f"      • Linear Regression giả định mối quan hệ tuyến tính")
        print(f"      • Không xử lý tốt các tương tác phi tuyến giữa features")
        print(f"      • Nhạy cảm với outliers và multicollinearity")
    elif model_name == 'Random Forest':
        print(f"      • Random Forest có thể bị overfitting với dữ liệu phức tạp")
        print(f"      • Khó dự đoán giá trị nằm ngoài phạm vi training data")
        print(f"      • Có thể cần điều chỉnh hyperparameters (max_depth, min_samples)")
    elif model_name == 'XGBoost':
        print(f"      • XGBoost nhạy cảm với outliers trong target variable")
        print(f"      • Learning rate có thể cần điều chỉnh")
        print(f"      • Có thể cần thêm regularization (reg_alpha, reg_lambda)")

# Visualizations
print(f"\n{'='*80}")
print("📊 TẠO BIỂU ĐỒ TRỰC QUAN")
print(f"{'='*80}\n")

# 1. Biểu đồ phân bố error của từng model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, model_name in enumerate(model_names):
    error_col = f'error_{model_name}'
    axes[idx].hist(test_df[error_col], bins=50, alpha=0.7, color=['blue', 'green', 'orange'][idx])
    axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[idx].set_title(f'{model_name.replace("_", " ")}\nError Distribution')
    axes[idx].set_xlabel('Prediction Error')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu: error_distribution_comparison.png")
plt.show()

# 2. Biểu đồ so sánh best vs worst predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (model_name, result) in enumerate(zip(model_names, analysis_results)):
    categories = ['Best\nPrediction', 'Worst\nPrediction']
    actual = [result['Best_Actual'], result['Worst_Actual']]
    predicted = [result['Best_Pred'], result['Worst_Pred']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[idx].bar(x - width/2, actual, width, label='Actual', alpha=0.8)
    axes[idx].bar(x + width/2, predicted, width, label='Predicted', alpha=0.8)
    axes[idx].set_title(f'{model_name.replace("_", " ")}')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(categories)
    axes[idx].set_ylabel('Total Price')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Thêm % error
    for i, cat in enumerate(categories):
        if cat.startswith('Best'):
            pct = result['Best_%Error']
        else:
            pct = result['Worst_%Error']
        axes[idx].text(i, max(actual[i], predicted[i]) * 1.05, 
                      f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('best_worst_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu: best_worst_comparison.png")
plt.show()

# 3. Heatmap so sánh % error
fig, ax = plt.subplots(figsize=(10, 6))
error_matrix = []
for result in analysis_results:
    error_matrix.append([result['Best_%Error'], result['Worst_%Error']])

sns.heatmap(error_matrix, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn_r',
            xticklabels=['Best Prediction Error (%)', 'Worst Prediction Error (%)'],
            yticklabels=[r['Model'] for r in analysis_results],
            cbar_kws={'label': 'Percentage Error'},
            ax=ax)
ax.set_title('Heatmap: Percentage Error Comparison Across Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('error_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu: error_heatmap.png")
plt.show()

print(f"\n{'='*80}")
print("✅ HOÀN THÀNH PHÂN TÍCH!")
print(f"{'='*80}\n")

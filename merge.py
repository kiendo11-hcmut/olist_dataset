import pandas as pd

# === 1️⃣ Đọc dữ liệu ===
data_cleaning = pd.read_csv("data_cleaning.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
translation = pd.read_csv("product_category_name_translation.csv")

# === 2️⃣ Chuẩn hóa tên cột ===
for df in [data_cleaning, order_items, products, translation]:
    df.columns = df.columns.str.strip()

# === 3️⃣ Merge tuần tự (không drop cột gốc) ===
# Merge order_id -> product_id
merged = pd.merge(
    data_cleaning, 
    order_items[['order_id','product_id']], 
    on='order_id', 
    how='left'
)

# Merge product_id -> product_category_name
merged = pd.merge(
    merged, 
    products[['product_id','product_category_name']], 
    on='product_id', 
    how='left'
)

# Merge product_category_name -> product_category_name_english
translation_unique = translation.drop_duplicates(subset=['product_category_name'])
merged = pd.merge(
    merged, 
    translation_unique[['product_category_name','product_category_name_english']],
    on='product_category_name', 
    how='left'
)

# === 4️⃣ Gộp các category theo order_id để tránh nhân dòng ===
merged_grouped = merged.groupby('order_id').agg({
    **{col: 'first' for col in data_cleaning.columns if col != 'order_id'},
    'product_id': list,
    'product_category_name': list,
    'product_category_name_english': list
}).reset_index()

# Chuyển list → chuỗi duy nhất chỉ 1 lần
for col in ['product_id','product_category_name','product_category_name_english']:
    merged_grouped[col] = merged_grouped[col].apply(lambda x: ','.join(sorted(set(map(str,x)))))
# === 5️⃣ Lưu kết quả ===
merged_grouped.to_csv("data_cleaning0.csv", index=False, encoding="utf-8-sig")

print("✅ Merge hoàn tất, giữ nguyên các cột cũ và thêm các cột mới")
print("Tổng số dòng:", len(merged_grouped))
print("\n📋 5 dòng đầu:")
print(merged_grouped.head())

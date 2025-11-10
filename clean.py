import pandas as pd
import numpy as np

# === 1️⃣ Đọc dữ liệu ===
df = pd.read_csv("data_cleaning0.csv")

# === 2️⃣ Chuyển đổi cột thời gian ===
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# === 3️⃣ Kiểm tra và loại bỏ dữ liệu sai logic thời gian ===
# Giao hàng không thể sớm hơn ngày đặt hàng hoặc ngày duyệt
df = df[df["order_delivered_customer_date"] >= df["order_purchase_timestamp"]]
df = df[df["order_delivered_carrier_date"] >= df["order_approved_at"]]

# === 4️⃣ Loại bỏ ngoại lai bằng IQR (cho các cột số) ===
numeric_cols = [
    "delivery_days", "execution_days", "estimated_days",
    "avg_review_score", "total_price",
    "total_freight_value", "total_payment", "payment_installments"
]

def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

# === 5️⃣ Kiểm tra logic giá trị thanh toán ===
# total_payment phải >= total_price + total_freight_value
df = df[df["total_payment"] >= df["total_price"] + df["total_freight_value"]]

# === 6️⃣ Chuẩn hóa dữ liệu âm hoặc 0 không hợp lý ===
df = df[df["total_price"] > 0]
df = df[df["total_freight_value"] >= 0]
df = df[df["delivery_days"] >= 0]

# === 7️⃣ Làm mượt các giá trị quá lớn (Winsorizing nhẹ) ===
for col in ["total_price", "total_freight_value", "total_payment"]:
    upper = df[col].quantile(0.99)
    df[col] = np.where(df[col] > upper, upper, df[col])

# === 8️⃣ Xuất dữ liệu sạch ===
df.to_csv("data_cleaned0_olist.csv", index=False)

print("✅ Hoàn tất làm sạch dữ liệu!")
print("➡️ Số dòng sau khi xử lý:", len(df))

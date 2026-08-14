import pandas as pd
import os

# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "data/E Commerce Dataset-E Comm.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)

# ==========================================
# 2. Create Outputs Folder
# ==========================================

os.makedirs("outputs", exist_ok=True)

# ==========================================
# 3. Handle Missing Values
# ==========================================

df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())
df["OrderCount"] = df["OrderCount"].fillna(df["OrderCount"].median())
df["DaySinceLastOrder"] = df["DaySinceLastOrder"].fillna(
    df["DaySinceLastOrder"].median()
)

# ==========================================
# 4. Order Frequency by Customer Segment
# ==========================================

order_frequency = (
    df.groupby("Churn")["OrderCount"]
    .agg(["count", "mean", "sum"])
    .reset_index()
)

order_frequency.columns = [
    "Churn",
    "CustomerCount",
    "AverageOrderCount",
    "TotalOrders"
]

print("\n==========================================")
print("ORDER FREQUENCY BY CHURN SEGMENT")
print("==========================================")

print(order_frequency)

# ==========================================
# 5. Tenure Analysis
# ==========================================

tenure_analysis = (
    df.groupby("Churn")["Tenure"]
    .agg(["count", "mean", "min", "max"])
    .reset_index()
)

tenure_analysis.columns = [
    "Churn",
    "CustomerCount",
    "AverageTenure",
    "MinimumTenure",
    "MaximumTenure"
]

print("\n==========================================")
print("TENURE ANALYSIS")
print("==========================================")

print(tenure_analysis)

# ==========================================
# 6. Days Since Last Order Analysis
# ==========================================

last_order_analysis = (
    df.groupby("Churn")["DaySinceLastOrder"]
    .agg(["count", "mean", "min", "max"])
    .reset_index()
)

last_order_analysis.columns = [
    "Churn",
    "CustomerCount",
    "AverageDaysSinceLastOrder",
    "MinimumDays",
    "MaximumDays"
]

print("\n==========================================")
print("DAYS SINCE LAST ORDER ANALYSIS")
print("==========================================")

print(last_order_analysis)

# ==========================================
# 7. Customer Activity Growth Metric
# ==========================================

total_customers = len(df)

active_customers = len(
    df[df["OrderCount"] > 0]
)

inactive_customers = total_customers - active_customers

active_user_rate = (
    active_customers / total_customers
) * 100

print("\n==========================================")
print("CUSTOMER ACTIVITY METRICS")
print("==========================================")

print("Total Customers:", total_customers)
print("Active Customers:", active_customers)
print("Inactive Customers:", inactive_customers)
print("Active User Rate: {:.2f}%".format(active_user_rate))

# ==========================================
# 8. Rolling Behavioral Average
# ==========================================

df_sorted = df.sort_values("Tenure").copy()

df_sorted["RollingOrderAverage"] = (
    df_sorted["OrderCount"]
    .rolling(window=10, min_periods=1)
    .mean()
)

rolling_summary = df_sorted[
    ["Tenure", "OrderCount", "RollingOrderAverage"]
].tail(20)

print("\n==========================================")
print("ROLLING ORDER BEHAVIOR")
print("==========================================")

print(rolling_summary)

# ==========================================
# 9. Save Analytics Outputs
# ==========================================

order_frequency.to_csv(
    "outputs/order_frequency_by_churn.csv",
    index=False
)

tenure_analysis.to_csv(
    "outputs/tenure_analysis.csv",
    index=False
)

last_order_analysis.to_csv(
    "outputs/last_order_analysis.csv",
    index=False
)

rolling_summary.to_csv(
    "outputs/rolling_order_behavior.csv",
    index=False
)

print("\n==========================================")
print("DAY 5 ANALYTICS COMPLETED SUCCESSFULLY!")
print("Analytics files saved inside outputs/ folder.")
print("==========================================")
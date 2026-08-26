import pandas as pd
from ydata_profiling import ProfileReport

# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "Data/E Commerce Dataset-E Comm.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# ==========================================
# 2. Display Basic Information
# ==========================================

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 3. Generate EDA Profile Report
# ==========================================

profile = ProfileReport(
    df,
    title="E-Commerce Customer Churn - EDA Report",
    explorative=True
)

# ==========================================
# ==========================================
# 4. Save HTML Report
# ==========================================

profile.to_file("outputs/churn_profile_report.html")

print("\nEDA Report generated successfully!")
print("File: outputs/churn_profile_report.html")

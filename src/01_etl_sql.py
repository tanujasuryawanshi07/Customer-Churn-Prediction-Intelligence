import pandas as pd
import numpy as np
import sqlite3

# -----------------------------
# 1. Load and Validate Dataset
# -----------------------------

file_path = "data/E Commerce Dataset-E Comm.csv"

try:
    df = pd.read_csv(file_path)

    print("========== DATASET VALIDATION ==========")
    print("Dataset Loaded Successfully!")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

except Exception as e:
    print("Error loading dataset:", e)
    exit()

# -----------------------------
# 2. NumPy Processing
# -----------------------------

print("\n========== NUMPY PROCESSING ==========")

numeric_cols = ["Tenure", "OrderCount"]

for col in numeric_cols:
    if col in df.columns:
        arr = df[col].to_numpy()

        print(f"\nStatistics for {col}")
        print("Mean:", np.nanmean(arr))
        print("Maximum:", np.nanmax(arr))
    else:
        print(f"{col} column not found.")

# -----------------------------
# 3. SQLite Database Simulation
# -----------------------------

print("\n========== SQLITE ANALYSIS ==========")

conn = sqlite3.connect(":memory:")

df.to_sql("customer_churn", conn, index=False, if_exists="replace")

query = """
SELECT
    CityTier,
    COUNT(*) AS Total_Customers,
    ROUND(AVG(Tenure),2) AS Average_Tenure,
    SUM(Churn) AS Total_Churn
FROM customer_churn
GROUP BY CityTier
HAVING COUNT(*) > 0;
"""

result = pd.read_sql_query(query, conn)

print(result)

conn.close()

print("\nProgram Executed Successfully!")
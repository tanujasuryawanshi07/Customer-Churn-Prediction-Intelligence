import pandas as pd
import sqlite3

# ==========================================
# 1. Load Main Customer Dataset
# ==========================================

file_path = "data/E Commerce Dataset-E Comm.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)

# ==========================================
# 2. Create CustomerID
# ==========================================

df["CustomerID"] = range(1, len(df) + 1)

print("\nCustomerID Created Successfully!")

# ==========================================
# 3. Create Secondary Table
# ==========================================

feedback_df = pd.DataFrame({
    "CustomerID": df["CustomerID"],
    "FeedbackScore": (df.index % 5) + 1,
    "PreferredOffers": (
        ["Discount", "Cashback", "Free Shipping", "Coupon", "Loyalty Points"]
        * ((len(df) + 4) // 5)
    )[:len(df)]
})

# Make sure secondary table has same number of rows
feedback_df = feedback_df.iloc[:len(df)]

print("\nSecondary Table Created Successfully!")
print(feedback_df.head())

# ==========================================
# 4. Create SQLite Database
# ==========================================

conn = sqlite3.connect(":memory:")

# ==========================================
# 5. Push Tables into SQLite
# ==========================================

df.to_sql("customers", conn, index=False, if_exists="replace")

feedback_df.to_sql(
    "customer_feedback",
    conn,
    index=False,
    if_exists="replace"
)

print("\nBoth tables inserted into SQLite successfully!")

# ==========================================
# 6. SQL INNER JOIN
# ==========================================

query = """
SELECT
    c.CustomerID,
    c.Churn,
    c.CashbackAmount,
    f.FeedbackScore,
    f.PreferredOffers
FROM customers c
INNER JOIN customer_feedback f
ON c.CustomerID = f.CustomerID
"""

joined_df = pd.read_sql_query(query, conn)

print("\n==========================================")
print("JOINED DATA")
print("==========================================")

print(joined_df.head(10))

# ==========================================
# 7. Business Insight
# ==========================================

insight_query = """
SELECT
    f.FeedbackScore,
    COUNT(*) AS TotalCustomers,
    ROUND(AVG(c.Churn), 2) AS AverageChurn,
    ROUND(AVG(c.CashbackAmount), 2) AS AverageCashback
FROM customers c
INNER JOIN customer_feedback f
ON c.CustomerID = f.CustomerID
GROUP BY f.FeedbackScore
ORDER BY f.FeedbackScore
"""

insight_df = pd.read_sql_query(insight_query, conn)

print("\n==========================================")
print("BUSINESS INSIGHTS")
print("==========================================")

print(insight_df)

# ==========================================
# 8. Validation
# ==========================================

print("\n==========================================")
print("VALIDATION")
print("==========================================")

print("Main table rows:", len(df))
print("Secondary table rows:", len(feedback_df))
print("Joined table rows:", len(joined_df))

print("\nJoined Columns:")
print(joined_df.columns.tolist())

# ==========================================
# 9. Close Database
# ==========================================

conn.close()

print("\nDatabase connection closed.")
print("DAY 2 TASK COMPLETED SUCCESSFULLY!")

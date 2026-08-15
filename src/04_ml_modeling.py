import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold

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

numeric_columns = df.select_dtypes(include=["number"]).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

# ==========================================
# 4. Feature Engineering
# ==========================================

# Avoid division by zero
df["Complaint_to_Order_Ratio"] = (
    df["Complain"] / (df["OrderCount"] + 1)
)

df["Average_Cashback_Per_Order"] = (
    df["CashbackAmount"] / (df["OrderCount"] + 1)
)

# Engagement score based on app usage and devices
df["Engagement_Score"] = (
    df["HourSpendOnApp"]
    + df["NumberOfDeviceRegistered"]
)

# Customer activity score
df["Customer_Activity_Score"] = (
    df["OrderCount"]
    + df["CouponUsed"]
)

print("\n==========================================")
print("ENGINEERED FEATURES")
print("==========================================")

print(
    df[
        [
            "Complaint_to_Order_Ratio",
            "Average_Cashback_Per_Order",
            "Engagement_Score",
            "Customer_Activity_Score"
        ]
    ].head()
)

# ==========================================
# 5. Select Numeric Features
# ==========================================

numeric_df = df.select_dtypes(include=["number"]).copy()

# Remove target from feature set
X = numeric_df.drop(columns=["Churn"])

y = numeric_df["Churn"]

# ==========================================
# 6. Variance Threshold
# ==========================================

selector = VarianceThreshold(threshold=0.01)

X_variance = selector.fit_transform(X)

selected_variance_columns = X.columns[
    selector.get_support()
]

X_variance_df = pd.DataFrame(
    X_variance,
    columns=selected_variance_columns
)

print("\n==========================================")
print("VARIANCE THRESHOLD")
print("==========================================")

print("Original Numeric Features:", X.shape[1])
print("After Variance Filtering:", X_variance_df.shape[1])

# ==========================================
# 7. Pearson Correlation with Churn
# ==========================================

correlation_data = X_variance_df.copy()

correlation_data["Churn"] = y.values

correlation_with_churn = (
    correlation_data.corr()["Churn"]
    .drop("Churn")
    .sort_values(key=abs, ascending=False)
)

print("\n==========================================")
print("PEARSON CORRELATION WITH CHURN")
print("==========================================")

print(correlation_with_churn)

# ==========================================
# 8. Select Features Based on Correlation
# ==========================================

correlation_threshold = 0.05

selected_correlation_features = (
    correlation_with_churn[
        correlation_with_churn.abs() >= correlation_threshold
    ]
    .index
    .tolist()
)

selected_features = X_variance_df[
    selected_correlation_features
].copy()

# ==========================================
# 9. Create Final Feature Matrix
# ==========================================

selected_features["Churn"] = y.values

print("\n==========================================")
print("FINAL SELECTED FEATURES")
print("==========================================")

print(
    "Number of Selected Features:",
    len(selected_correlation_features)
)

print("\nSelected Feature Names:")

for feature in selected_correlation_features:
    print("-", feature)

# ==========================================
# 10. Save Feature Matrix
# ==========================================

selected_features.to_csv(
    "outputs/selected_features.csv",
    index=False
)

print("\nSaved: outputs/selected_features.csv")

# ==========================================
# 11. Save Correlation Results
# ==========================================

correlation_with_churn.to_csv(
    "outputs/feature_churn_correlations.csv",
    header=["CorrelationWithChurn"]
)

print(
    "Saved: outputs/feature_churn_correlations.csv"
)

# ==========================================
# 12. Completion Message
# ==========================================

print("\n==========================================")
print("DAY 7 FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("Feature selection and filter methods completed.")
print("==========================================")
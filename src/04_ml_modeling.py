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


# ==========================================
# DAY 8 - SIMPLE LINEAR REGRESSION
# BINARY CLASSIFICATION
# ==========================================

import numpy as np

print("\n==========================================")
print("DAY 8 - LINEAR REGRESSION CLASSIFICATION")
print("==========================================")

# ------------------------------------------
# 1. Prepare Predictor and Target
# ------------------------------------------

model_df = df[
    ["DaySinceLastOrder", "Churn"]
].copy()

model_df["DaySinceLastOrder"] = model_df[
    "DaySinceLastOrder"
].fillna(
    model_df["DaySinceLastOrder"].median()
)

X = model_df["DaySinceLastOrder"].to_numpy(
    dtype=float
)

y = model_df["Churn"].to_numpy(
    dtype=float
)

# ------------------------------------------
# 2. Add Intercept Column
# ------------------------------------------

X_matrix = np.column_stack(
    [
        np.ones(len(X)),
        X
    ]
)

# ------------------------------------------
# 3. Closed-Form Linear Regression
#    w = (X^T X)^-1 X^T y
# ------------------------------------------

X_transpose = X_matrix.T

weights = np.linalg.inv(
    X_transpose @ X_matrix
) @ X_transpose @ y

print("Model Weights:")
print(weights)

# ------------------------------------------
# 4. Generate Continuous Predictions
# ------------------------------------------

predictions = X_matrix @ weights

# ------------------------------------------
# 5. Apply 0.5 Classification Threshold
# ------------------------------------------

predicted_classes = (
    predictions >= 0.5
).astype(int)

# ------------------------------------------
# 6. Confusion Matrix Calculation
# ------------------------------------------

true_positive = np.sum(
    (y == 1) & (predicted_classes == 1)
)

true_negative = np.sum(
    (y == 0) & (predicted_classes == 0)
)

false_positive = np.sum(
    (y == 0) & (predicted_classes == 1)
)

false_negative = np.sum(
    (y == 1) & (predicted_classes == 0)
)

# ------------------------------------------
# 7. Accuracy
# ------------------------------------------

accuracy = (
    true_positive + true_negative
) / len(y)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print("True Positive :", true_positive)
print("True Negative :", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)

print("\nClassification Accuracy: {:.2f}%".format(
    accuracy * 100
))

# ------------------------------------------
# 8. Save Predictions
# ------------------------------------------

model_results = model_df.copy()

model_results["PredictedScore"] = predictions
model_results["PredictedChurn"] = predicted_classes

model_results.to_csv(
    "outputs/linear_regression_predictions.csv",
    index=False
)

print(
    "\nSaved: outputs/linear_regression_predictions.csv"
)

print("\n==========================================")
print("DAY 8 LINEAR REGRESSION COMPLETED!")
print("==========================================")

# ==========================================
# DAY 9 - A/B TESTING SIMULATION
# ==========================================

import numpy as np

print("\n==========================================")
print("DAY 9 - A/B TESTING SIMULATION")
print("==========================================")

# ------------------------------------------
# 1. Create a copy of customer data
# ------------------------------------------

ab_df = df.copy()

# ------------------------------------------
# 2. Randomly assign customers to groups
# ------------------------------------------

np.random.seed(42)

ab_df["Group"] = np.random.choice(
    ["Control", "Variant"],
    size=len(ab_df)
)

# ------------------------------------------
# 3. Define Retention
#    Churn = 0 means retained customer
# ------------------------------------------

ab_df["Retained"] = (
    ab_df["Churn"] == 0
).astype(int)

# ------------------------------------------
# 4. Calculate Group Statistics
# ------------------------------------------

group_summary = (
    ab_df.groupby("Group")
    .agg(
        TotalCustomers=("CustomerID", "count"),
        RetainedCustomers=("Retained", "sum"),
        ChurnedCustomers=("Churn", "sum")
    )
    .reset_index()
)

group_summary["RetentionRate"] = (
    group_summary["RetainedCustomers"]
    / group_summary["TotalCustomers"]
) * 100

group_summary["ChurnRate"] = (
    group_summary["ChurnedCustomers"]
    / group_summary["TotalCustomers"]
) * 100

print("\n==========================================")
print("A/B TEST RESULTS")
print("==========================================")

print(group_summary)

# ------------------------------------------
# 5. Compare Control vs Variant
# ------------------------------------------

control_retention = group_summary.loc[
    group_summary["Group"] == "Control",
    "RetentionRate"
].iloc[0]

variant_retention = group_summary.loc[
    group_summary["Group"] == "Variant",
    "RetentionRate"
].iloc[0]

retention_difference = (
    variant_retention - control_retention
)

print("\nControl Retention Rate: {:.2f}%".format(
    control_retention
))

print("Variant Retention Rate: {:.2f}%".format(
    variant_retention
))

print("Retention Difference: {:.2f} percentage points".format(
    retention_difference
))

# ------------------------------------------
# 6. Simple Statistical Comparison
# ------------------------------------------

from math import sqrt

control_row = group_summary[
    group_summary["Group"] == "Control"
].iloc[0]

variant_row = group_summary[
    group_summary["Group"] == "Variant"
].iloc[0]

p1 = control_row["RetentionRate"] / 100
p2 = variant_row["RetentionRate"] / 100

n1 = control_row["TotalCustomers"]
n2 = variant_row["TotalCustomers"]

pooled_p = (
    control_row["RetainedCustomers"]
    + variant_row["RetainedCustomers"]
) / (n1 + n2)

standard_error = sqrt(
    pooled_p
    * (1 - pooled_p)
    * (1 / n1 + 1 / n2)
)

if standard_error != 0:
    z_score = (p2 - p1) / standard_error
else:
    z_score = 0

print("\nZ-Score: {:.4f}".format(z_score))

# ------------------------------------------
# 7. Business Interpretation
# ------------------------------------------

if retention_difference > 0:
    print(
        "\nVariant group has a higher retention rate "
        "than the Control group."
    )
elif retention_difference < 0:
    print(
        "\nVariant group has a lower retention rate "
        "than the Control group."
    )
else:
    print(
        "\nBoth groups have the same retention rate."
    )

# ------------------------------------------
# 8. Save A/B Test Results
# ------------------------------------------

group_summary.to_csv(
    "outputs/ab_test_results.csv",
    index=False
)

print("\nSaved: outputs/ab_test_results.csv")

print("\n==========================================")
print("DAY 9 A/B TESTING COMPLETED!")
print("==========================================")







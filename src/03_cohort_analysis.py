import pandas as pd
import plotly.graph_objects as go
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
# 3. Create Tenure Cohorts
# ==========================================

def create_tenure_cohort(tenure):
    if tenure <= 6:
        return "0-6 Months"
    elif tenure <= 12:
        return "7-12 Months"
    elif tenure <= 24:
        return "13-24 Months"
    else:
        return "25+ Months"


df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())

df["Cohort"] = df["Tenure"].apply(create_tenure_cohort)

# ==========================================
# 4. Calculate Cohort Statistics
# ==========================================

cohort_analysis = (
    df.groupby("Cohort")
    .agg(
        TotalCustomers=("CustomerID", "count"),
        ChurnedCustomers=("Churn", "sum"),
        AverageOrderCount=("OrderCount", "mean"),
        AverageCashback=("CashbackAmount", "mean")
    )
    .reset_index()
)

cohort_analysis["ChurnRate"] = (
    cohort_analysis["ChurnedCustomers"]
    / cohort_analysis["TotalCustomers"]
) * 100

cohort_analysis["RetentionRate"] = (
    100 - cohort_analysis["ChurnRate"]
)

# ==========================================
# 5. Arrange Cohorts in Correct Order
# ==========================================

cohort_order = [
    "0-6 Months",
    "7-12 Months",
    "13-24 Months",
    "25+ Months"
]

cohort_analysis["Cohort"] = pd.Categorical(
    cohort_analysis["Cohort"],
    categories=cohort_order,
    ordered=True
)

cohort_analysis = cohort_analysis.sort_values("Cohort")

print("\n==========================================")
print("COHORT ANALYSIS")
print("==========================================")

print(cohort_analysis)

# ==========================================
# 6. Save Cohort Summary
# ==========================================

cohort_analysis.to_csv(
    "outputs/cohort_analysis.csv",
    index=False
)

print("\nSaved: outputs/cohort_analysis.csv")

# ==========================================
# 7. Create Interactive Retention Dashboard
# ==========================================

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=cohort_analysis["Cohort"].astype(str),
        y=cohort_analysis["RetentionRate"],
        name="Retention Rate",
        text=cohort_analysis["RetentionRate"].round(2),
        texttemplate="%{text}%",
        textposition="auto"
    )
)

fig.update_layout(
    title="Customer Cohort Retention Analysis",
    xaxis_title="Tenure Cohort",
    yaxis_title="Retention Rate (%)",
    yaxis=dict(range=[0, 100]),
    template="plotly_white"
)

fig.write_html(
    "outputs/cohort_retention.html"
)

print("Saved: outputs/cohort_retention.html")

# ==========================================
# 8. Create Interactive Churn Dashboard
# ==========================================

churn_fig = go.Figure()

churn_fig.add_trace(
    go.Scatter(
        x=cohort_analysis["Cohort"].astype(str),
        y=cohort_analysis["ChurnRate"],
        mode="lines+markers",
        name="Churn Rate"
    )
)

churn_fig.update_layout(
    title="Customer Churn Rate Across Tenure Cohorts",
    xaxis_title="Tenure Cohort",
    yaxis_title="Churn Rate (%)",
    template="plotly_white"
)

churn_fig.write_html(
    "outputs/cohort_churn_trend.html"
)

print("Saved: outputs/cohort_churn_trend.html")

# ==========================================
# 9. Completion Message
# ==========================================

print("\n==========================================")
print("DAY 6 COHORT ANALYSIS COMPLETED SUCCESSFULLY!")
print("Interactive Plotly dashboards saved in outputs/.")
print("==========================================")
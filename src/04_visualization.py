import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "Data/E Commerce Dataset-E Comm.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# ==========================================
# 2. Create Outputs Folder
# ==========================================

os.makedirs("outputs", exist_ok=True)

# ==========================================
# 3. Set Plot Style
# ==========================================

sns.set_theme(style="whitegrid")

# ==========================================
# 4. Churn Count by Satisfaction Score
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="SatisfactionScore",
    hue="Churn"
)

plt.title("Customer Churn by Satisfaction Score")
plt.xlabel("Satisfaction Score")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    "outputs/churn_by_satisfaction.png",
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/churn_by_satisfaction.png")

# ==========================================
# 5. Churn Distribution
# ==========================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    "outputs/churn_distribution.png",
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/churn_distribution.png")

# ==========================================
# 6. Tenure Distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Tenure",
    kde=True
)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    "outputs/tenure_distribution.png",
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/tenure_distribution.png")

# ==========================================
# 7. Warehouse Distance by Churn
# ==========================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="WarehouseToHome"
)

plt.title("Warehouse to Home Distance by Churn")
plt.xlabel("Churn")
plt.ylabel("Warehouse to Home Distance")
plt.tight_layout()

plt.savefig(
    "outputs/warehouse_distance_by_churn.png",
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/warehouse_distance_by_churn.png")

# ==========================================
# 8. Cashback Amount Distribution
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="CashbackAmount",
    kde=True
)

plt.title("Cashback Amount Distribution")
plt.xlabel("Cashback Amount")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    "outputs/cashback_distribution.png",
    bbox_inches="tight"
)

plt.close()

print("Saved: outputs/cashback_distribution.png")

# ==========================================
# 9. Completion Message
# ==========================================

print("\n==========================================")
print("DAY 4 VISUALIZATION COMPLETED SUCCESSFULLY!")
print("All charts saved inside outputs/ folder.")
print("==========================================")

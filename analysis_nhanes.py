import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
demo = pd.read_csv("data/demographic.csv", encoding="latin-1", low_memory=False)
exam = pd.read_csv("data/examination.csv", encoding="latin-1", low_memory=False)
labs = pd.read_csv("data/labs.csv", encoding="latin-1", low_memory=False)

plt.style.use("ggplot")

# ------------------------
# 1. Age distribution
# ------------------------
plt.figure(figsize=(8,5))
demo["RIDAGEYR"].hist(bins=40)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("images/age_distribution.png", dpi=300)
plt.close()

# ------------------------
# 2. Gender proportion
# ------------------------
plt.figure(figsize=(6,5))
demo["RIAGENDR"].replace({1:"Male", 2:"Female"}).value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.ylabel("Count")
plt.savefig("images/gender_distribution.png", dpi=300)
plt.close()

# ------------------------
# 3. BMI distribution
# ------------------------
plt.figure(figsize=(8,5))
if "BMXBMI" in exam.columns:
    exam["BMXBMI"].hist(bins=40)
    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Count")
    plt.savefig("images/bmi_distribution.png", dpi=300)
plt.close()

# ------------------------
# 4. Glucose distribution (lab)
# ------------------------
glucose_cols = [c for c in labs.columns if "GLU" in c.upper() or "GL" in c.upper()]

if len(glucose_cols) > 0:
    gcol = glucose_cols[0]
    plt.figure(figsize=(8,5))
    labs[gcol].hist(bins=40)
    plt.title(f"{gcol} Distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.savefig("images/glucose_distribution.png", dpi=300)
    plt.close()

# ------------------------
# 5. Correlation heatmap (subset)
# ------------------------
subset_cols = []

# Age + Gender
subset_cols += ["RIDAGEYR", "RIAGENDR"]

# If BMI exists
if "BMXBMI" in exam.columns:
    exam_small = exam[["SEQN", "BMXBMI"]]
    demo = demo.merge(exam_small, on="SEQN", how="left")
    subset_cols.append("BMXBMI")

# If glucose exists
if len(glucose_cols) > 0:
    glucose_small = labs[["SEQN", gcol]]
    demo = demo.merge(glucose_small, on="SEQN", how="left")
    subset_cols.append(gcol)

corr_df = demo[subset_cols].dropna()

plt.figure(figsize=(7,5))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Key Features")
plt.savefig("images/correlation_heatmap.png", dpi=300)
plt.close()

print("All plots saved to /images")

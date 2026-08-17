import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("Lung_Cancer.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

# Remove spaces from column names
data.columns = data.columns.str.strip()

print("\nOriginal columns:")
print(data.columns.tolist())


# ==========================================
# 2. Clean Column Names
# ==========================================

data.columns = (
    data.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_")
)

print("\nCleaned columns:")
print(data.columns.tolist())


# ==========================================
# 3. Find Target Column
# ==========================================

target = "LUNG_CANCER"

if target not in data.columns:
    print("\nERROR: LUNG_CANCER column not found!")
    print("Available columns:")
    print(data.columns.tolist())
    exit()


# ==========================================
# 4. Convert Gender
# ==========================================

if "GENDER" in data.columns:

    data["GENDER"] = (
        data["GENDER"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data["GENDER"] = data["GENDER"].map({
        "M": 1,
        "MALE": 1,
        "F": 0,
        "FEMALE": 0
    })


# ==========================================
# 5. Convert YES / NO Columns
# ==========================================

binary_columns = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN"
]


for column in binary_columns:

    if column in data.columns:

        # String values
        if data[column].dtype == "object":

            data[column] = (
                data[column]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({
                    "YES": 1,
                    "NO": 0,
                    "Y": 1,
                    "N": 0
                })
            )

        # Numeric values
        else:

            unique_values = set(
                data[column].dropna().unique()
            )

            # Dataset commonly uses:
            # 1 = NO
            # 2 = YES

            if unique_values.issubset({1, 2}):

                data[column] = data[column].map({
                    1: 0,
                    2: 1
                })


# ==========================================
# 6. Convert Target Column
# ==========================================

if data[target].dtype == "object":

    data[target] = (
        data[target]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "YES": 1,
            "NO": 0,
            "Y": 1,
            "N": 0
        })
    )

else:

    unique_target = set(
        data[target].dropna().unique()
    )

    if unique_target.issubset({1, 2}):

        data[target] = data[target].map({
            1: 0,
            2: 1
        })


# ==========================================
# 7. Remove Missing Values
# ==========================================

data = data.dropna()

print("\nData after preprocessing:")
print(data.head())


# ==========================================
# 8. Separate Features and Target
# ==========================================

X = data.drop(target, axis=1)

y = data[target]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(target)


# ==========================================
# 9. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 10. Create Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 11. Train Model
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 12. Test Model
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL RESULT")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 13. Save Model
# ==========================================

model_data = {
    "model": model,
    "features": list(X.columns)
}

joblib.dump(
    model_data,
    "lung_cancer_model.pkl"
)


print("\n==============================")
print("SUCCESS!")
print("==============================")

print(
    "Model saved as: lung_cancer_model.pkl"
)
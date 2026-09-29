import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ============================================================
# CUSTOM PREDICTION SYSTEM
# ============================================================

print("=" * 55)
print("             CUSTOM PREDICTION SYSTEM")
print("=" * 55)


# ============================================================
# 1. GET FILE
# ============================================================

file_name = input("\nEnter file name (.xlsx / .xls / .csv): ").strip()

if not os.path.exists(file_name):
    print("\nERROR: File not found!")
    exit()


# ============================================================
# 2. READ FILE
# ============================================================

extension = os.path.splitext(file_name)[1].lower()

try:

    if extension == ".csv":
        data = pd.read_csv(file_name)

    elif extension == ".xlsx":
        data = pd.read_excel(
            file_name,
            engine="openpyxl"
        )

    elif extension == ".xls":
        data = pd.read_excel(
            file_name,
            engine="xlrd"
        )

    else:
        print("\nERROR: Only CSV, XLSX and XLS files are supported.")
        exit()

except Exception as e:

    print("\nERROR while reading file:")
    print(e)
    exit()


print("\nFile loaded successfully!")


# ============================================================
# 3. REMOVE EMPTY EXCEL COLUMNS
# ============================================================

data = data.loc[
    :,
    ~data.columns.astype(str).str.contains("^Unnamed")
]


# ============================================================
# 4. SHOW AVAILABLE COLUMNS
# ============================================================

print("\nAvailable columns:")

for i, column in enumerate(data.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 5. SELECT DATE COLUMN
# ============================================================

date_column = input(
    "\nEnter the DATE column name: "
).strip()

if date_column not in data.columns:

    print("\nERROR: Date column not found!")

    print("\nAvailable columns are:")

    for column in data.columns:
        print(column)

    exit()


# ============================================================
# 6. SELECT TARGET COLUMN
# ============================================================

target_column = input(
    "\nEnter the VALUE column you want to predict: "
).strip()

if target_column not in data.columns:

    print("\nERROR: Target column not found!")

    print("\nAvailable columns are:")

    for column in data.columns:
        print(column)

    exit()


# ============================================================
# 7. CONVERT DATE COLUMN
# ============================================================

data[date_column] = pd.to_datetime(
    data[date_column],
    errors="coerce"
)


# ============================================================
# 8. CONVERT TARGET TO NUMERIC
# ============================================================

data[target_column] = pd.to_numeric(
    data[target_column],
    errors="coerce"
)


# ============================================================
# 9. REMOVE INVALID DATA
# ============================================================

data = data.dropna(
    subset=[
        date_column,
        target_column
    ]
)


# ============================================================
# 10. SORT BY DATE
# ============================================================

data = data.sort_values(
    by=date_column
).reset_index(drop=True)


# ============================================================
# 11. CHECK DATA
# ============================================================

if len(data) < 10:

    print(
        "\nERROR: Not enough valid data "
        "to train the model."
    )

    exit()


print("\nData prepared successfully!")

print("\nNumber of valid records:", len(data))


# ============================================================
# 12. CONVERT DATE TO NUMBER
# ============================================================

first_date = data[date_column].min()

data["Days"] = (
    data[date_column] - first_date
).dt.days


# ============================================================
# 13. CREATE X AND Y
# ============================================================

X = data[["Days"]]

y = data[target_column]


# ============================================================
# 14. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(data) * 0.8)

train_x = X.iloc[:split_index]
test_x = X.iloc[split_index:]

train_y = y.iloc[:split_index]
test_y = y.iloc[split_index:]


# ============================================================
# 15. CREATE MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    train_x,
    train_y
)

print("Model trained successfully!")


# ============================================================
# 17. TEST MODEL
# ============================================================

test_prediction = model.predict(test_x)


# ============================================================
# 18. EVALUATION
# ============================================================

mae = mean_absolute_error(
    test_y,
    test_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        test_y,
        test_prediction
    )
)

# Mean Absolute Percentage Error
mape = np.mean(
    np.abs(
        (test_y - test_prediction) / test_y
    )
) * 100

# Convert MAPE to an accuracy-like percentage
accuracy = max(0, 100 - mape)


# ============================================================
# 19. SHOW MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 55)
print("                 MODEL RESULTS")
print("=" * 55)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


# ============================================================
# ASK FUTURE DATE
# ============================================================

future_date_text = input(
    "\nEnter the FUTURE DATE you want to predict (YYYY-MM-DD): "
).strip()

future_date = pd.to_datetime(
    future_date_text,
    errors="coerce"
)

if pd.isna(future_date):
    print("\nERROR: Invalid date!")
    exit()


# ============================================================
# CHECK FUTURE DATE
# ============================================================

last_date = data[date_column].max()

if future_date <= last_date:
    print("\nERROR: Please enter a date after:")
    print(last_date.date())
    exit()


# ============================================================
# CONVERT DATE TO NUMBER
# ============================================================

future_days = (
    future_date - first_date
).days


# ============================================================
# PREDICT FUTURE VALUE
# ============================================================

future_prediction = model.predict(
    pd.DataFrame({
        "Days": [future_days]
    })
)

predicted_value = future_prediction[0]


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 55)
print("                 FINAL RESULT")
print("=" * 55)

print(f"Prediction Date : {future_date.date()}")
print(f"Target Column   : {target_column}")
print(f"Predicted Value : {predicted_value:.2f}")
print(f"Model Accuracy  : {accuracy:.2f}%")

print("=" * 55)
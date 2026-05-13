import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load Dataset
data = pd.read_csv("student_data.csv")

# Input Features
X = data[[
    "study_hours",
    "attendance",
    "previous_marks"
]]

# Target Variable
y = data["final_marks"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy Check
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)

# Predict New Student Performance
study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))
previous_marks = float(input("Enter Previous Marks: "))

new_data = [[
    study_hours,
    attendance,
    previous_marks
]]

predicted_marks = model.predict(new_data)

print(
    "Predicted Final Marks:",
    round(predicted_marks[0], 2)
)

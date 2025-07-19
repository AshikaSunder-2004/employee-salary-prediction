import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("employee_data.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Initialize separate encoders for each categorical column
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_job = LabelEncoder()
le_loc = LabelEncoder()

# Fit and transform
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Education"] = le_edu.fit_transform(df["Education"])
df["Job_Title"] = le_job.fit_transform(df["Job_Title"])
df["Location"] = le_loc.fit_transform(df["Location"])

# Save encoders
pickle.dump(le_gender, open("le_gender.pkl", "wb"))
pickle.dump(le_edu, open("le_edu.pkl", "wb"))
pickle.dump(le_job, open("le_job.pkl", "wb"))
pickle.dump(le_loc, open("le_loc.pkl", "wb"))

# Features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("salary_model.pkl", "wb"))

print("✅ Model and label encoders saved.")

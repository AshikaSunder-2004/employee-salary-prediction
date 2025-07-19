import pandas as pd

df = pd.read_csv("employee_data.csv")
print(df.head())
print(df.info())

# Handle missing values (if any)
df.dropna(inplace=True)

# Label encode categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ["Gender", "Education", "Job_Title", "Location"]:
    df[col] = le.fit_transform(df[col])

df.to_csv("employee_cleaned.csv", index=False)

from sklearn.datasets import load_iris
import pandas as pd

# 1. Load the dataset
iris = load_iris()

# 2. Convert to a DataFrame for easier statistics
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# 3. Print basic statistics (Requirement 2)
print("--- Iris Dataset Statistics ---")
print(df.describe())

# 4. Optional: Print the first 5 rows to verify data
print("\n--- First 5 Rows ---")
print(df.head())
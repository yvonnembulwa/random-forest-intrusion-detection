# load_data.py
import pandas as pd
import config

print("=" * 50)
print("Loading AWID-CLS-R Dataset")
print("=" * 50)

# Load the training data
df = pd.read_csv(config.DATA_PATH['train'], low_memory=False)

print(f"\n Training set shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Show all column names to find the class column
print("\n All column names (last 10):")
for i, col in enumerate(df.columns[-10:]):
    print(f"   {i+1}. '{col}'")

# The class column is the last column
# the last column contains normal, injection, impersonation, flooding
label_col = df.columns[-1]
print(f"\n The class/label column is: '{label_col}'")

# class distribution ( 4 ATTACK TYPES!)
print(f"\n Class Distribution (Training Set):")
class_counts = df[label_col].value_counts()
for class_name, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   {class_name}: {count:,} ({percentage:.2f}%)")

# Load test set
print(f"\n Loading test set...")
df_test = pd.read_csv(config.DATA_PATH['test'], low_memory=False)
print(f"Test set shape: {df_test.shape[0]:,} rows, {df_test.shape[1]} columns")

# test set class distribution
test_label_col = df_test.columns[-1]
print(f"\n Class Distribution (Test Set):")
test_class_counts = df_test[test_label_col].value_counts()
for class_name, count in test_class_counts.items():
    percentage = (count / len(df_test)) * 100
    print(f"   {class_name}: {count:,} ({percentage:.2f}%)")

print("\n" + "=" * 50)
print(" Ready for preprocessing!")
print("=" * 50)
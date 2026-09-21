# preprocess.py Import StandardScaler - used to scale features to zero mean and unit variance
# Import LabelEncoder - used to convert string labels to numeric values (0,1,2,3)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config

print("=" * 60)
print("PREPROCESSING AWID-CLS-R DATASET")
print("=" * 60)

# Load training dataset from CSV file specified in config.py
# low_memory=False ensures mixed data types are read correctly

print("\n1. Loading data...")
df_train = pd.read_csv(config.DATA_PATH['train'], low_memory=False)
df_test = pd.read_csv(config.DATA_PATH['test'], low_memory=False)

print(f"   Train shape: {df_train.shape}")
print(f"   Test shape: {df_test.shape}")

# The class column is named 'normal' (last column)
class_col = 'normal'

# Get all feature columns (all columns except the class column)
feature_cols = [col for col in df_train.columns if col != class_col]

# Get common columns between train and test,ensures consistency when splitting and evaluating

common_cols = [col for col in feature_cols if col in df_test.columns]
print(f"\n2. Found {len(common_cols)} common feature columns")

# Keep only common columns Extract only the common feature columns from  sets
# Extract the class labels from training set (original string labels)
# y_train contains values: 'normal', 'flooding', 'impersonation', 'injection'
X_train = df_train[common_cols]
X_test = df_test[common_cols]
y_train = df_train[class_col]
y_test = df_test[class_col]

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Convert to numeric, coerce errors to NaN
print("\n3. Converting to numeric...")
X_train = X_train.apply(pd.to_numeric, errors='coerce') # Convert every value in training features to numeric

X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with column mean from training set
# Calculate the mean (average) of the column, ignoring NaN values
print("\n4. Handling missing values...")
for col in X_train.columns:
    mean_val = X_train[col].mean()
    if pd.isna(mean_val):
        mean_val = 0
    X_train[col] = X_train[col].fillna(mean_val)# Replace all NaN values in training column with the calculated mean
    X_test[col] = X_test[col].fillna(mean_val)# Replace all NaN values in test column with the SAME mean from training

# Encode labels Fit the encoder on training labels (learns the mapping) and transform them
# Converts: flooding→0, impersonation→1, injection→2, normal→3
# Transform test labels using the same mapping (no re-fitting)
# Ensures consistent encoding across both datasets

print("\n5. Encoding target labels...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# loop through each value with its index
#  classes_ returns ['flooding', 'impersonation', 'injection', 'normal']
print(f"\n   Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"      {label} -> {i}")

# Class distribution: Find unique values in encoded training labels and count occurrences
# unique: array of class IDs [0,1,2,3]
# Loop through each class name and its count
print(f"\n6. Class distribution:")
unique, counts = np.unique(y_train_encoded, return_counts=True)
for label, count in zip(label_encoder.classes_, counts):
    print(f"      {label}: {count:,} ({count/len(y_train_encoded)*100:.2f}%)")

# Scale features # StandardScaler transforms data to have mean=0 and standard deviation=1
# Fit the scaler on training data (calculates mean and std for each feature)
# Then transform training data using those parameters
print("\n7. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n   Final shapes:")
print(f"      X_train_scaled: {X_train_scaled.shape}")
print(f"      X_test_scaled: {X_test_scaled.shape}")
print(f"      y_train: {y_train_encoded.shape}")
print(f"      y_test: {y_test_encoded.shape}")

# Save processed data for later use to NumPy binary file
print("\n8. Saving processed data...")
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train_encoded)
np.save('y_test.npy', y_test_encoded)
print("   Saved to: X_train.npy, X_test.npy, y_train.npy, y_test.npy")

print("\n" + "=" * 60)
print("Preprocessing complete!")
print("=" * 60)
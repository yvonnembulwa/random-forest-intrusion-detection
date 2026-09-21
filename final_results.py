# final_results.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import config
import joblib
import pandas as pd

print("=" * 60) # prints a separator line of 60 equals signs for  visual clarity 
print("FINAL RESULTS FOR THE DOCUMENT")
print("=" * 60)

# print step indicator to show progress
# load the preprossed training labels from the .npy files created by preprocess.py
# x_full contains all feature vectors(1 ,795,574 samples,122 features)
# y_full conttains encoded labels (0=flooding,1=impersonation,2=injection,3=normal)
print("\n1. Loading data...")
X_full = np.load('X_train.npy')
y_full = np.load('y_train.npy')

# Split into train (80%) and test (20%)
# stratify=y_full preserves class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_full
)

class_names = ['flooding', 'impersonation', 'injection', 'normal']

print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")

# Create balanced training set defining the samples,create empty lists to collect balanced feature samples and balanced labels
print("\n2. Creating balanced training set (20,000 per class)...")
samples_per_class = 20000

X_balanced = []
y_balanced = []

# Loop through each class ID (0,1,2,3)Find all indices in the training set where the label equals the current class ID
# Randomly select 20,000 indices from this class without replacement
# 'replace=False' ensures no duplicate samples are selected ,Add the selected feature samples and corresponding labels to the balanced list
for class_id in range(4):
    class_indices = np.where(y_train == class_id)[0]
    sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)
    X_balanced.append(X_train[sampled_indices])
    y_balanced.append(y_train[sampled_indices])
    print(f"   {class_names[class_id]}: {samples_per_class} samples")

# Vertically stack all the feature arrays into one array Result shape: (80,000, 122) - 20,000 samples × 4 classes = 80,000 total
# horizontally stack all the label arrays into one array Result shape: (80,000,) - one label per sample
X_balanced = np.vstack(X_balanced)
y_balanced = np.hstack(y_balanced)

# Train Random Forest
print("\n3. Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, # Number of decision trees in the forest (100 trees)
    max_depth=15, # Maximum depth of each tree (prevents overfitting)
    min_samples_split=10, # Minimum samples required to split an internal node
    min_samples_leaf=5, # Minimum samples required at a leaf node
    random_state=config.RANDOM_STATE,
    n_jobs=-1 # Use all available CPU cores for parallel processing
)
# Train the Random Forest model on the balanced training data (80,000 samples)
rf.fit(X_balanced, y_balanced)
print("  Random Forest training complete!")

# Evaluate:Make predictions on the test set ,Returns an array of predicted class labels (0,1,2,3)

print("\n4. Evaluating on test set...")
y_pred = rf.predict(X_test)

# Calculate metrics:Calculate macro-averaged precision (average of per-class precision, equal weight)
#Calculate weighted-averaged precision (weighted by number of samples per class)
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)# zero_division=0 prevents warnings when a class has no predicted samples
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "=" * 60)
print("RANDOM FOREST PERFORMANCE - Multi-class Classification")
print("=" * 60)

print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")# Print overall accuracy as decimal and percentage

print(f"\nMacro-averaged Metrics (equal weight per class):")# Macro-averaged gives equal importance to each attack type (good for imbalanced data)

print(f"   Precision: {precision_macro:.4f}")
print(f"   Recall:    {recall_macro:.4f}")
print(f"   F1-Score:  {f1_macro:.4f}")

print(f"\nWeighted-averaged Metrics (weighted by class size):")# Weighted-averaged gives more importance to larger classes (normal traffic)
print(f"   Precision: {precision_weighted:.4f}")
print(f"   Recall:    {recall_weighted:.4f}")
print(f"   F1-Score:  {f1_weighted:.4f}")

# per class performance
print("\n" + "=" * 60)
print("PER-CLASS PERFORMANCE")
print("=" * 60)

per_class_results = [] # Create empty list to store results for each class (will be exported to CSV)
# the loop goes through each class with its index (0,1,2,3) and name,Creates a boolean mask: True where test label equals current class,
# Checks if there are any samples of this class in the test set and Count how many predictions for this class were correct,Calculates accuracy for this class: correct / total samples of this class
for i, class_name in enumerate(class_names):
    class_mask = (y_test == i) 

    if np.sum(class_mask) > 0:
        correct = np.sum(y_pred[class_mask] == i)
        class_acc = correct / np.sum(class_mask)
        
        # Get precision, recall, f1 for this class
        tp = np.sum((y_pred == i) & (y_test == i)) # Calculate True Positives (predicted as class i AND actually class i)

        fp = np.sum((y_pred == i) & (y_test != i)) # Calculate False Positives (predicted as class i BUT actually not class i)

        fn = np.sum((y_pred != i) & (y_test == i)) # Calculate False Negatives (predicted NOT as class i BUT actually class i)

        
        precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_class = 2 * precision_class * recall_class / (precision_class + recall_class) if (precision_class + recall_class) > 0 else 0
        # Store results in dictionary for CSV export

        per_class_results.append({
            'Attack Type': class_name,
            'Samples': np.sum(class_mask),
            'Detected': correct,
            'Accuracy': f"{class_acc*100:.2f}%",
            'Precision': f"{precision_class:.4f}",
            'Recall': f"{recall_class:.4f}",
            'F1-Score': f"{f1_class:.4f}"
        })
        
        print(f"\n{class_name.upper()}:")
        print(f"   Samples: {np.sum(class_mask):,}")
        print(f"   Detected: {correct:,} out of {np.sum(class_mask):,}")
        print(f"   Accuracy: {class_acc*100:.2f}%")
        print(f"   Precision: {precision_class:.4f}")
        print(f"   Recall: {recall_class:.4f}")
        print(f"   F1-Score: {f1_class:.4f}")

# Create confusion matrix showing actual vs predicted class counts Rows = actual labels, Columns = predicted labels
print("\n5. Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix - Random Forest (Multi-class Attack Detection)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('final_confusion_matrix.png')
print("   Saved to: final_confusion_matrix.png")

# Feature importance : extract feature importance scores from the trained model
# Sort features by importance (descending) and take top 15 indices
print("\n6. Analyzing Feature Importance...")
importances = rf.feature_importances_ 
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(12, 8))
plt.barh(range(15), importances[indices][::-1])
plt.yticks(range(15), [f"Feature_{i}" for i in indices[::-1]])
plt.xlabel('Importance Score')
plt.title('Top 15 Most Important MAC-Layer Features')
plt.tight_layout()
plt.savefig('final_feature_importance.png')
print("   Saved to: final_feature_importance.png")

# Save results to CSV Convert the list of dictionaries to a Pandas DataFrame (table format)
print("\n7. Saving results to CSV...")
results_df = pd.DataFrame(per_class_results)
results_df.to_csv('final_results.csv', index=False)
print("   Saved to: final_results.csv")

# Save model Save the trained Random Forest model to a .pkl file using joblib which allows reloading of the model later without retraining
print("\n8. Saving model...")
joblib.dump(rf, 'final_random_forest_model.pkl')
print("   Saved to: final_random_forest_model.pkl")

print("\n" + "=" * 60)
print("RESULTS READY FOR THE DOCUMENT!")
print("=" * 60)

# summary table
print("\nSUMMARY TABLE (For the document):")
print("-" * 60)
print(f"{'Attack Type':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 60)
for result in per_class_results:
    print(f"{result['Attack Type']:<15} {result['Accuracy']:<12} {result['Precision']:<12} {result['Recall']:<12} {result['F1-Score']:<12}")
print("-" * 60)
print(f"{'MACRO AVG':<15} {f1_macro*100:>5.2f}%{'':<6} {precision_macro:.4f}{'':<8} {recall_macro:.4f}{'':<8} {f1_macro:.4f}")
print(f"{'WEIGHTED AVG':<15} {accuracy*100:>5.2f}%{'':<6} {precision_weighted:.4f}{'':<8} {recall_weighted:.4f}{'':<8} {f1_weighted:.4f}")
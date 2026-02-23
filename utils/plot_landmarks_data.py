import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Read data
raw_train_path = "processed_data/raw_landmarks_training_set.csv"
raw_test_path = "processed_data/raw_landmarks_testing_set.csv"
std_train_path = "processed_data/standardized_landmarks_training_set.csv"
std_test_path = "processed_data/standardized_landmarks_testing_set.csv"

raw_train = pd.read_csv(raw_train_path)
raw_test = pd.read_csv(raw_test_path)
std_train = pd.read_csv(std_train_path)
std_test = pd.read_csv(std_test_path)

# Delete 'label' column
X_raw_train = raw_train.drop(columns=["label"])
X_raw_test = raw_test.drop(columns=["label"])
X_std_train = std_train.drop(columns=["label"])
X_std_test = std_test.drop(columns=["label"])

# boxplot
plt.figure(figsize=(15, 8))

# raw training chart
plt.subplot(2, 2, 1)
plt.title("Raw Training Set")
plt.boxplot(X_raw_train.values)
plt.xticks([], [])
plt.ylabel("Raw Value")

# standardized training chart
plt.subplot(2, 2, 2)
plt.title("Standardized Training Set")
plt.boxplot(X_std_train.values)
plt.xticks([], [])
plt.ylabel("Standardized Value")

# raw testing chart
plt.subplot(2, 2, 3)
plt.title("Raw Testing Set")
plt.boxplot(X_raw_test.values)
plt.xticks([], [])
plt.xlabel("Landmark Index")
plt.ylabel("Raw Value")

# standardized testing chart
plt.subplot(2, 2, 4)
plt.title("Standardized Testing Set")
plt.boxplot(X_std_test.values)
plt.xticks([], [])
plt.xlabel("Landmark Index")
plt.ylabel("Standardized Value")

plt.tight_layout()
plt.savefig('boxplot_data_chart.png', dpi=300, bbox_inches='tight')
plt.show()

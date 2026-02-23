import pandas as pd

std_train_path = "processed_data/raw_landmarks_training_set.csv"
std_test_path = "processed_data/raw_landmarks_testing_set.csv"

df_train = pd.read_csv(std_train_path)
df_test = pd.read_csv(std_test_path)

print(df_train.head())
print(df_test.head())

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def standardize_landmarks(input_csv, output_csv):
    """
    Standardize landmark columns in CSV files using StandardScaler.
    Leave the 'label' column intact.
    """
    # Read data
    print(f"\nProcessing file: {input_csv}")
    df = pd.read_csv(input_csv)

    # Separate the features (all columns except 'label') and labels
    X = df.drop(columns=['label'])
    y = df['label']

    print(f" - Number of row (sample): {X.shape[0]}")
    print(f" - Number of column (features): {X.shape[1]}")

    # Initialize Standardizer
    scaler = StandardScaler()

    # Fit and transform features data
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame (columns remaining)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Attach column 'label'
    X_scaled_df['label'] = y

    # Save standardized data
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    X_scaled_df.to_csv(output_csv, index=False)

    print(f"Saved standardized file: {output_csv}\n")

if __name__ == "__main__":
    input_folder = "processed_data"

    files = [
        ("raw_landmarks_training_set.csv", "standardized_landmarks_training_set.csv"),
        ("raw_landmarks_testing_set.csv", "standardized_landmarks_testing_set.csv")
    ]

    for input_file, output_file in files:
        input_csv = os.path.join(input_folder, input_file)
        output_csv = os.path.join(input_folder, output_file)
        
        if os.path.exists(input_csv):
            standardize_landmarks(input_csv, output_csv)
        else:
            print(f"Can not find file: {input_csv}")

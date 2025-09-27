import os
import json
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    """
    Reads the single, official VQA_RAD JSON file, splits it into train and test sets
    at the QUESTION level, and further splits the test set into open-ended and 
    close-ended questions, saving them as separate CSV files.
    
    The official VQA-RAD dataset can be downloaded from: https://osf.io/89kps/files/osfstorage
    This script correctly follows the official train/test split methodology.
    """
    parser = argparse.ArgumentParser(description="Prepare VQA-RAD CSV files with a question-level split.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True, 
        help="Path to the original VQA_RAD_Dataset.json file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the output CSV files."
    )
    args = parser.parse_args()

    print(f"Reading data from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # --- Initial Data Analysis ---
    print("\n--- Initial Dataset Analysis ---")
    total_questions = len(df)
    total_unique_images = len(df['image_name'].unique())
    df['answer_type'] = df['answer_type'].str.upper() # Standardize column
    total_open = len(df[df['answer_type'] == 'OPEN'])
    total_close = len(df[df['answer_type'] == 'CLOSED'])
    
    print(f"Total Question-Answer Pairs: {total_questions}")
    print(f"Total Unique Images: {total_unique_images}")
    print(f"  - Total Open-Ended Questions: {total_open}")
    print(f"  - Total Close-Ended Questions: {total_close}")

    # --- Create DataFrames based on Question-Level Split ---
    train_rows = [item for item in data if 'test' not in item.get('phrase_type', '')]
    test_rows = [item for item in data if 'test' in item.get('phrase_type', '')]

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)
    
    # Standardize answer_type columns for accurate counting
    df_train['answer_type'] = df_train['answer_type'].str.upper()
    df_test['answer_type'] = df_test['answer_type'].str.upper()

    # --- Create and Save CSV Files ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save train.csv
    train_csv_path = os.path.join(args.output_dir, "train.csv")
    df_train[['image_name', 'question', 'answer']].to_csv(train_csv_path, index=False)
    
    # Split test data into open and close and save
    df_test_open = df_test[df_test['answer_type'] == 'OPEN']
    df_test_close = df_test[df_test['answer_type'] == 'CLOSED']
    
    open_csv_path = os.path.join(args.output_dir, "test_open.csv")
    close_csv_path = os.path.join(args.output_dir, "test_close.csv")
    
    df_test_open[['image_name', 'question', 'answer']].to_csv(open_csv_path, index=False)
    df_test_close[['image_name', 'question', 'answer']].to_csv(close_csv_path, index=False)

    # --- Final Detailed Summary ---
    print("\n--- Final Data Split Summary ---")
    print(f"Training Set:")
    print(f"  - Total Questions: {len(df_train)}")
    print(f"    - Open-Ended: {len(df_train[df_train['answer_type'] == 'OPEN'])}")
    print(f"    - Close-Ended: {len(df_train[df_train['answer_type'] == 'CLOSED'])}")
    print(f"  - Unique Images: {len(df_train['image_name'].unique())}")
    
    print(f"\nTest Set:")
    print(f"  - Total Questions: {len(df_test)}")
    print(f"    - Open-Ended (test_open.csv): {len(df_test_open)}")
    print(f"    - Close-Ended (test_close.csv): {len(df_test_close)}")
    print(f"  - Unique Images: {len(df_test['image_name'].unique())}")
    
    print(f"\nPre-processing complete. Files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
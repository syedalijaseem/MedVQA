import json
import pandas as pd
import argparse
import os

def main():
    """
    Reads the single, official VQA_RAD_Dataset JSON file, splits it into train and test sets at
    the QUESTION level, and further splits the test set into open-ended and close-ended questions.
    
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

    train_rows = []
    test_rows = []

    # Split data based on whether 'phrase_type' contains the word 'test'
    for item in data:
        entry = {
            'image_name': item.get('image_name'),
            'question': item.get('question'),
            'answer': item.get('answer'),
            'answer_type': item.get('answer_type', 'OPEN').upper()
        }
        if 'test' in item.get('phrase_type', ''):
            test_rows.append(entry)
        else:
            train_rows.append(entry)

    # --- Create DataFrames from the question-level split ---
    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)

    # --- Create and Save CSV Files ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save train.csv
    train_csv_path = os.path.join(args.output_dir, "train.csv")
    df_train[['image_name', 'question', 'answer']].to_csv(train_csv_path, index=False)
    
    # Split test data into open and close and save
    df_open = df_test[df_test['answer_type'] == 'OPEN']
    df_close = df_test[df_test['answer_type'] == 'CLOSED']
    
    open_csv_path = os.path.join(args.output_dir, "test_open.csv")
    close_csv_path = os.path.join(args.output_dir, "test_close.csv")
    
    df_open[['image_name', 'question', 'answer']].to_csv(open_csv_path, index=False)
    df_close[['image_name', 'question', 'answer']].to_csv(close_csv_path, index=False)

    print("\n--- Pre-processing Complete ---")
    print(f"Total training questions: {len(df_train)}")
    print(f"Total testing questions: {len(df_test)}")
    print(f"  - Open-ended test questions: {len(df_open)}")
    print(f"  - Close-ended test questions: {len(df_close)}")
    print(f"\nSaved files to: {args.output_dir}")

if __name__ == "__main__":
    main()
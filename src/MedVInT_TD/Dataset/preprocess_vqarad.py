import json
import pandas as pd
import argparse
import os

def main():
    """
    Reads the single, official VQA_RAD JSON file, splits it into train and test sets
    based on the 'phrase_type' key, and further splits the test set into open-ended
    and close-ended questions, saving them as separate CSV files.
    """
    parser = argparse.ArgumentParser(description="Pre-process the VQA-RAD dataset from a single JSON file.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        default="/workspace/Data/VQA_RAD/VQA_RAD_Dataset.json", 
        help="Path to the original VQA_RAD_Dataset.json file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/workspace/Data/VQA_RAD/", 
        help="Directory to save the output CSV files."
    )
    args = parser.parse_args()

    print(f"Reading data from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load()

    train_data = []
    test_data = []
    
    # Split data based on whether 'phrase_type' contains the word 'test'
    for item in data:
        entry = {
            'img_name': item.get('image_name'),
            'question': item.get('question'),
            'answer': item.get('answer'),
            'answer_type': item.get('answer_type', 'OPEN').upper()
        }
        if 'test' in item.get('phrase_type', ''):
            test_data.append(entry)
        else:
            train_data.append(entry)

    # --- Process and Save Training Data ---
    df_train = pd.DataFrame(train_data)
    train_csv_path = os.path.join(args.output_dir, "train.csv")
    df_train[['img_name', 'question', 'answer']].to_csv(train_csv_path, index=False)
    print(f"Saved {len(df_train)} training samples to: {train_csv_path}")
    
    # --- Process and Split Test Data ---
    df_test = pd.DataFrame(test_data)
    df_open = df_test[df_test['answer_type'] == 'OPEN']
    df_close = df_test[df_test['answer_type'] == 'CLOSED']
    
    open_csv_path = os.path.join(args.output_dir, "test_open.csv")
    close_csv_path = os.path.join(args.output_dir, "test_close.csv")
    
    df_open[['img_name', 'question', 'answer']].to_csv(open_csv_path, index=False)
    df_close[['img_name', 'question', 'answer']].to_csv(close_csv_path, index=False)
    
    print(f"\nProcessing of test data complete!")
    print(f"Total test samples: {len(df_test)}")
    print(f" - Saved {len(df_open)} open-ended questions to: {open_csv_path}")
    print(f" - Saved {len(df_close)} close-ended questions to: {close_csv_path}")

if __name__ == "__main__":
    main()
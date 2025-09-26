import json
import pandas as pd
import argparse
import os

def process_json_to_dataframe(json_path):
    """Reads a VQA-RAD JSON file and converts it to a pandas DataFrame."""
    print(f"Reading data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        processed_data.append({
            'img_name': item['image_name'],
            'question': item['question'],
            'answer': item['answer'],
            'answer_type': item.get('answer_type', 'OPEN').upper() # Default to OPEN if not present
        })
    return pd.DataFrame(processed_data)

def main():
    parser = argparse.ArgumentParser(description="Pre-process the VQA-RAD dataset for training and testing.")
    parser.add_argument("--train_json", required=True, help="Path to the original trainset.json file.")
    parser.add_argument("--test_json", required=True, help="Path to the original testset.json file.")
    parser.add_argument("--output_dir", default="./Data/VQA_RAD/", help="Directory to save the output CSV files.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Process Training Data ---
    df_train = process_json_to_dataframe(args.train_json)
    train_csv_path = os.path.join(args.output_dir, "train.csv")
    # Save only the required columns
    df_train[['img_name', 'question', 'answer']].to_csv(train_csv_path, index=False)
    print(f"Saved {len(df_train)} training samples to: {train_csv_path}")

    # --- Process and Split Test Data ---
    df_test = process_json_to_dataframe(args.test_json)
    
    # The official VQA-RAD split is based on the 'answer_type' field
    df_open = df_test[df_test['answer_type'] == 'OPEN']
    df_close = df_test[df_test['answer_type'] == 'CLOSED']

    # Save split files
    open_csv_path = os.path.join(args.output_dir, "test_open.csv")
    close_csv_path = os.path.join(args.output_dir, "test_close.csv")
    
    # Save only the required columns for consistency
    df_open[['img_name', 'question', 'answer']].to_csv(open_csv_path, index=False)
    df_close[['img_name', 'question', 'answer']].to_csv(close_csv_path, index=False)

    print(f"\nProcessing of test data complete!")
    print(f"Total test samples: {len(df_test)}")
    print(f" - Saved {len(df_open)} open-ended questions to: {open_csv_path}")
    print(f" - Saved {len(df_close)} close-ended questions to: {close_csv_path}")

if __name__ == "__main__":
    main()
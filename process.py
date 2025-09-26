import json
import pandas as pd
import argparse
import os

def main():
    """
    Reads the original VQA-RAD JSON data, splits it into open-ended and
    close-ended questions, and saves them as separate CSV files.
    """
    parser = argparse.ArgumentParser(description="Pre-process the VQA-RAD dataset.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True, 
        help="Path to the original VQA-RAD testset.json file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./Data/VQA_RAD/", 
        help="Directory to save the output CSV files."
    )
    args = parser.parse_args()

    print(f"Reading data from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    open_ended = []
    close_ended = []

    for item in data:
        # The official VQA-RAD split defines 'yes/no' questions as close-ended.
        # All other questions are considered open-ended.
        if item['answer_type'].lower() == 'yes/no':
            close_ended.append({
                'img_name': item['image_name'],
                'question': item['question'],
                'answer': item['answer']
            })
        else:
            open_ended.append({
                'img_name': item['image_name'],
                'question': item['question'],
                'answer': item['answer']
            })

    # Create DataFrames
    df_open = pd.DataFrame(open_ended)
    df_close = pd.DataFrame(close_ended)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to CSV files
    open_csv_path = os.path.join(args.output_dir, "test_open.csv")
    close_csv_path = os.path.join(args.output_dir, "test_close.csv")
    
    df_open.to_csv(open_csv_path, index=False)
    df_close.to_csv(close_csv_path, index=False)

    print(f"\nProcessing complete!")
    print(f"Total samples: {len(data)}")
    print(f" - Saved {len(df_open)} open-ended questions to: {open_csv_path}")
    print(f" - Saved {len(df_close)} close-ended questions to: {close_csv_path}")


if __name__ == "__main__":
    main()
import os
import json
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    """
    A unified script to fully process the raw VQA-RAD dataset.
    
    The official dataset can be downloaded from: https://osf.io/89kps/files/osfstorage
    
    This script performs two main actions:
    1. Processes the JSON data to create train.csv, test_open.csv, and test_close.csv
       using the official question-level split.
    2. Copies the corresponding image files into 'train' and 'test' subdirectories.
    """
    parser = argparse.ArgumentParser(description="Prepare the VQA-RAD dataset from raw files.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True, 
        help="Path to the original VQA_RAD_Dataset.json file."
    )
    parser.add_argument(
        "--image_folder", 
        type=str, 
        required=True, 
        help="Path to the top-level VQA_RAD_Image_Folder containing all unsorted images."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the directory where the output CSV files will be saved."
    )
    args = parser.parse_args()

    print(f"Reading data from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['answer_type'] = df['answer_type'].str.upper()

    # --- 1. Create DataFrames based on Question-Level Split ---
    print("\n--- Step 1: Creating DataFrames based on question-level split ---")
    
    train_rows = [item for item in data if 'test' not in item.get('phrase_type', '')]
    test_rows = [item for item in data if 'test' in item.get('phrase_type', '')]

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)
    
    df_train['answer_type'] = df_train['answer_type'].str.upper()
    df_test['answer_type'] = df_test['answer_type'].str.upper()

    # --- 2. Identify Unique Images for Each Split ---
    train_images_needed = set(df_train['image_name'].unique())
    test_images_needed = set(df_test['image_name'].unique())

    # --- 3. Organize Image Files by COPYING ---
    print("\n--- Step 2: Copying images into train/test subdirectories ---")
    train_image_dir = os.path.join(args.image_folder, 'train')
    test_image_dir = os.path.join(args.image_folder, 'test')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    print(f"Found {len(train_images_needed)} unique images for the training set.")
    for image_name in tqdm(train_images_needed, desc="Copying train images"):
        src = os.path.join(args.image_folder, image_name)
        dst = os.path.join(train_image_dir, image_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"Found {len(test_images_needed)} unique images for the test set.")
    for image_name in tqdm(test_images_needed, desc="Copying test images"):
        src = os.path.join(args.image_folder, image_name)
        dst = os.path.join(test_image_dir, image_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            
    print("Image organization complete.")

    # --- 4. Create and Save Final CSV Files ---
    print("\n--- Step 3: Creating train.csv, test_open.csv, and test_close.csv ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    df_train[['image_name', 'question', 'answer']].to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    
    df_test_open = df_test[df_test['answer_type'] == 'OPEN']
    df_test_close = df_test[df_test['answer_type'] == 'CLOSED']
    
    df_test_open[['image_name', 'question', 'answer']].to_csv(os.path.join(args.output_dir, "test_open.csv"), index=False)
    df_test_close[['image_name', 'question', 'answer']].to_csv(os.path.join(args.output_dir, "test_close.csv"), index=False)

    print(f"\n--- Final Data Summary ---")
    print(f"Training Questions: {len(df_train)}")
    print(f"Test Questions: {len(df_test)}")
    print(f"  - Open: {len(df_test_open)}")
    print(f"  - Close: {len(df_test_close)}")
    print(f"Images in train folder: {len(os.listdir(train_image_dir))}")
    print(f"Images in test folder: {len(os.listdir(test_image_dir))}")
    print(f"\n✅ All pre-processing tasks are complete.")

if __name__ == "__main__":
    main()
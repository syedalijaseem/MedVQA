import os
import json
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    """
    A unified script to pre-process the raw VQA-RAD dataset.
    
    The official dataset can be downloaded from: https://osf.io/89kps/files/osfstorage
    
    This script performs two main actions:
    1. Splits the image files inside the "VQA_RAD Image Folder" into 'train' and 'test' subdirectories 
       based on an image-wise split determined by the JSON file "VQA_RAD_Dataset.json".
    2. Processes the JSON data to create train.csv, test_open.csv, and test_close.csv.
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

    # --- 1. Split Images ---
    print("--- Step 1: Splitting images into train/test sets ---")
    train_image_path = os.path.join(args.image_folder, 'train')
    test_image_path = os.path.join(args.image_folder, 'test')

    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    print(f"Ensured directories exist: '{train_image_path}' and '{test_image_path}'")

    with open(args.json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    test_types = ['test_freeform', 'test_para']
    test_images = set(df[df['phrase_type'].isin(test_types)]['image_name'].unique())
    all_images = set(df['image_name'].unique())
    train_images = all_images - test_images

    print(f"Found {len(all_images)} total unique images.")
    print(f"Identified {len(test_images)} unique images for the test set.")
    print(f"Identified {len(train_images)} unique images for the training set.")

    for image_name in tqdm(test_images, desc="Moving test images"):
        src = os.path.join(args.image_folder, image_name)
        dst = os.path.join(test_image_path, image_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    for image_name in tqdm(train_images, desc="Moving train images"):
        src = os.path.join(args.image_folder, image_name)
        dst = os.path.join(train_image_path, image_name)
        if os.path.exists(src):
            shutil.move(src, dst)
    
    print("Image splitting complete.")

    # --- 2. Create CSV Files ---
    print("\n--- Step 2: Creating train.csv, test_open.csv, and test_close.csv ---")
    
    train_rows = [item for item in data if 'test' not in item.get('phrase_type', '')]
    test_rows = [item for item in data if 'test' in item.get('phrase_type', '')]

    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)
    
    df_open = df_test[df_test['answer_type'].str.upper() == 'OPEN']
    df_close = df_test[df_test['answer_type'].str.upper() == 'CLOSED']
    
    # Save CSVs to the specified output directory
    os.makedirs(args.output_dir, exist_ok=True)
    df_train[['image_name', 'question', 'answer']].to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    df_open[['image_name', 'question', 'answer']].to_csv(os.path.join(args.output_dir, "test_open.csv"), index=False)
    df_close[['image_name', 'question', 'answer']].to_csv(os.path.join(args.output_dir, "test_close.csv"), index=False)
    
    print(f"Saved {len(df_train)} training samples to: {os.path.join(args.output_dir, 'train.csv')}")
    print(f"Saved {len(df_open)} open-ended test samples to: {os.path.join(args.output_dir, 'test_open.csv')}")
    print(f"Saved {len(df_close)} close-ended test samples to: {os.path.join(args.output_dir, 'test_close.csv')}")

    print("\nAll pre-processing tasks are complete.")

if __name__ == "__main__":
    main()
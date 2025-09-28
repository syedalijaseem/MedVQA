import argparse
import os
import csv
import transformers
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm

from Dataset.VQA_RAD_Dataset import VQA_RAD_Dataset
from models.QA_model import QA_model

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="./LLaMA/7B_hf")
    ckp: Optional[str] = field(default="./Results/QA_no_pretrain_no_aug/VQA_RAD/checkpoint-16128")
    
    # FIXED: Added the missing 'checkpointing' argument required by the QA_model.
    # It should be False during evaluation.
    checkpointing: Optional[bool] = field(default=False)
    
    # Q_former parameters
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    
    # Basic model settings
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    
    # Image Encoder settings
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='./img_checkpoint/PMC-CLIP/checkpoint.pt')
    
    # PEFT (LoRA) settings
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    img_dir: str = field(default='./Data/VQA_RAD_Image_Folder/', metadata={"help": "Path to the image directory."})
    Test_csv_path: str = field(default='./Data/VQA_RAD/test.csv', metadata={"help": "Path to the test data csv."})
    tokenizer_path: str = field(default='./LLaMA/tokenizer', metadata={"help": "Path to the tokenizer."})

# REFACTORED: Created a single function to handle evaluation logic
def run_evaluation(model, dataloader, tokenizer, device, output_filename):
    """Runs model inference on the provided dataloader and saves results to a CSV."""
    model.eval()
    with open(output_filename, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Question', 'Prediction', 'Label', 'Image_Path'])
        
        for sample in tqdm.tqdm(dataloader, desc=f"Evaluating {os.path.basename(output_filename)}"):
            # FIXED: Directly use the tokenized tensor from the dataloader
            input_ids = sample['input_ids'].to(device)
            images = sample['images'].to(device)
            
            with torch.no_grad():
                # Assuming `generate` is the correct method name in your QA_model
                generation_ids = model.generate(input_ids=input_ids, images=images, max_new_tokens=256)
            
            # The generated IDs include the prompt, so we slice it off
            prompt_len = input_ids.shape[1]
            generated_ids_only = generation_ids[:, prompt_len:]
            
            generated_texts = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)
            
            # Unpack results for this batch
            questions = sample['question']
            labels = sample['answer']
            img_names = [os.path.basename(p) for p in sample.get('img_name', ['N/A']*len(questions))]

            for i in range(len(generated_texts)):
                writer.writerow([questions[i], generated_texts[i], labels[i], img_names[i]])

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setup Model")
    # Correctly build the path to the model binary
    ckp_path = os.path.join(model_args.ckp, 'pytorch_model.bin')
    print(f"Loading checkpoint from: {ckp_path}")

    model = QA_model(model_args)
    ckpt = torch.load(ckp_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)
    
    tokenizer = VQA_RAD_Dataset.tokenizer # Re-use the tokenizer from the class for decoding

    # --- Setup for "close" answer test ---
    print("Setting up 'close' answer dataset")
    test_close_csv = data_args.Test_csv_path.replace('.csv', '_close.csv')
    # UPDATED: Correctly initialize the dataset
    test_dataset_close = VQA_RAD_Dataset(
        csv_path=test_close_csv,
        tokenizer_path=data_args.tokenizer_path,
        img_dir=data_args.img_dir,
        mode='Test'
    )
    test_dataloader_close = DataLoader(test_dataset_close, batch_size=8, shuffle=False, num_workers=4)
    
    # --- Setup for "open" answer test ---
    print("Setting up 'open' answer dataset")
    test_open_csv = data_args.Test_csv_path.replace('.csv', '_open.csv')
    # UPDATED: Correctly initialize the dataset
    test_dataset_open = VQA_RAD_Dataset(
        csv_path=test_open_csv,
        tokenizer_path=data_args.tokenizer_path,
        img_dir=data_args.img_dir,
        mode='Test'
    )
    test_dataloader_open = DataLoader(test_dataset_open, batch_size=8, shuffle=False, num_workers=4)

    # --- Run Evaluations ---
    # Define output filenames based on the checkpoint directory
    base_output_name = os.path.basename(os.path.normpath(model_args.ckp))
    
    output_file_close = f"results_close_{base_output_name}.csv"
    output_file_open = f"results_open_{base_output_name}.csv"
    
    run_evaluation(model, test_dataloader_close, tokenizer, device, output_file_close)
    run_evaluation(model, test_dataloader_open, tokenizer, device, output_file_open)

    print("Evaluation complete. Results saved to CSV files.")

if __name__ == "__main__":
    main()
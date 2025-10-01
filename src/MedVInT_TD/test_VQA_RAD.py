import argparse
import os
import csv
import transformers
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm
# Import the safetensors library
from safetensors.torch import load_file as load_safetensors

from Dataset.VQA_RAD_Dataset import VQA_RAD_Dataset
from models.QA_model import QA_model

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="./LLaMA/7B_hf")
    ckp: Optional[str] = field(default="./Results/QA_no_pretrain_no_aug/VQA_RAD/checkpoint-16128")
    checkpointing: Optional[bool] = field(default=False)
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='./img_checkpoint/PMC-CLIP/checkpoint.pt')
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    img_dir: str = field(default='./Data/VQA_RAD/VQA_RAD_Image_Folder/', metadata={"help": "Path to the image directory."})
    Test_csv_path: str = field(default='./Data/VQA_RAD/test.csv', metadata={"help": "Path to the test data csv."})
    tokenizer_path: str = field(default='./LLaMA/tokenizer', metadata={"help": "Path to the tokenizer."})

def run_evaluation(model, dataloader, tokenizer, device, output_filename):
    """Runs model inference on the provided dataloader and saves results to a CSV."""
    model.eval()
    with open(output_filename, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Question', 'Prediction', 'Label', 'Image_Name'])
        
        for sample in tqdm.tqdm(dataloader, desc=f"Evaluating {os.path.basename(output_filename)}"):
            input_ids = sample['input_ids'].to(device)
            images = sample['images'].to(device)
            
            with torch.no_grad():
                generation_ids = model.generate_long_sentence(input_ids, images)
            
            generated_texts = tokenizer.batch_decode(generation_ids, skip_special_tokens=True)
            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            predictions = [text.replace(prompt, '').strip() for text, prompt in zip(generated_texts, prompts)]

            questions = sample['question']
            labels = sample['answer']
            image_names = sample['image_name']

            for i in range(len(predictions)):
                writer.writerow([questions[i], predictions[i], labels[i], image_names[i]])

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Setting up Model ---")
    model = QA_model(model_args)
    
    # FIXED: Added logic to handle both .bin and .safetensors files
    bin_path = os.path.join(model_args.ckp, 'pytorch_model.bin')
    safetensors_path = os.path.join(model_args.ckp, 'model.safetensors')

    if os.path.exists(bin_path):
        ckp_path = bin_path
        print(f"Loading adapter checkpoint from: {ckp_path}")
        ckpt = torch.load(ckp_path, map_location='cpu')
    elif os.path.exists(safetensors_path):
        ckp_path = safetensors_path
        print(f"Loading adapter checkpoint from: {ckp_path}")
        ckpt = load_safetensors(ckp_path, device='cpu')
    else:
        raise FileNotFoundError(f"No checkpoint file found at {model_args.ckp}. Looked for 'pytorch_model.bin' and 'model.safetensors'.")

    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    
    print(f"Loading tokenizer from: {data_args.tokenizer_path}")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(data_args.tokenizer_path)

    print("\n--- Setting up Datasets ---")
    test_image_dir = os.path.join(data_args.img_dir, 'test')
    
    test_close_csv = data_args.Test_csv_path.replace('test.csv', 'test_close.csv')
    test_dataset_close = VQA_RAD_Dataset(
        csv_path=test_close_csv, tokenizer_path=data_args.tokenizer_path,
        img_dir=test_image_dir, mode='Test'
    )
    test_dataloader_close = DataLoader(test_dataset_close, batch_size=8, shuffle=False, num_workers=0)
    
    test_open_csv = data_args.Test_csv_path.replace('test.csv', 'test_open.csv')
    test_dataset_open = VQA_RAD_Dataset(
        csv_path=test_open_csv, tokenizer_path=data_args.tokenizer_path,
        img_dir=test_image_dir, mode='Test'
    )
    test_dataloader_open = DataLoader(test_dataset_open, batch_size=8, shuffle=False, num_workers=0)

    print("\n--- Running Evaluations ---")
    base_output_name = os.path.basename(os.path.normpath(model_args.ckp))
    output_file_close = f"results_close_{base_output_name}.csv"
    output_file_open = f"results_open_{base_output_name}.csv"
    
    run_evaluation(model, test_dataloader_close, tokenizer, device, output_file_close)
    run_evaluation(model, test_dataloader_open, tokenizer, device, output_file_open)

    print("\nEvaluation complete. Results saved to CSV files.")

if __name__ == "__main__":
    main()
import torch
import argparse
import csv
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project source to path to allow imports
sys.path.append('/workspace/PMC-VQA/src/MedVInT_TD')
from models.QA_model_infer import QA_model_infer
from Dataset.VQA_RAD_Dataset import VQA_RAD_Dataset

def run_evaluation(model, dataloader, tokenizer, output_filename):
    model.eval()
    device = model.device
    with open(output_filename, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Question', 'Prediction', 'Label', 'Image_Name'])
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device, dtype=torch.float16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.no_grad():
                output_ids = model.generate(input_ids=input_ids, images=images, attention_mask=attention_mask)
            
            # Decode only the newly generated tokens
            generated_tokens = output_ids[:, input_ids.shape[1]:]
            predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for i in range(len(predictions)):
                writer.writerow([batch['question'][i], predictions[i].strip(), batch['answer'][i], batch['image_name'][i]])

def main():
    parser = argparse.ArgumentParser(description="Run 4-bit batch evaluation on VQA-RAD.")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--pmc_clip_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_csv_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Tokenizer (with the use_fast=False fix)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load 4-bit Model and Adapter
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
    base_llm = AutoModelForCausalLM.from_pretrained(args.base_model_path, quantization_config=bnb_config, device_map='auto')
    
    # 3. CRITICAL: Set the pad_token_id on the model's generation config
    base_llm.config.pad_token_id = tokenizer.pad_token_id
    
    llm_with_lora = PeftModel.from_pretrained(base_llm, args.adapter_path, is_trainable=False, device_map='auto')
    
    # 4. Instantiate the inference model
    model = QA_model_infer(llm=llm_with_lora, pmc_clip_ckpt_path=args.pmc_clip_path)
    
    # 5. Load the non-LLM weights (Q-Former, etc.)
    ckpt_file = Path(args.adapter_path) / 'adapter_model.bin'
    if ckpt_file.exists():
        sd = torch.load(ckpt_file, map_location='cpu')
        for name, module in model.named_children():
            if name != 'llamacasual':
                module_weights = {k.replace(f'{name}.', ''): v for k, v in sd.items() if k.startswith(f'{name}.')}
                if module_weights:
                    module.load_state_dict(module_weights, strict=False)
    
    # 6. Setup Dataset and run evaluation
    test_dataset = VQA_RAD_Dataset(csv_path=args.test_csv_path, tokenizer_path=args.tokenizer_path, img_dir=args.img_dir, mode='Test')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0) # num_workers=0 is safest
    
    output_filename = f"evaluation_results_{Path(args.adapter_path).name}.csv"
    run_evaluation(model, test_dataloader, tokenizer, output_filename)
    print(f"Evaluation complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
import os
import transformers
from dataclasses import dataclass, field
from typing import Optional
import torch
from Dataset.VQA_RAD_Dataset import VQA_RAD_Dataset
from Dataset.Slake_Dataset import Slake_Dataset
from models.QA_model import QA_model
from transformers import Trainer

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="./LLaMA/7B_hf")
    ckp: Optional[str] = field(default="./MedVInT_be_pre-trained/checkpoint-13000", metadata={"help": "Path to the pre-trained checkpoint directory."})
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    checkpointing: Optional[bool] = field(default=True)
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='./img_checkpoint/PMC-CLIP/checkpoint.pt')
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

@dataclass
class DataArguments:
    img_dir: str = field(default='./Data/VQA_RAD_Image_Folder/', metadata={"help": "Path to the image directory."})
    dataset_name: str = field(default='vqa_rad', metadata={"help": "The name of the dataset to use (vqa_rad or slake)."})
    Train_csv_path: str = field(default='./Data/VQA_RAD/train.csv', metadata={"help": "Path to the training data."})
    Eval_csv_path: str = field(default='./Data/VQA_RAD/test.csv', metadata={"help": "Path to the evaluation data."})
    tokenizer_path: str = field(default='./LLaMA/tokenizer', metadata={"help": "Path to the tokenizer."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Setup Data")
    if data_args.dataset_name == 'vqa_rad':
        training_args.run_name = training_args.run_name or 'finetune_vqa_rad'
        training_args.output_dir = os.path.join(training_args.output_dir, 'VQA_RAD_finetune')
        Train_dataset = VQA_RAD_Dataset(
            csv_path=data_args.Train_csv_path,
            tokenizer_path=data_args.tokenizer_path,
            img_dir=data_args.img_dir,
            mode='Train'
        )
        Eval_dataset = VQA_RAD_Dataset(
            csv_path=data_args.Eval_csv_path,
            tokenizer_path=data_args.tokenizer_path,
            img_dir=data_args.img_dir,
            mode='Test'
        )
    elif data_args.dataset_name == 'slake':
        training_args.run_name = training_args.run_name or 'finetune_slake'
        training_args.output_dir = os.path.join(training_args.output_dir, 'Slake_finetune')
        Train_dataset = Slake_Dataset(
            csv_path=data_args.Train_csv_path,
            tokenizer_path=data_args.tokenizer_path,
            img_dir=data_args.img_dir,
            mode='Train'
        )
        Eval_dataset = Slake_Dataset(
            csv_path=data_args.Eval_csv_path,
            tokenizer_path=data_args.tokenizer_path,
            img_dir=data_args.img_dir,
            mode='Test'
        )
    else:
        raise ValueError(f"Unknown dataset: {data_args.dataset_name}")

    print("Setup Model")
    ckp_path = os.path.join(model_args.ckp, 'pytorch_model.bin')
    print(f"Loading pre-trained weights from: {ckp_path}")
    
    model = QA_model(model_args)
    
    # FIXED: Added strict=False to correctly load the LoRA adapter
    model.load_state_dict(torch.load(ckp_path, map_location='cpu'), strict=False)

    print('Start training')
    trainer = Trainer(
        model=model,
        train_dataset=Train_dataset,
        eval_dataset=Eval_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    main()
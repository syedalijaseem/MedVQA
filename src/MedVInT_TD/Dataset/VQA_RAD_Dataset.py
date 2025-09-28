import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import LlamaTokenizer
import os
from .randaugment import RandomAugment

class VQARADDataset(Dataset):
    """
    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        tokenizer_path (str): Path to the pretrained Hugging Face tokenizer directory.
        img_dir (str): Directory where the images are stored.
        seq_length (int): The maximum sequence length for text inputs.
        img_tokens (int): The number of tokens to reserve for the image features.
        mode (str): The dataset mode, either 'Train' or 'Test'.
    """
    def __init__(self, csv_path: str, tokenizer_path: str, img_dir: str,
                 img_tokens: int = 32, seq_length: int = 512, mode: str = 'Train'):
        
        super().__init__()
        self.img_root = img_dir
        self.data = pd.read_csv(csv_path)
        
        if mode not in ['Train', 'Test']:
            raise ValueError("Mode must be either 'Train' or 'Test'")
        self.mode = mode
        
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.seq_length = seq_length
        self.image_size = (512, 512)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        if self.mode == 'Train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                RandomAugment(2, 7, isPIL=True, augs=[
                    'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness'
                ]),
                transforms.ToTensor(),
                normalize,
            ])
        else: # Test mode
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
            
        self.img_padding = [-100] * img_tokens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data.iloc[idx]
        question = sample['question']
        answer = str(sample['answer'])
        
        # FIXED #1: Use 'image_name' to match the CSV header
        image_name = sample['image_name']
        
        img_path = os.path.join(self.img_root, image_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping to next item.")
            return self.__getitem__((idx + 1) % len(self))

        prompt = 'Question: ' + question + 'The Answer is:'

        if self.mode == 'Train':
            full_text = prompt + answer
            tokenized_full_text = self.tokenizer(
                full_text, max_length=self.seq_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids = tokenized_full_text.input_ids.squeeze(0)
            labels = input_ids.clone()
            
            tokenized_prompt = self.tokenizer(
                prompt, max_length=self.seq_length, truncation=True, return_tensors='pt'
            )
            prompt_len = tokenized_prompt.input_ids.squeeze(0).shape[0]

            labels[:prompt_len] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            final_labels = torch.cat([
                torch.tensor(self.img_padding, dtype=torch.long), labels
            ], dim=0)

            return {
                'input_ids': input_ids,
                'images': image_tensor,
                'labels': final_labels,
            }
        else: # Test mode
            tokenized_prompt = self.tokenizer(
                prompt, max_length=self.seq_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids = tokenized_prompt.input_ids.squeeze(0)

            # FIXED #2: Added 'image_name' to the dictionary returned in Test mode
            return {
                'input_ids': input_ids,
                'images': image_tensor,
                'answer': answer,
                'question': question,
                'image_name': image_name, 
            }
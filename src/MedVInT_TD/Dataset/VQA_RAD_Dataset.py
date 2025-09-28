import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import LlamaTokenizer
from .randaugment import RandomAugment

class VQA_RAD_Dataset(Dataset):
    """
    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        tokenizer_path (str): Path to the pretrained Hugging Face tokenizer directory.
        img_dir (str): Directory where the images are stored.
        seq_length (int): The maximum sequence length for text inputs.
        img_tokens (int): The number of tokens to reserve for the image features.
        mode (str): The dataset mode, either 'Train' or 'Test'.
        start (int): The starting index to slice the dataframe.
    """
    def __init__(self, csv_path: str, tokenizer_path: str, img_dir: str = './Data/VQA_RAD/VQA_RAD_Image_Folder/',
                 img_tokens: int = 32, seq_length: int = 512, mode: str = 'Train', start: int = 0):
        
        super().__init__()
        self.img_root = img_dir
        self.data = pd.read_csv(csv_path).iloc[start:]
        
        # Validate mode
        if mode not in ['Train', 'Test']:
            raise ValueError("Mode must be either 'Train' or 'Test'")
        self.mode = mode
        
        # --- Tokenizer Setup ---
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # A common practice for autoregressive models
        
        self.seq_length = seq_length
        
        # --- Image Transforms Setup ---
        self.image_size = (512, 512)
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        if self.mode == 'Train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
                transforms.ToTensor(),
                normalize,
            ])
        else: # Test mode
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
            
        # For creating labels that align with a multimodal model's output
        # This masks the loss for the image part of the sequence
        self.img_padding = [-100] * img_tokens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data.iloc[idx]
        question = sample['question']
        answer = str(sample['answer'])  # Ensure answer is a string
        
        # --- Image Processing ---
        img_path = f"{self.img_root}/{sample['image_name']}"
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            # Return a placeholder or handle appropriately
            return self.__getitem__((idx + 1) % len(self))

        # --- Text Processing ---
        # Replicating the EXACT prompt from pre-training, including the missing space.
        prompt = 'Question: ' + question + 'The Answer is:'

        if self.mode == 'Train':
            # For training, we need input_ids (prompt + answer) and labels (only the answer part)
            full_text = prompt + answer

            # Tokenize the full sequence
            tokenized_full_text = self.tokenizer(
                full_text,
                max_length=self.seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokenized_full_text.input_ids.squeeze(0)

            # Create labels by cloning input_ids and masking non-answer parts
            labels = input_ids.clone()
            
            # Tokenize the prompt to find its length for masking
            tokenized_prompt = self.tokenizer(
                prompt,
                max_length=self.seq_length,
                truncation=True,
                return_tensors='pt'
            )
            prompt_len = tokenized_prompt.input_ids.squeeze(0).shape[0]

            # Mask the prompt part in the labels
            labels[:prompt_len] = -100
            
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100

            # Prepend image padding to the labels tensor
            final_labels = torch.cat([
                torch.tensor(self.img_padding, dtype=torch.long),
                labels
            ], dim=0)

            return {
                'input_ids': input_ids,
                'images': image_tensor,
                'labels': final_labels,
            }
        else: # Test mode
            # For testing, we only provide the prompt as input and keep the answer for evaluation
            tokenized_prompt = self.tokenizer(
                prompt,
                max_length=self.seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokenized_prompt.input_ids.squeeze(0)

            return {
                'input_ids': input_ids,
                'images': image_tensor,
                'answer': answer,  # Return raw answer for evaluation
                'question': question,
            }
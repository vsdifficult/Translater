import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

class TranslationDataset(Dataset):
    """Dataset for machine translation."""
    
    def __init__(self, tokenizer, config, split="train"):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset(
            config.dataset_name, 
            config.dataset_config,
            split=split
        )
        
        # Preprocess the dataset
        self.processed_data = self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess and tokenize the dataset."""
        processed_data = []
        
        for example in tqdm(self.dataset, desc=f"Preprocessing {self.split} data"):
            source_text = example[self.config.source_column]
            target_text = example[self.config.target_column]
            
            # Tokenize source text
            source_tokens = self.tokenizer.encode_source(
                source_text, 
                max_length=self.config.max_source_length,
                padding='max_length',
                truncation=True
            )
            
            # Tokenize target text
            target_tokens = self.tokenizer.encode_target(
                target_text,
                max_length=self.config.max_target_length,
                padding='max_length',
                truncation=True
            )
            
            processed_data.append({
                'source_ids': source_tokens['input_ids'],
                'source_mask': source_tokens['attention_mask'],
                'target_ids': target_tokens['input_ids'],
                'target_mask': target_tokens['attention_mask'],
                'source_text': source_text,
                'target_text': target_text
            })
            
        return processed_data
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return {
            'source_ids': torch.tensor(self.processed_data[idx]['source_ids']),
            'source_mask': torch.tensor(self.processed_data[idx]['source_mask']),
            'target_ids': torch.tensor(self.processed_data[idx]['target_ids']),
            'target_mask': torch.tensor(self.processed_data[idx]['target_mask']),
            'source_text': self.processed_data[idx]['source_text'],
            'target_text': self.processed_data[idx]['target_text']
        }

def get_translation_dataloader(tokenizer, config, split="train", shuffle=True):
    """Create a dataloader for the translation dataset."""
    dataset = TranslationDataset(tokenizer, config, split)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers
    )
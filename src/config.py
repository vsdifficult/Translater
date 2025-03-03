import os
import json
from dataclasses import dataclass, field, asdict

@dataclass
class TranslationConfig:
    """Configuration for the neural translation model."""
    
    # Model parameters
    encoder_model_name: str = "bert-base-multilingual-cased"
    decoder_model_name: str = "bert-base-multilingual-cased"
    
    # Tokenizer parameters
    source_tokenizer_name: str = "bert-base-multilingual-cased"
    target_tokenizer_name: str = "bert-base-multilingual-cased"
    
    # Languages
    source_lang: str = "en"
    target_lang: str = "ru"
    
    # Dataset parameters
    dataset_name: str = "opus100"
    dataset_config: str = "en-ru"
    source_column: str = "source"
    target_column: str = "target"
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_workers: int = 4
    
    # Model hyperparameters
    dropout_rate: float = 0.1
    hidden_size: int = 768
    
    # Sequence parameters
    max_source_length: int = 128
    max_target_length: int = 128
    
    # Special token IDs (to be filled during initialization)
    pad_token_id: int = 0
    bos_token_id: int = 101  # [CLS] token for BERT
    eos_token_id: int = 102  # [SEP] token for BERT
    
    # Vocabulary sizes (to be filled during initialization)
    source_vocab_size: int = 0
    target_vocab_size: int = 0
    
    # Paths
    output_dir: str = "./models"
    tensorboard_dir: str = "./runs"
    
    def save(self, output_path):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, config_path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def update_vocab_sizes(self, tokenizer):
        """Update vocabulary sizes based on tokenizer."""
        self.source_vocab_size = tokenizer.get_vocab_size("source")
        self.target_vocab_size = tokenizer.get_vocab_size("target")
from transformers import AutoTokenizer
import os

class TranslationTokenizer:
    """Wrapper around Hugging Face tokenizers for translation tasks."""
    
    def __init__(self, config):
        self.config = config
        
        # Load source tokenizer
        self.source_tokenizer = AutoTokenizer.from_pretrained(
            config.source_tokenizer_name
        )
        
        # Load target tokenizer
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            config.target_tokenizer_name
        )
        
        # Special tokens
        self.source_language_token = f"<{config.source_lang}>"
        self.target_language_token = f"<{config.target_lang}>"
        
        # Add special tokens if they don't exist
        special_tokens = {
            'additional_special_tokens': [
                self.source_language_token,
                self.target_language_token
            ]
        }
        
        self.source_tokenizer.add_special_tokens(special_tokens)
        self.target_tokenizer.add_special_tokens(special_tokens)
        
        # Store important token IDs
        self.source_lang_token_id = self.source_tokenizer.convert_tokens_to_ids(self.source_language_token)
        self.target_lang_token_id = self.target_tokenizer.convert_tokens_to_ids(self.target_language_token)
        
    def encode_source(self, text, **kwargs):
        """Tokenize source text."""
        return self.source_tokenizer(
            f"{self.source_language_token} {text}",
            **kwargs
        )
    
    def encode_target(self, text, **kwargs):
        """Tokenize target text."""
        return self.target_tokenizer(
            f"{self.target_language_token} {text}",
            **kwargs
        )
    
    def decode_source(self, token_ids, skip_special_tokens=True):
        """Decode source token IDs back to text."""
        text = self.source_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        # Remove language token if present
        if not skip_special_tokens and self.source_language_token in text:
            text = text.replace(f"{self.source_language_token} ", "")
        return text
    
    def decode_target(self, token_ids, skip_special_tokens=True):
        """Decode target token IDs back to text."""
        text = self.target_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        # Remove language token if present
        if not skip_special_tokens and self.target_language_token in text:
            text = text.replace(f"{self.target_language_token} ", "")
        return text
    
    def get_vocab_size(self, side="target"):
        """Get vocabulary size for source or target."""
        if side == "source":
            return len(self.source_tokenizer)
        else:
            return len(self.target_tokenizer)
    
    def save_tokenizers(self, output_dir):
        """Save tokenizers to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        source_dir = os.path.join(output_dir, "source_tokenizer")
        target_dir = os.path.join(output_dir, "target_tokenizer")
        
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        
        self.source_tokenizer.save_pretrained(source_dir)
        self.target_tokenizer.save_pretrained(target_dir)
        
        return source_dir, target_dir
    
    @classmethod
    def from_saved(cls, config, model_dir):
        """Load tokenizers from saved files."""
        config.source_tokenizer_name = os.path.join(model_dir, "source_tokenizer")
        config.target_tokenizer_name = os.path.join(model_dir, "target_tokenizer")
        
        return cls(config)
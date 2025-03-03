import os
import argparse
import torch
from tqdm import tqdm

from model import TranslationModel
from tokenizer import TranslationTokenizer
from config import TranslationConfig

class Translator:
    """Translator class for inferencing."""
    
    def __init__(self, source_lang="en", target_lang="ru", model_path=None):
        """Initialize the translator."""
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # If no model path is provided, use the default one
        if model_path is None:
            # Find the latest model
            base_dir = os.path.join("./models", f"{source_lang}_{target_lang}")
            if os.path.exists(base_dir):
                dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))]
                
                if dirs:
                    # Get the latest directory by modification time
                    model_path = max(dirs, key=os.path.getmtime)
                else:
                    raise ValueError(f"No model found for {source_lang}-{target_lang}")
            else:
                raise ValueError(f"No model found for {source_lang}-{target_lang}")
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        self.config = TranslationConfig.load(config_path)
        
        # Load tokenizer
        self.tokenizer = TranslationTokenizer.from_saved(self.config, model_path)
        
        # Update vocab sizes
        self.config.update_vocab_sizes(self.tokenizer)
        
        # Load model
        self.model = TranslationModel(self.config)
        
        # Load weights
        model_weights_path = os.path.join(model_path, "best_model.pt")
        if not os.path.exists(model_weights_path):
            model_weights_path = os.path.join(model_path, "final_model.pt")
        
        self.model.load_state_dict(torch.load(
            model_weights_path,
            map_location=torch.device('cpu')
        ))
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def translate(self, text, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Translate a single text."""
        # Tokenize source text
        tokens = self.tokenizer.encode_source(
            text,
            padding='max_length',
            max_length=self.config.max_source_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        source_ids = tokens['input_ids'].to(self.device)
        source_mask = tokens['attention_mask'].to(self.device)
        
        # Generate translation
        with torch.no_grad():
            output_ids = self.model.generate(
                source_ids,
                source_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode output
        output_text = self.tokenizer.decode_target(output_ids[0])
        
        return output_text
    
    def translate_batch(self, texts, batch_size=8, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Translate a batch of texts."""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            batch_tokens = []
            for text in batch_texts:
                tokens = self.tokenizer.encode_source(
                    text,
                    padding='max_length',
                    max_length=self.config.max_source_length,
                    truncation=True,
                    return_tensors='pt'
                )
                batch_tokens.append(tokens)
            
            # Concatenate tensors
            source_ids = torch.cat([t['input_ids'] for t in batch_tokens], dim=0).to(self.device)
            source_mask = torch.cat([t['attention_mask'] for t in batch_tokens], dim=0).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                output_ids = self.model.generate(
                    source_ids,
                    source_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            
            # Decode outputs
            batch_results = [
                self.tokenizer.decode_target(output_ids[j])
                for j in range(output_ids.size(0))
            ]
            
            results.extend(batch_results)
        
        return results
    
    def translate_file(self, input_file, output_file, batch_size=8, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Translate a text file."""
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Strip newlines
        lines = [line.strip() for line in lines]
        
        # Translate batch by batch
        translations = []
        print(f"Translating {len(lines)} lines from {input_file}...")
        
        for i in tqdm(range(0, len(lines), batch_size)):
            batch_texts = lines[i:i+batch_size]
            batch_translations = self.translate_batch(
                batch_texts,
                batch_size=batch_size,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            translations.extend(batch_translations)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        
        print(f"Translation completed. Output saved to {output_file}")
        
        return translations

def main():
    """Command line interface for the translator."""
    parser = argparse.ArgumentParser(description="Neural Machine Translation")
    parser.add_argument("--source_lang", type=str, default="en", help="Source language code")
    parser.add_argument("--target_lang", type=str, default="ru", help="Target language code")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model directory")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Text to translate")
    input_group.add_argument("--input_file", type=str, help="Input file to translate")
    
    # Output options
    parser.add_argument("--output_file", type=str, help="Output file to save translations")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated translation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for file translation")
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = Translator(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        model_path=args.model_path
    )
    
    # Translate text or file
    if args.text:
        translation = translator.translate(
            args.text,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"Original: {args.text}")
        print(f"Translation: {translation}")
    else:
        if not args.output_file:
            # Default output file name
            base_name = os.path.splitext(args.input_file)[0]
            args.output_file = f"{base_name}.{args.target_lang}"
            
        translator.translate_file(
            args.input_file,
            args.output_file,
            batch_size=args.batch_size,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

if __name__ == "__main__":
    main()
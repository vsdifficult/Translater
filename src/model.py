import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TranslationEncoder(nn.Module):
    """Transformer encoder for the translation task."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = AutoConfig.from_pretrained(config.encoder_model_name)
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TranslationDecoder(nn.Module):
    """Transformer decoder for the translation task."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = AutoConfig.from_pretrained(config.decoder_model_name)
        self.decoder = AutoModel.from_pretrained(config.decoder_model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.output_layer = nn.Linear(model_config.hidden_size, config.target_vocab_size)
        
    def forward(self, input_ids, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        logits = self.output_layer(hidden_states)
        return logits

class TranslationModel(nn.Module):
    """Full translation model combining encoder and decoder."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TranslationEncoder(config)
        self.decoder = TranslationDecoder(config)
        
    def forward(self, 
                source_ids, 
                target_ids,
                source_mask=None,
                target_mask=None):
        encoder_hidden_states = self.encoder(source_ids, source_mask)
        logits = self.decoder(
            target_ids,
            encoder_hidden_states,
            target_mask,
            source_mask
        )
        return logits

    def generate(self, 
                source_ids, 
                source_mask=None, 
                max_length=50, 
                temperature=1.0, 
                top_k=50,
                top_p=0.9):
        """Generate translations using beam search."""
        batch_size = source_ids.size(0)
        encoder_hidden_states = self.encoder(source_ids, source_mask)
        
        # Start with BOS token
        current_ids = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=source_ids.device
        )
        
        for _ in range(max_length):
            logits = self.decoder(
                current_ids, 
                encoder_hidden_states,
                None,  # No target mask for generation
                source_mask
            )
            
            # Get the next token prediction
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # Keep at least one token
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate with previous input
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Stop if EOS token
            if (next_token == self.config.eos_token_id).any():
                break
                
        return current_ids
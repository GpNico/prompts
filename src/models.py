


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM


class ModelWrapper(nn.Module):
    
    def __init__(self, 
                 model_type: str,
                 model_name: str,
                 device: str,
                 output_hidden_states: bool = True):
        """
            Wrapper class around hugging face models that support
            different model_type.
        """
        super(ModelWrapper, self).__init__()
        assert model_type in ['bert', 'xlm']
        
        # attributes
        self.device = device
        self.model_type = model_type
        
        # For now
        config = AutoConfig.from_pretrained(model_name,
                                            output_hidden_states = output_hidden_states)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                          config=config).to(device)
        self.model.eval()
        
        
    def forward(self,
                input_ids: torch.Tensor = None, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
            Forward the model and returns logits.
            
            Args:
                input_ids (torch.Tensor) shape [BS, L]
                attention_mask (torch.Tensor) shape [BS, L]
            Returns:
                logits (torch.Tensor) shape [BS, L, Voc Size]
        
        """

        logits = self.model(input_ids = input_ids,
                            attention_mask = attention_mask).logits
            
        return logits
    
    def embedding(self,
                  input_ids: torch.Tensor = None, 
                  attention_mask: torch.Tensor = None,
                  output_hidden_states: bool = False) -> torch.Tensor:
        """
            Compute embeddings using the model without its LM head.
            
            Args:
                input_ids (torch.Tensor) shape [BS, L]
                attention_mask (torch.Tensor) shape [BS, L]
            Returns:
                embeddings (torch.Tensor) shape [BS, L, Hidden Dim] if output_hidden_states == False
                                                [Num Layers, BS, L, Hidden Dim] if output_hidden_states == True
        
        """
        
        hidden_states = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask).hidden_states
        
        if output_hidden_states:
            return torch.stack(hidden_states) # shape [Num Layers, BS, L, Hidden Dim]
        else:
            return hidden_states[-1] # shape [BS, L, Hidden Dim]
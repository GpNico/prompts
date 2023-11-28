


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelWithLMHead


class ModelWrapper(nn.Module):
    
    def __init__(self, 
                 model_type: str,
                 model_name: str,
                 device: str):
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
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
        self.model.eval()
        
        
    def forward(self,
                input_ids: torch.Tensor = None, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        
        if self.model_type == 'bert':
            logits = self.model(input_ids = input_ids,
                                attention_mask = attention_mask).logits
            
        return logits
    
    def embedding(self,
                  input_ids: torch.Tensor = None, 
                  attention_mask: torch.Tensor = None) -> torch.Tensor:
        
        if self.model_type == 'bert':
            outputs = self.model.bert(
                                input_ids = input_ids,
                                attention_mask = attention_mask
                                )
            
        return outputs
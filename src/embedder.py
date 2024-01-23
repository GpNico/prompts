

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm

from src.models import ModelWrapper
from src.tokenizer import TokenizerWrapper 



class Embedder:
    """
    
    Wrapper Class that deals with everything that has to 
    do with embeddings.
    
    Note that for different (albeit related) tasks like
    NLL computation, it is not done here.

    """
    
    def __init__(self,
                 model: ModelWrapper,
                 tokenizer: TokenizerWrapper,) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
    
    def embed(self,
              df: pd.DataFrame,
              method: int = 1,
              pooling: str = 'avg',
              output_hidden_states: bool = False,
              **kwargs) -> torch.Tensor:
        """
            Compute embeddings of prompts from df.
            
            Args:
                df (pandas Dataframe) contains the LAMA-based dataset (ie. templates, sub_surfaces, obj_srfaces, etc.).
                
                method (int) method number (cf. tokenizer.py for more information)
                
                pooling (str) how to pooled computed embeddings.
                              For now: 'avg' computes the average over the length dim.
                                       'sum' computes the sum over the length dim.
                                       
                output_hidden_states (bool) 
                                       
            Returns:
                embedds (torch.Tensor) shape [Dataset Size, L, Hidden Dim] if output_hidden_states = False
                                             [Num Layers, Dataset Size, L, Hidden Dim] if output_hidden_states = True
        
        """
        if method == 1:
            df = df.drop_duplicates(subset=['template'])
            # Some relations are the same in LAMA
            # Doing this results in only 38
        
        
        embedds = []
        for k in tqdm.tqdm(range(0, len(df), kwargs['batch_size'])): # We'll do it by hand
            # Retrieve elem
            elems = df.iloc[k:k + kwargs['batch_size']]
            sub_surfaces = elems['sub_surface'] # [X]
            obj_surfaces = elems['obj_surface'] # [Y]
            templates = elems['template']
            
            # Tokenization
            input_ids, attention_mask = self.tokenizer.tokenize(
                                                    templates = templates,
                                                    sub_surfaces = sub_surfaces,
                                                    obj_surfaces = obj_surfaces,
                                                    token_sequence = kwargs['token_sequence'],
                                                    method = method
                                                    )
            # Compute filter for sentences where len([Y]) > 1
            labels = self.tokenizer.tokenizer(obj_surfaces.tolist(), 
                                              padding = True, 
                                              return_tensors = 'pt')
            filter = torch.logical_not((labels.attention_mask.sum(axis = 1) > 3))

            # Filter out sentences where len([Y]) > 1
            input_ids = input_ids * filter[:, None]
            input_ids = input_ids[(input_ids != 0).any(dim=1)]
            attention_mask = attention_mask * filter[:, None]
            attention_mask = attention_mask[(attention_mask != 0).any(dim=1)] 
            with torch.no_grad():
                outputs = self.model.embedding(
                                    input_ids = input_ids.to(self.model.device),
                                    attention_mask = attention_mask.to(self.model.device),
                                    output_hidden_states=output_hidden_states
                                    )
            # /!\ IMPORTANT: Removing embeddinsg according to attention mask!
            # Keeping [CLS] & [SEP]; shape [BS, L, Hidden Dim] or [Num Layers, BS, L, Hidden Dim]
            if output_hidden_states:
                _embedds = outputs.cpu() * attention_mask[None,:,:,None] 
            else:
                _embedds = outputs.cpu() * attention_mask[:,:,None] 
                
            if pooling == 'avg':
                embedds.append(_embedds.mean(axis = -2)) # Shape [BS, Hidden Dim] or [Num Layers, BS, Hidden Dim]
            elif pooling == 'sum':
                embedds.append(_embedds.sum(axis = -2))
            elif pooling == 'mask':
                mask_pos_i, mask_pos_j = torch.where(input_ids == self.tokenizer.tokenizer.mask_token_id)
                if output_hidden_states:
                    embedds.append(_embedds[:, mask_pos_i, mask_pos_j])
                else:
                    embedds.append(_embedds[mask_pos_i, mask_pos_j])
            elif pooling is None:
                embedds.append(_embedds)
            else:
                raise Exception('Check pooling.')
        if pooling is None:
            # We need this complex code because all embeddings will have a different length
            
            # retreive max length size
            maxLength = max([e.shape[2] for e in embedds])
            for k in range(len(embedds)):
                embedds[k] = torch.nn.functional.pad(embedds[k], (0, 0, 0, maxLength - embedds[k].shape[2])) # The last arg means that you pad 0 to the left of the last dim
                                                                                                                                             # 0 to the right of the last dim
                                                                                                                                             # 0 to the left of the last dim - 1
                                                                                                                                             # maxLength - embedds[k].shape[2] to the right of the last dim -1                                                
        if output_hidden_states:
            embedds = torch.cat(embedds, dim = 1) # Shape [Num Layers, Dataset Size, Hidden Dim]
        else:
            embedds = torch.cat(embedds, dim = 0) # Shape [Dataset Size, Hidden Dim]
        
        print("Embeddings Shape: ", embedds.shape)
        
        return embedds
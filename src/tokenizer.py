
from typing import List, Tuple, Optional
from torch import Tensor, tensor, where, ones
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer



class TokenizerWrapper():
    """
    Wrapper class that deals with all the specificity of each
    tokenization/embedding process. It will be one ugly function 
    but at least everything is centralized here.
    
    Different methods illustrated with:
    
        (1) [CLS] is located in [SEP]
            
            Here [X] and [Y] are removed and not instantiated.
            The issue would be that therefore the sentence becomes ungrammatical
            for english of course (even more when [X] or [Y] are not on the sentence 
            boundary) but even (pehaps) for machine prompts.
            In this case the model only have to evaluate only one prompt. 
        
        (2) [CLS] Paris is located in [SEP]
        
            Here only [X] has been instantiated. The advantage is that every prompts 
            that end with [Y] would be grammatical up to where it has been cut. However
            in a MaskedLM the [SEP] token would provide ungrammatical signal. Moreover
            for prompts where [Y] is in the middle (eg. '[X] and [Y] are twin cities') it
            is again ungrammatical.
            In this case the model needs to evaluate each row from the dataset.
        
        (3) [CLS] Paris is located in [MASK] [SEP]
        
            Here [X] has been instantiated and [Y] has been replaced by [MASK]. It would
            need to be changed for other models than BERT. The advantage is that it is
            the cloze form needed to solve LAMA. Hence the [MASK] embedding is supposed
            to encode what is needed to predict [Y]. Moreover the sentence is now
            grammatical. It's only working with one-token [Y]. 
            In this case the model needs to evaluate each row from the dataset.
        
        (4) [CLS] Paris is located in France [SEP]
        
            Here [X] and [Y] are instantiated. 
            Perfect to evaluate embeddings. It raises a question regarding
            perplexity or NLL. 
            /!\ Need to look the difference between [Y] embedding here
                and [MASK] embedding, or perplexity /!\
            In this case the model needs to evaluate each row from the dataset.
            
            
    On top of that attention_mask could be different things:
        
        (i) For embedding analysis for example attention mask should be the whole
            sentence (and 0s would be the padding).
            
        (ii) For NLL computation 1s would design only the prompt's tokens.
    """
    
    def __init__(self,
                 tokenizer: PreTrainedTokenizer) -> None:
        
        # Get tokenizer of studied model
        
        self.tokenizer = tokenizer
        
        # Compute token2id dict usefull to compute non-human prompts tokenization
        
        self.token2id = {} # The idea is that when tokenizing ##s for example (which is a token)
                           # the tokenizer will treat it as # - # - s which is unfortunate...
        for id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode(id)
            self.token2id[token] = id
            
        
    def tokenize(self,
                 templates: List[str],
                 sub_surfaces: Optional[List[str]],
                 obj_surfaces: Optional[List[str]],
                 prompt_attention_mask: bool = False,
                 method: int = 4,
                 token_sequence: bool = False,
                 remove_dot: bool = False) -> Tuple[Tensor, Tensor]:
        """
            Main method of the class. Wrapper that calls different submethods depending on the request.
            It depends mainly on the method used and the type of the prompts.
            
            Args:
                templates (List[str]) List of prompt templates that are to be tokenized/
                                      Each template is of the form:
                                      tok_0 ... tok_i [X]/[Y] tok_{i+1} ... tok_j [Y]/[X] ... tok_n
                
                sub_surfaces (List[str]) List of instantiations of [X].
                                         It can be composed of multiple tokens.
                                         
                obj_surfaces (List[str]) List of instantiations of [Y].
                                         It can be composed of multiple tokens.
                                         
                prompt_attention_mask (bool) if set to True attention_mask will be as (ii)
                                             if set to False attention_mask will be as (i) 
                                             Default: False
                
                method (int) Is refering to the method number from the class description.
                             Default: 4
                
                token_sequence (bool) If True then the tokenizer will treat it as a sequence of token 
                                      separated by a space and will tokenize it using token2id dict.
                                      Otherwise it will use the model's tokenizer directly.
                                      Default: False
                
                remove_dot (bool) if True then we remove dots from prompts.
                                  It hurts LAMA scores when we remove it.
                                  Default: False
                
            Returns:
                input_ids (Tensor) shape [BS, L]
                                   token ids from self.tokenizer.
                
                attention_mask (Tensor) shape [BS, L]
                                        Here attention mask can be two things:
                                            (i) Classical attention_masks e.g. 1s on every tokens
                                                except for padding ones where it is 0s.
                                            (ii) Customized masks where 1s are only on the prompts
                                                 tokens and are 0s everywhere else.
            
        
        """
        if remove_dot:
            # Removing ' .' hurts a lot the model performances
            # For perplexity computation it doesn't matters as we reveal
            # context from left to right...
            templates = [template.replace(' .', '') for template in templates]
        
        if prompt_attention_mask:
            ### Masks ###
            if token_sequence:
                attention_mask = [ [self.token2id['[CLS]']] + [self.token2id[tok] for tok in template.replace('[X]', '[MASK]').replace('[Y]', '[UNK]').split(' ')] + [self.token2id['[SEP]']]\
                                    for template in templates]
            else:
                attention_mask = [self.tokenizer(template.replace('[X]', '[MASK]').replace('[Y]', '[UNK]')).input_ids for template in templates] # to id [X] and [Y] after tokenization
        else:
            attention_mask = None
            
        ### Preprocess ###
        if method == 1:
            sentences, attention_mask = self._preprocess_1(templates,
                                                           attention_mask,
                                                           prompt_attention_mask)
        elif method == 2:
            sentences, attention_mask = self._preprocess_2(templates,
                                                           attention_mask,
                                                           sub_surfaces,
                                                           token_sequence,
                                                           prompt_attention_mask)
        elif method == 3:
            sentences, attention_mask = self._preprocess_3(templates,
                                                           attention_mask,
                                                           sub_surfaces,
                                                           token_sequence,
                                                           prompt_attention_mask)
        elif method == 4:
            sentences, attention_mask = self._preprocess_4(templates,
                                                           attention_mask,
                                                           sub_surfaces,
                                                           obj_surfaces,
                                                           token_sequence,
                                                           prompt_attention_mask)
            
        ### Tokenize ###
        if token_sequence:
            input_ids = [
                            tensor(
                                [self.token2id['[CLS]']] + [self.token2id[token] for token in sentence.split(' ')] + [self.token2id['[SEP]']] \
                                ) 
                            for sentence in sentences        
                        ]
            if not(prompt_attention_mask):
                # Attention mask computation was ignored in the preprocessing
                attention_mask = [ ones(len(t)) for t in input_ids]
            
            input_ids = pad_sequence(input_ids, 
                                     batch_first = True)
        else:
            encodings = self.tokenizer(sentences, 
                                       padding = True, 
                                       return_tensors = 'pt')
            input_ids = encodings.input_ids
            if not(prompt_attention_mask):
                # Attention mask computation was ignored in the preprocessing
                attention_mask = encodings.attention_mask
            
        attention_mask = pad_sequence(attention_mask, 
                                      batch_first = True)
        
        # beacause as we return a mask of [0] for sentences where len(word2) > 1 the padding doesn't take them into account
        input_ids = input_ids[:, :attention_mask.shape[1]] 
        
        return input_ids, attention_mask
    
    
    ### PREPROCESSING ###
    
    
    def _preprocess_1(self,
                      templates: List[str],
                      attention_mask: List[int],
                      prompt_attention_mask: bool) -> Tuple[List[str], List[Tensor]]:
        sentences = [template.replace('[X] ', '').replace(' [Y]', '') for template in templates]
                                                                                                        
        if prompt_attention_mask:                                                                                                
            attention_mask = [self.mask_helper(mask) for mask in attention_mask] 
        
        return sentences, attention_mask
    
    def _preprocess_2(self,
                      templates: List[str],
                      attention_mask: List[int],
                      sub_surfaces: Optional[List[str]],
                      token_sequence: bool = False,
                      prompt_attention_mask: bool = False) -> Tuple[List[str], List[Tensor]]:
        
        if token_sequence:
            sentences = [
                        template.replace('[X]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(sub_surface).input_ids[1:-1])).replace(' [Y]', '') \
                        for template, sub_surface in zip(templates, sub_surfaces)
                        ]
        else:
            sentences = [template.replace('[X]', sub_surface).replace(' [Y]', '') for template, sub_surface in zip(templates, sub_surfaces)]
            
        if prompt_attention_mask:
            attention_mask = [self.mask_helper(mask, word1 = sub_surface) for mask, sub_surface in zip(attention_mask, sub_surfaces)] 
            
        return sentences, attention_mask
    
    def _preprocess_3(self,
                      templates: List[str],
                      attention_mask: List[int],
                      sub_surfaces: Optional[List[str]],
                      token_sequence: bool = False,
                      prompt_attention_mask: bool = False) -> Tuple[List[str], List[Tensor]]:
        
        if token_sequence:
            sentences = [
                        template.replace('[X]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(sub_surface).input_ids[1:-1])).replace('[Y]', self.tokenizer.mask_token) \
                        for template, sub_surface in zip(templates, sub_surfaces)
                        ]
        else:
            sentences = [template.replace('[X]', sub_surface).replace('[Y]', '[MASK]') for template, sub_surface in zip(templates, sub_surfaces)]
            
        if prompt_attention_mask:
            attention_mask = [self.mask_helper(mask, word1 = sub_surface, word2 = self.tokenizer.mask_token) for mask, sub_surface in zip(attention_mask, sub_surfaces)]
        
        return sentences, attention_mask
    
    def _preprocess_4(self,
                      templates: List[str],
                      attention_mask: List[int],
                      sub_surfaces: Optional[List[str]],
                      obj_surfaces: Optional[List[str]],
                      token_sequence: bool = False,
                      prompt_attention_mask: bool = False) -> Tuple[List[str], List[Tensor]]:
        
        if token_sequence:
            sentences = [
                        template.replace('[X]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(sub_surface).input_ids[1:-1])).replace('[Y]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(obj_surface).input_ids[1:-1])) \
                        for template, sub_surface, obj_surface in zip(templates, sub_surfaces, obj_surfaces)
                        ]
        else:
            sentences = [template.replace('[X]', sub_surface).replace('[Y]', obj_surface) for template, sub_surface, obj_surface in zip(templates, sub_surfaces, obj_surfaces)]
            
        if prompt_attention_mask:
            attention_mask = [self.mask_helper(mask, word1 = sub_surface, word2 = obj_surface) for mask, sub_surface, obj_surface in zip(attention_mask, sub_surfaces, obj_surfaces)]
        
        return sentences, attention_mask
    
    
    ### UTILS ###
    
    
    def mask_helper(self, 
                    mask: List[int], 
                    word1: str = None, 
                    word2: str = None) -> Tensor:
        """
            Basically the mask is 1 when the token 
            belongs to the prompt and 0 evrywhere else.
            
            Example: '[CLS] Paris is the capital of France [SEP]'
                     -> [0, 0, 1, 1, 1, 1, 0, 0]
                     
            Rq: if [Y] is tokenized in more than one token
                it is dropped and this method returns 0
            
            Args:
                mask (List[int]) This list contains the prompt's tokens ids except that
                                 [X] has been replaced by the [MASK] token id and [Y] by
                                 the [UNK] token id.
                
                word1 (str) is the sub_surface word (e.g. what replaces [X])
                
                word2 (str) is the obj_surface word (e.g. what replaces [Y])
                
            Returns:
                h (Tensor) shape [L]
                           It's 0s everywhere except on prompt's tokens where it's 1s.
                
        """
        h = []
        for e in mask:
            if e == self.tokenizer.mask_token_id:
                if word1 is not None:
                    h += [0]*len(self.tokenizer(word1).input_ids[1:-1]) 
            elif e == self.tokenizer.unk_token_id:
                if word2 is not None:
                    word2_ids = self.tokenizer(word2).input_ids[1:-1]
                    if len(word2_ids) > 1:
                        return tensor([0]) # if the label is tokenized in more than one, skip!
                    h += [1]
            elif e == self.tokenizer.cls_token_id:
                h.append(0)
            elif e == self.tokenizer.sep_token_id:
                h.append(0)
            else:
                h.append(1)
        return tensor(h)
    

    def get_tokens_list(self,
                        template: str,
                        token_sequence: bool) -> Tuple[List[str], int, int]:
        """
            Given a template returns the list of tokens as well as the position
            of the [X] and the [Y].
            
            This method is only usefull for plotting purposes. 
            
            Ex: '[X] is the capital of [Y]'
                -> (['[X]', 'is', 'the', 'capital', 'of', '[Y]'], 0, 5)
                
            Rq: If token_sequence is True then template is assumed to start with [X]
                and end with [Y].
            
            Args:
                template (str) 
                token_sequence (bool)
            Returns:
                tokens (List[str]) list of the tokens excluding [CLS] and [SEP]
                pos_x (int) index of [X] in tokens
                pos_y (int) index of [Y] in tokens
            
        """
        ### Need to deal with tokens properly ###
        # for lama prompts [X] and [Y] are not necesseraly in first and last position
        if token_sequence:
            template_tokenized = [self.token2id[token] for token in template.replace('[X] ', '').replace(' [Y]', '').replace(' .', '').split(' ')]
            tokens = [self.tokenizer.decode(tok) for tok in template_tokenized]
            pos_x = 0
            pos_y = len(tokens) + 1
            tokens = ['[X]'] + [self.tokenizer.decode(tok) for tok in template_tokenized] + ['[Y]']
        else:
            template_tokenized = self.tokenizer(template.replace('[X]', '[MASK]').replace('[Y]', '[UNK]').replace(' .', ''), return_tensors = 'pt').input_ids[0,1:-1]
            pos_x = where( template_tokenized == self.tokenizer.mask_token_id )[0].item()
            pos_y = where( template_tokenized == self.tokenizer.unk_token_id )[0].item()
            tokens = [self.tokenizer.decode(tok) for tok in template_tokenized]
            tokens[pos_x] = '[X]'
            tokens[pos_y] = '[Y]'
                
        return tokens, pos_x, pos_y

import numpy as np
import pandas as pd
import torch
import tqdm



class MetricsWrapper:
    
    def __init__(self, model, tokenizer, device):
        # need to rework the model type thing in every method
        self.model = model
        self.tokenizer = tokenizer
        self.device = device 
        
        # Useful
        self.token2id = {} # The idea is that when tokenizing ##s for example (which is a token)
                    # the tokenizr will treat it as # - # - s which is unfortunate...
        for id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode(id)
            self.token2id[token] = id
    
    def compute_nll_perplexity(self,
                               df: pd.core.frame.DataFrame = None, 
                               method: int = 2,
                               **kwargs):
        """
            Compute NLL and Perplexity of the dataset df.
        
            Method 1: [CLS] is located in [SEP]
            Method 2: Avg. on [CLS] Paris is located in [SEP]
            Method 3: Avg. on [CLS] Paris is located in [MASK] [SEP]
            Method 4: Avg. on [CLS] Paris is located in France [SEP]


            Note:
            We give the model the whole sentence but as [X] can be 
            multiple tokens we only keep the nll of the prompt!
        """

        # The loop
        full_nlls = []
        full_perplexities = []
        for k in tqdm.tqdm(range(0, len(df), kwargs['batch_size']), disable = True):
            
            # Retrieve elem
            elems = df.iloc[k:k+kwargs['batch_size']]
            sub_surfaces = elems['sub_surface'].tolist() # [X]
            obj_surfaces = elems['obj_surface'].tolist() # [Y]
            templates = elems['template'].tolist()
            rela = elems['predicate_id']

            # 
            input_ids, attention_mask = self.get_encodings(
                                                templates = templates,
                                                sub_surfaces = sub_surfaces,
                                                obj_surfaces = obj_surfaces,
                                                autoprompt = kwargs['autoprompt'],
                                                method = method
                                                )
            
            mask_left_to_right = torch.zeros_like(input_ids).to(self.device)
            
            input_ids = input_ids.to(self.device)
            
            # Forward passes and NLL
            nlls = []
            perplexities = []
            for k in range(input_ids.size(1)):
                mask_left_to_right[:, k] = 1.
                with torch.no_grad():
                    logits = self.model(
                                    input_ids = input_ids, 
                                    attention_mask = mask_left_to_right
                                    )
                probs = logits.softmax(-1)
                
                ids = input_ids[:,k] # shape (BS)
                ps = probs[torch.arange(input_ids.shape[0]), k, ids] # shape (BS)
                
                nlls.append(-torch.log(ps).cpu())
                
                perplexities.append(
                            self.compute_perplexity(probs[:,k]).cpu() # will change when we'll do the batch-wise computation
                            )
                
            nlls = torch.vstack(nlls).t() # (BS, L) (L = input_ids.shape[1])   
            perplexities = torch.vstack(perplexities).t()                                 

            if method > 1:
                # Here nlls size depend on the number of token of [X]
                # So we need to cut it out!
                
                full_nlls.append( nlls[(nlls * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1) ) # if method 3 or 4 shape of 
                                                                                                                           # nll is prompt_size + 1
                                                                                                                           # because of [MASK] or obj
                                                                                                                           
                                                                                                                           # (Not-zero BS, prompt length)
                full_perplexities.append(  perplexities[(perplexities * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1) )
            else:
                
                full_nlls.append( nlls[(nlls * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1)[0] ) # Method 1 each prompt is the same, only need the first one of the batch
                
                full_perplexities.append( perplexities[(perplexities * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1)[0] )
                break
            
        if method > 1:
            full_nlls = torch.cat(full_nlls, axis = 0)
            full_perplexities = torch.cat(full_perplexities, axis = 0)
        else:
            full_nlls = torch.stack(full_nlls)
            full_perplexities = torch.stack(full_perplexities)       

        ### Need to deal with tokens properly ###
        # for lama prompts [X] and [Y] are not necesseraly in first and last position
        if kwargs['autoprompt']:
            template_tokenized = [self.token2id[token] for token in templates[0].replace('[X] ', '').replace(' [Y]', '').replace(' .', '').split(' ')]
            tokens = [self.tokenizer.decode(tok) for tok in template_tokenized]
            pos_x = 0
            pos_y = len(tokens) + 1
            if method != 3:
                tokens = ['[X]'] + [self.tokenizer.decode(tok) for tok in template_tokenized] + ['[Y]']
            else:
                tokens = ['[X]'] + [self.tokenizer.decode(tok) for tok in template_tokenized] + ['[MASK]']
        else:
            template_tokenized = self.tokenizer(templates[0].replace('[X]', '[MASK]').replace('[Y]', '[UNK]').replace(' .', ''), return_tensors = 'pt').input_ids[0,1:-1]
            pos_x = torch.where( template_tokenized == self.tokenizer.mask_token_id )[0].item()
            pos_y = torch.where( template_tokenized == self.tokenizer.unk_token_id )[0].item()
            tokens = [self.tokenizer.decode(tok) for tok in template_tokenized]
            tokens[pos_x] = '[X]'
            if method != 3:
                tokens[pos_y] = '[Y]'
            else:
                tokens[pos_y] = '[MASK]'

        return {'nll': full_nlls, 
                'perplexity': full_perplexities, 
                'tokens': {'pos_x': pos_x,
                           'pos_y': pos_y,
                           'tokens': tokens}}
    
    
    def evaluate_on_lama(self, 
                         dataset, 
                         **kwargs): # need to rework the model type thing in every method

        print("Evaulate LAMA score...")
        
        scores = {'P@1': 0,
                  'P@5': 0,
                  'P@20': 0,
                  'P@100': 0}

        num_eval = kwargs['num_eval'] if kwargs['num_eval'] != -1 else len(dataset)

        total_eval = 0
        for k in tqdm.tqdm(range(0, num_eval, kwargs['batch_size'])): # We'll do it by hand
            # Retrieve elem
            elems = dataset.iloc[k:k + kwargs['batch_size']]
            sub_surfaces = elems['sub_surface'] # [X]
            obj_surfaces = elems['obj_surface'] # [Y]
            templates = elems['template']
            relas = elems['predicate_id']

            # Create and tokenize template
 
            if not(kwargs['autoprompt']):
                sentences = [
                        template.replace('[X]', sub_surface).replace('[Y]', self.tokenizer.mask_token) \
                        for template, sub_surface in zip(templates, sub_surfaces)
                        ]
                encodings = self.tokenizer(sentences, padding = True, return_tensors = 'pt')
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask
            else:
                # This is tricky because we need to tokenize the sub_surfaces first
                sentences = [
                        template.replace('[X]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(sub_surface).input_ids[1:-1])).replace('[Y]', self.tokenizer.mask_token) \
                        for template, sub_surface in zip(templates, sub_surfaces)
                        ]

                input_ids = [
                                torch.tensor(
                                    [self.token2id['[CLS]']] + [self.token2id[token] for token in sentence.split(' ')] + [self.token2id['[SEP]']] \
                                 ) 
                                for sentence in sentences
                                
                            ]
                attention_mask = [ torch.ones(len(t)) for t in input_ids]
                
                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first = True)
                attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first = True)
                
            mask_pos_i, mask_pos_j = torch.where(input_ids == self.tokenizer.mask_token_id)

            # forward pass
            with torch.no_grad():
                logits = self.model(
                                    input_ids = input_ids.to(self.device),
                                    attention_mask = attention_mask.to(self.device)
                                    ) # the .logits is in the wrapper class

            _, ids = torch.topk(logits[mask_pos_i,mask_pos_j], k = 100)
            ids = ids.cpu()
            
            # From here: black magic.
            # (1) tokenize labels
            # (2) use attention_mask to see which labels have been tokenized in more than one token
            # (3) get only 1:2 from axis 1 of labels_ids as the rows that contains more than one token will be filtered out
            # (4) if one get True then True ( any(axis = 1) )
            # (5) filter using filter multiplication
            # sum
            labels = self.tokenizer(obj_surfaces.tolist(), padding = True, return_tensors = 'pt')
            labels_ids = labels.input_ids
            filter = torch.logical_not((labels.attention_mask.sum(axis = 1) > 3)) # True if tokenized in 1, (3 because [cls] and [sep])

            scores['P@1'] += ((labels_ids[:,1:2] == ids[:,:1]).any(axis = 1) * filter).sum().item()
            scores['P@5'] += ((labels_ids[:,1:2] == ids[:,:5]).any(axis = 1) * filter).sum().item()
            scores['P@20'] += ((labels_ids[:,1:2] == ids[:,:20]).any(axis = 1) * filter).sum().item()
            scores['P@100'] += ((labels_ids[:,1:2] == ids[:,:100]).any(axis = 1) * filter).sum().item()
            
            #scores['P@100'] += float(label_id in ids[0, :100])

            total_eval += filter.sum().item() # count only the relevanrt rows

        print(f"Total Number of Evaluations: {total_eval} (dropped {num_eval - total_eval})")
        
        return {k: v/total_eval for k,v in scores.items()}
            
    def compute_perplexity(self, probs: torch.Tensor) -> torch.Tensor:
        """
            Here perplexity means according to us.
            
            Args:
                probs (tensor) shape (BS, vocab_size)
            
        """
        H = - (probs * torch.log(probs)).sum(axis = 1) # entropy base e
        return torch.exp(H) # shape BS
    
    def get_encodings(self, 
                      templates: list,
                      sub_surfaces: list,
                      obj_surfaces: list,
                      autoprompt: bool,
                      method: int):
        
        ##### DEAL with ' .' ?
        templates = [template.replace(' .', '') for template in templates]
        
        # Masks
        if autoprompt:
            attention_mask = [ [self.token2id['[CLS]']] + [self.token2id[tok] for tok in template.replace('[X]', '[MASK]').replace('[Y]', '[UNK]').split(' ')] + [self.token2id['[SEP]']]\
                              for template in templates]
        else:
            attention_mask = [self.tokenizer(template.replace('[X]', '[MASK]').replace('[Y]', '[UNK]')).input_ids for template in templates] # to id [X] and [Y] after tokenization
            
        # Create and tokenize template according to method
        if method == 1:
            sentences = [template.replace('[X] ', '').replace(' [Y]', '') for template in templates] # Method 1 is not to be used here as we do a pass on each
                                                                                                     # line of the relation, and with method 1 each sentence 
                                                                                                     # is the same!
            attention_mask = [self.mask_helper(mask) for mask in attention_mask]  
        elif method == 2:
            
            if autoprompt:
                sentences = [
                            template.replace('[X]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(sub_surface).input_ids[1:-1])).replace(' [Y]', '') \
                            for template, sub_surface in zip(templates, sub_surfaces)
                            ]
            else:
                sentences = [template.replace('[X]', sub_surface).replace(' [Y]', '') for template, sub_surface in zip(templates, sub_surfaces)]
                
            attention_mask = [self.mask_helper(mask, word1 = sub_surface) for mask, sub_surface in zip(attention_mask, sub_surfaces)] 

        elif method == 3:
            
            if autoprompt:
                sentences = [
                            template.replace('[X]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(sub_surface).input_ids[1:-1])).replace('[Y]', self.tokenizer.mask_token) \
                            for template, sub_surface in zip(templates, sub_surfaces)
                            ]
            else:
                sentences = [template.replace('[X]', sub_surface).replace('[Y]', '[MASK]') for template, sub_surface in zip(templates, sub_surfaces)]
                
            attention_mask = [self.mask_helper(mask, word1 = sub_surface, word2 = self.tokenizer.mask_token) for mask, sub_surface in zip(attention_mask, sub_surfaces)]

        elif method == 4:
            
            if autoprompt:
                sentences = [
                            template.replace('[X]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(sub_surface).input_ids[1:-1])).replace('[Y]', ' '.join(self.tokenizer.decode(id) for id in self.tokenizer(obj_surface).input_ids[1:-1])) \
                            for template, sub_surface, obj_surface in zip(templates, sub_surfaces, obj_surfaces)
                            ]
            else:
                sentences = [template.replace('[X]', sub_surface).replace('[Y]', obj_surface) for template, sub_surface, obj_surface in zip(templates, sub_surfaces, obj_surfaces)]
                
            attention_mask = [self.mask_helper(mask, word1 = sub_surface, word2 = obj_surface) for mask, sub_surface, obj_surface in zip(attention_mask, sub_surfaces, obj_surfaces)]
            
        if autoprompt:
            #print(sentences)
            input_ids = [
                            torch.tensor(
                                [self.token2id['[CLS]']] + [self.token2id[token] for token in sentence.split(' ')] + [self.token2id['[SEP]']] \
                                ) 
                            for sentence in sentences        
                        ]
            #attention_mask = [ torch.ones(len(t)) for t in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first = True)
        else:
            encodings = self.tokenizer(sentences, padding = True, return_tensors = 'pt')
            input_ids = encodings.input_ids
          
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first = True)
        
        input_ids = input_ids[:, :attention_mask.shape[1]] # beacause as we return a mask of [0] for sentences where len(word2) > 1 the padding doesn't take them into account
        
        return input_ids, attention_mask
    
    def mask_helper(self, 
                    mask: list, 
                    word1: str = None, 
                    word2: str = None):
        """
            Basically the mask is 1 when the token 
            belongs to the prompt and zero evrywhere else. 
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
                        return torch.tensor([0]) # if the label is tokenized in more than one, skip!
                    h += [1]
            elif e == self.tokenizer.sep_token_id:
                h.append(0)
            else:
                h.append(1)
        h[0] = 0
        return torch.tensor(h)
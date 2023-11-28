

import pandas as pd


def get_template_from_lama(df: pd.core.frame.DataFrame) -> dict:
        """
            Retrieve templats from a LAMA type dataframe.
            
            Rk: remove [X] and [Y]. Could be interesting to replace it by instantaiations.
        """
        templates = {}
        for k in range(len(df)):
            elem = df.iloc[k]
            rela = elem['predicate_id']
            if rela in templates.keys():
                continue
            templates[rela] = elem['template'].replace('[X] ', '').replace(' [Y]', '').replace(' .', '')
        return templates
    
class Embedder():
    """
    Wrapper class that deals with all the specificity of each
    embeddings. It will be one ugly function but at least everything
    is centralized here.
    
    
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
                 tokenizer):
        self.tokenizer = tokenizer
        self.token2id = {} # The idea is that when tokenizing ##s for example (which is a token)
                           # the tokenizer will treat it as # - # - s which is unfortunate...
        for id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode(id)
            self.token2id[token] = id
        
    def embed(self,):
        ##### DEAL with ' .' ? 
        # Removing ' .' hurts a lot the model performances
        # Might need a flag to remove or not the ' .'
        # For perplexity computation it doesn't matters as we reveal
        # context from left to right...
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
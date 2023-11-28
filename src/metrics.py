
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics.cluster import completeness_score, homogeneity_score
from sklearn.cluster import SpectralClustering, KMeans
import torch
import tqdm

from src.utils import get_template_from_lama


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

        return {'nll': full_nlls, # [Dataset Length, L]
                'perplexity': full_perplexities, 
                'tokens': {'pos_x': pos_x,
                           'pos_y': pos_y,
                           'tokens': tokens}}
        
    def compute_embeddings(self,
                           df: pd.core.frame.DataFrame = None,
                           prompt_list: list = None,
                           pooling: str = 'avg',
                           **kwargs):
        """
            Compute embeddings of prompts from df.
            
            rk: for now always remove instantiation of [X] and [Y].
        
        """
        
        if prompt_list is not None:
            sentences = prompt_list
        else:
            templates = get_template_from_lama(df = df)
            sentences = list(templates.values())
        
        if kwargs['autoprompt']:
            # Tokenize
            input_ids = [
                        torch.tensor(
                            [self.token2id['[CLS]']] + [self.token2id[token] for token in sentence.split(' ')] + [self.token2id['[SEP]']] \
                            )
                        for sentence in sentences
                        ]
            input_ids = torch.vstack(input_ids)
            attention_mask = torch.ones_like(input_ids)
            inputs = {'input_ids': input_ids,
                    'attention_mask': attention_mask}
        else:
            inputs = self.tokenizer(sentences,
                            return_tensors = 'pt',
                            padding = True)

        print("Inputs shape: ", inputs['input_ids'].shape)
        with torch.no_grad():
            outputs = self.model.embedding(
                                input_ids = inputs['input_ids'].to(self.device),
                                attention_mask = inputs['attention_mask'].to(self.device)
                                )
        _embeds = outputs.last_hidden_state.cpu() * inputs['attention_mask'][:,:,None]
        # Keeping [CLS] & [SEP]

        if pooling == 'avg':
            embeds = _embeds.mean(axis = 1)
        elif pooling == 'sum':
            embeds = _embeds.sum(axis = 1)
        else:
            raise Exception('Check pooling.')
        
        print("Embeddings shape ", embeds.shape)
        # rela to embedding dict can be useful
        
        if prompt_list is not None:
            rela2embed = None
        else:
            rela2embed = {k: v for k, v in zip(templates.keys(), embeds)}
        
        return embeds, rela2embed
        
            
    
    def evaluate_on_lama(self, 
                         dataset, 
                         **kwargs): # need to rework the model type thing in every method

        print("Evaulate LAMA score...")
        
        scores = {'P@1': 0,
                  'P@5': 0,
                  'P@20': 0,
                  'P@100': 0}

        relas_and_scores = {'rela': [],
                            'filter': [],
                            'P@1': [],
                            'P@5': [],
                            'P@20': [],
                            'P@100': []}

        num_eval = kwargs['num_eval'] if kwargs['num_eval'] != -1 else len(dataset)

        total_eval = 0
        for k in tqdm.tqdm(range(0, num_eval, kwargs['batch_size'])): # We'll do it by hand
            # Retrieve elem
            elems = dataset.iloc[k:k + kwargs['batch_size']]
            sub_surfaces = elems['sub_surface'] # [X]
            obj_surfaces = elems['obj_surface'] # [Y]
            templates = elems['template']
            relas = elems['predicate_id']
            
            # /!\ Removing the dot leads to a huge drop in perf /!\
            templates = [
                            self.shuffle_template(
                                            template = template,
                                            shuffle = kwargs['shuffle']
                            ) \
                            for template in templates
                        ]

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
            
            relas_and_scores['rela'] += relas.to_list()
            relas_and_scores['filter'] += (1.*filter).tolist()
            relas_and_scores['P@1'] += (1.* (labels_ids[:,1:2] == ids[:,:1]).any(axis = 1) * filter).tolist()
            relas_and_scores['P@5'] += (1.* (labels_ids[:,1:2] == ids[:,:5]).any(axis = 1) * filter).tolist()
            relas_and_scores['P@20'] += (1.* (labels_ids[:,1:2] == ids[:,:20]).any(axis = 1) * filter).tolist()
            relas_and_scores['P@100'] += (1.* (labels_ids[:,1:2] == ids[:,:100]).any(axis = 1) * filter).tolist()
            
        relas_and_scores = pd.DataFrame(data = relas_and_scores) 

        print(f"Total Number of Evaluations: {total_eval} (dropped {num_eval - total_eval})")
        
        return {k: v/total_eval for k,v in scores.items()}, self.compute_lama_scores_by_rela(relas_and_scores)
    
    
    def compute_embeddings_analysis(self, 
                                    embeds1: torch.Tensor, 
                                    embeds2: torch.Tensor,
                                    embeds3: torch.Tensor = None) -> dict:
        """
            Compute several metrics on embeds1 and embeds2 and return
            them in a dict.
            
            Metrics include:
                - cosine similarity
                - PCA then linear correlation & cosine similarity
                - distances matrix (via euclidian & cosine similarity)
                - MDS based on those matrices
                - t-SNE then V-measure, linear correlation, cosine similarity
                
            Args:
                embeds1 (torch.Tensor) shape [N prompts, model embedding size]
                embeds2 (torch.Tensor) shape [N prompts, model embedding size]
        
        """
        assert embeds1.shape[0] <= embeds2.shape[0]
        
        results = {}
        
        ### Cosine Similarity ###
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        sims = cos(embeds1, embeds2[:embeds1.shape[0]])
        
        results['cosim raw'] = sims
        results['cosim avg'] = sims.mean().item()
        results['cosim max'] = sims.max().item()
        
        print(f"\tCosine Sim - avg: {np.round(results['cosim avg'],3)}; max: {np.round(results['cosim max'],3)}")
        
        
        ### PCA ###
        
        if embeds3 is not None:
            pass
        else:
            full_embeds = np.concatenate(
                                    (embeds1.numpy(), 
                                    embeds2.numpy()), 
                                    axis = 0
                                    )
        
        pca = PCA(n_components=2)
        embeds_pca = pca.fit_transform(full_embeds) # Shape [2*(N prompts), 2]
        
        reg = LinearRegression().fit(embeds_pca[:embeds1.shape[0]], embeds_pca[embeds1.shape[0]:2*embeds1.shape[0]])
        # R²
        reg_score = reg.score(embeds_pca[:embeds1.shape[0]], embeds_pca[embeds1.shape[0]:2*embeds1.shape[0]])
        # Pearson 
        pearson = pearsonr(
                        embeds_pca[:embeds1.shape[0]].flatten(), 
                        embeds_pca[embeds1.shape[0]:2*embeds1.shape[0]].flatten()
                        )
        # Clustering
        true_labels = np.array([0]*embeds1.shape[0] + [1]*embeds2.shape[0])
        
        kmeans = KMeans(n_clusters = 2).fit(embeds_pca)
        kmeans_completeness = completeness_score(true_labels, kmeans.labels_)
        kmeans_homogeneity = homogeneity_score(true_labels, kmeans.labels_)
        
        spectral = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(embeds_pca)
        spectral_completeness = completeness_score(true_labels, kmeans.labels_)
        spectral_homogeneity = homogeneity_score(true_labels, kmeans.labels_)
        
        # Store Results
        results['pca R2'] = reg_score
        results['pca 1-corr'] = (pearson.statistic, pearson.pvalue)
        results['pca kmeans completeness'] = kmeans_completeness
        results['pca kmeans homogeneity'] = kmeans_homogeneity
        results['pca spectral completeness'] = spectral_completeness
        results['pca spectral homogeneity'] = spectral_homogeneity
        results['pca embeds1'] = embeds_pca[:embeds1.shape[0]]
        results['pca embeds2'] = embeds_pca[embeds1.shape[0]:]
        
        # Print
        print(f"\tPCA - R²: {np.round(results['pca R2'],3)}; 1-corr: {np.round(results['pca 1-corr'][0])} (p = {np.round(results['pca 1-corr'][1])})")
        print(f"\tPCA Clustering\
                \n\t\tKMeans - completeness: {np.round(results['pca kmeans completeness'],3)}; homogeneity {np.round(results['pca kmeans homogeneity'],3)}\
                \n\t\tSpectral - completeness: {np.round(results['pca spectral completeness'],3)}; homogeneity {np.round(results['pca spectral homogeneity'],3)}")
        
        ### Representational Similarities Analysis ###
        # TBD
        
        sim_mat = []
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(embeds1.shape[0]):
            row = []
            for j in range(embeds1.shape[0]):
                sim = cos(embeds1[i], embeds1[j])
                row.append(sim)
            sim_mat.append(row)
        sim_mat = np.array(sim_mat)
            
        #print(sim_mat.shape)
        #print(sim_mat)
        #exit(0)
        
        return results
    
            
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
                      sub_surfaces: list = None,
                      obj_surfaces: list = None,
                      autoprompt: bool = False,
                      method: int = 1):
        
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
    
    def compute_lama_scores_by_rela(self,
                                    raw_relas_and_scores: dict) -> dict:
        """
        
        """
        
        relas = list(set(raw_relas_and_scores['rela']))
        
        scores = {}
        for rela in relas:
            _scores = {}
            df = raw_relas_and_scores[(raw_relas_and_scores['rela'] == rela) & (raw_relas_and_scores['filter'] == 1) ]
            _scores['P@1'] = df['P@1'].sum()/len(df)
            _scores['P@5'] = df['P@5'].sum()/len(df)
            _scores['P@20'] = df['P@20'].sum()/len(df)
            _scores['P@100'] = df['P@100'].sum()/len(df)
            scores[rela] = _scores
        
        return scores
    
    def shuffle_template(self, 
                         template: str, 
                         shuffle: bool = False,
                         method: int = 2) -> str:
        """
            Args:
                template (str) str of the form "tok ... tok [X] tok tok ... tok [Y] tok ... tok".
                               We could try different method:
                                (i) shuffle everything
                                (ii) keep [X] and [Y] in place
                                /!\ [Y] is not always last...
                shuffle (bool) whether or not we apply a shuffling operation.         
        """
        if shuffle:
            words = template.split(' ')[:-1]
            if method == 1:
                # Let's only focus on (i) for now
                np.random.shuffle(words)
                return ' '.join(words) + ' .'
            elif method == 2:
                x_pos = words.index('[X]')
                y_pos = words.index('[Y]')
                words.remove('[X]')
                words.remove('[Y]')
                np.random.shuffle(words)
                words.insert(x_pos, '[X]')
                words.insert(y_pos, '[Y]')
                return ' '.join(words) + ' .'
        else:
            return template
        
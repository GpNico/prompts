
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn.metrics.cluster import completeness_score, homogeneity_score
from sklearn.cluster import SpectralClustering, KMeans
import torch
import tqdm
from transformers import PreTrainedTokenizer
from typing import Tuple

from src.embedder import Embedder
import src.utils as utils
from src.models import ModelWrapper
from src.tokenizer import TokenizerWrapper

class MetricsWrapper:
    
    def __init__(self, model: ModelWrapper, 
                       tokenizer: PreTrainedTokenizer, 
                       device: str) -> None:
        # need to rework the model type thing in every method
        self.model = model
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.device = device 
    
    ### MAIN METHODS ###
    
    def compute_nlls_perplexities(self,
                                  datasets: dict, 
                                  **kwargs) -> dict:
        """
            Compute NLL and Perplexity for all datasets.

            Note:
            We give the model the whole sentence but as [X] can be 
            multiple tokens we only keep the nll of the prompt!
            This behavior might change.
            
            Returns:
                res_dict (dict) keys: 'nlls'         values: (dict) keys: (str) relas from LAMA  values: (dict) keys: (str) dataset name  values: (Tensor) 
                                      'perplexities'         (dict) keys: (str) relas from LAMA  values: (dict) keys: (str) dataset name  values: (Tensor)
                                      'tokens'                  
        """
        
        print("Computing NLLs & Perplexities...")
        
        nlls = {}
        perplexities = {}
        tokens = {}
        
        # Get relation_ids
        relations_ids = list(set(datasets[list(datasets.keys())[0]]['predicate_id']))
        
        # Get LAMA name
        for name in datasets.keys():
            if 'lama' in name:
                lama_name = name
        
        # Start looping over relations
        for rela in tqdm.tqdm(relations_ids):
            
            rela_nlls = {}
            rela_perplexities = {}
            rela_tokens = {}
            
            # LAMA
            try:
                lama_res = self._compute_nlls_perplexities_one_rela(
                                        df = datasets[lama_name][datasets[lama_name]['predicate_id'] == rela], 
                                        token_sequence = False, 
                                        method = kwargs['method'],
                                        batch_size = kwargs['batch_size']
                                        )
                rela_nlls[lama_name] = lama_res['nll']
                rela_perplexities[lama_name] = lama_res['perplexity']
                rela_tokens[lama_name] = lama_res['tokens']
            except:
                pass
            
            # AutoPrompt
            for seed in kwargs['seeds']:
                try:
                    autoprompt_res = self._compute_nlls_perplexities_one_rela(
                                                df = datasets[f'autoprompt_seed{seed}'][datasets[f'autoprompt_seed{seed}']['predicate_id'] == rela], 
                                                token_sequence = True, 
                                                method = kwargs['method'],
                                                batch_size = kwargs['batch_size']
                                                )
                    rela_nlls[f'autoprompt_seed{seed}'] = autoprompt_res['nll']
                    rela_perplexities[f'autoprompt_seed{seed}'] = autoprompt_res['perplexity']
                    rela_tokens[f'autoprompt_seed{seed}'] = autoprompt_res['tokens']
                except:
                    pass

            # Random
            try:
                random_res = self._compute_nlls_perplexities_one_rela(
                                            df = datasets['random'][datasets['random']['predicate_id'] == rela], 
                                            token_sequence = True, 
                                            method = kwargs['method'],
                                            batch_size = kwargs['batch_size']
                                            )
                rela_nlls['random'] = random_res['nll']
                rela_perplexities['random'] = random_res['perplexity']
                rela_tokens['random'] = random_res['tokens']
            except:
                pass
            
            # store
            nlls[rela] = rela_nlls
            perplexities[rela] = rela_perplexities
            tokens[rela] = rela_tokens
            
        return {'nlls': nlls,
                'perplexities': perplexities,
                'tokens': tokens}
        
    def compute_lama_scores(self, 
                            datasets: dict, 
                            **kwargs) -> dict:
        """
            Compute precision at rank k (P@k) of human, machine and random prompts
            on the T-Rex subpart of the LAMA dataset.

            Note:
            Here LAMA is to be taken in a large sense as it can (and will) be mLAMA
            and/or ParaReL.
            
            Returns:
                res_dict (dict) key: (str) 'lama', 'autoprompt', 'random'    value: (dict) key: (str) 'scores', 'scores_by_rela'  value: (dict)                 
        """
    
        print("Computing LAMA score...")
                
        # LAMA
        lama_scores_lama, scores_by_rela_lama = self._compute_lama_scores(
                                    df = datasets[kwargs['lama_name']],
                                    num_eval = -1,
                                    token_sequence = False,
                                    batch_size = kwargs['batch_size'],
                                    )
        
        # AutoPrompt
        lama_scores_autoprompt = {}
        scores_by_rela_autoprompt = {}
        for seed in kwargs['seeds']:
            lama_scores_autoprompt[seed], scores_by_rela_autoprompt[seed] = self._compute_lama_scores(
                                                                df = datasets[f'autoprompt_seed{seed}'],
                                                                num_eval = -1,
                                                                token_sequence = True,
                                                                batch_size = kwargs['batch_size'],
                                                                )
                
        # Random
        lama_scores_random, scores_by_rela_random = self._compute_lama_scores(
                                        df = datasets['random'],
                                        num_eval = -1,
                                        token_sequence = True,
                                        batch_size = kwargs['batch_size'],
                                        )
        
        
        return {kwargs['lama_name']: {'scores': lama_scores_lama,
                                      'scores_by_rela': scores_by_rela_lama},
                'autoprompt': {'scores': lama_scores_autoprompt,
                               'scores_by_rela': scores_by_rela_autoprompt},
                'random': {'scores': lama_scores_random,
                           'scores_by_rela': scores_by_rela_random},}
        
        
    def compute_embeddings_analysis(self,
                                    datasets: dict,
                                    **kwargs) -> dict:
        """
        """
        
        print("Compute Embeddings Anlysis...")
        
        data_names = [kwargs["lama_name"]] + \
                     [f'autoprompt_seed{seed}' for seed in kwargs['seeds']] + \
                     ['random'] # Not that clean because we should use the data_to_load param but you know..
        
        # First we need to compute said embeddings
        
        embedder = Embedder(
                    model=self.model,
                    tokenizer=self.tokenizer
                    )
        
        if kwargs['pooling'] == 'mask':
            assert kwargs['method'] == 3
        
        embedds = {}
        
        embedds[kwargs['lama_name']] = embedder.embed(
                                    df = datasets[kwargs['lama_name']],
                                    method = kwargs['method'],
                                    pooling=kwargs['pooling'],
                                    output_hidden_states=kwargs['hidden_states'],
                                    batch_size = kwargs['batch_size'],
                                    token_sequence = False
                                    )
        
        for seed in kwargs['seeds']:
            embedds[f'autoprompt_seed{seed}'] = embedder.embed(
                                    df = datasets[f'autoprompt_seed{seed}'],
                                    method = kwargs['method'],
                                    pooling=kwargs['pooling'],
                                    output_hidden_states=kwargs['hidden_states'],
                                    batch_size = kwargs['batch_size'],
                                    token_sequence = True
                                    )
        
            
        embedds['random'] = embedder.embed(
                                    df = datasets['random'],
                                    method = kwargs['method'],
                                    pooling=kwargs['pooling'],
                                    output_hidden_states=kwargs['hidden_states'],
                                    batch_size = kwargs['batch_size'],
                                    token_sequence = True
                                    )
        
        # Select a subset of embeddings as treating it entirely is too much.
        
        if kwargs['hidden_states']:
            subset_dim = 1
        else:
            subset_dim = 0
            
        for name in data_names:
            embedds[name] = utils.select_subset(
                                    t = embedds[name],
                                    size = kwargs['subset_size'],
                                    dim = subset_dim
                                    )
            
        # Then Compute the Analysis
        
        res_dict = {}
        if kwargs['hidden_states']:
            for i in range(len(data_names)):
                for j in range(i+1, len(data_names)):
                    print(f"{data_names[i]} vs {data_names[j]}")
                    _res_dict_layers = {}
                    _res_dict_layers['label1'] = data_names[i]
                    _res_dict_layers['label2'] = data_names[j]
                    _res_dict_layers['num_layers'] = embedds[data_names[i]].shape[0]
                    for l in range(embedds[data_names[i]].shape[0]): # Num Layers
                        print(f"\t### Layer {l} ###")
                        _res_dict = self._compute_embeddings_analysis(embedds[data_names[i]][l], embedds[data_names[j]][l])
                        _res_dict = {k + f' layer{l}':v for k,v in _res_dict.items()}
                        _res_dict_layers.update(_res_dict)
                    res_dict[f'{data_names[i]}-{data_names[j]}'] = _res_dict_layers
        else:                        
            for i in range(len(data_names)):
                for j in range(i+1, len(data_names)):
                    print(f"{data_names[i]} vs {data_names[j]}")
                    _res_dict = self._compute_embeddings_analysis(embedds[data_names[i]], embedds[data_names[j]])
                    _res_dict['label1'] = data_names[i]
                    _res_dict['label2'] = data_names[j]
                    res_dict[f'{data_names[i]}-{data_names[j]}'] = _res_dict
        
        return res_dict
    
    
    def compute_curvatures(self,
                           datasets: dict,
                           **kwargs) -> dict:
        """
        
        Compute curvatures of embeddings for each layers of the model.
        We use this paper as ref: 
        Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language
        
        Returns:
            res_dict (dict) keys: dataset names values: (tensor) shape [Dataset Size, Num Layers]
        
        """
        
        print("Compute Curvatures...")
        
        # First we need to compute said embeddings
        
        print("\tComputing Embeddings...")
        
        embedder = Embedder(
                    model=self.model,
                    tokenizer=self.tokenizer
                    )
        
        embedds = {}
        
        embedds[kwargs['lama_name']] = embedder.embed(
                                    df = datasets[kwargs['lama_name']],
                                    method = 4, # We need [X] and [Y] 
                                    pooling = None, # We need the full embeddings
                                    output_hidden_states=True, # We want obviously
                                    batch_size = kwargs['batch_size'],
                                    token_sequence = False
                                    ) # Shape [Dataset Size, Num Layers, L, Dim]
        
        for seed in kwargs['seeds']:
            embedds[f'autoprompt_seed{seed}'] = embedder.embed(
                                    df = datasets[f'autoprompt_seed{seed}'],
                                    method = 4,
                                    pooling= None,
                                    output_hidden_states = True,
                                    batch_size = kwargs['batch_size'],
                                    token_sequence = True
                                    )
        
            
        embedds['random'] = embedder.embed(
                                    df = datasets['random'],
                                    method = 4,
                                    pooling=None,
                                    output_hidden_states=True,
                                    batch_size = kwargs['batch_size'],
                                    token_sequence = True
                                    )
        
        # Then compute curvatures
        
        print("\tComputing Curvatures...")
        res_dict = {}
        
        res_dict[kwargs['lama_name']] = self._compute_curvatures_from_embeddings(embedds[kwargs['lama_name']])
        
        for seed in kwargs['seeds']:
            res_dict[f'autoprompt_seed{seed}'] = self._compute_curvatures_from_embeddings(embedds[f'autoprompt_seed{seed}'])
        
        res_dict['random'] = self._compute_curvatures_from_embeddings(embedds['random'])
        
        return res_dict
    
    
    ### WHERE THE COMPUTATIONS ARE HIDDEN ####
    
    def _compute_nlls_perplexities_one_rela(self,
                                            df: pd.DataFrame, 
                                            method: int = 2,
                                            **kwargs) -> dict:
        """
            Compute NLL and Perplexity of the dataset df.
            
            We compute it left to right.
        
            Here df contains elements for only one relation.
            The loop over relations is taken care of in the 
            main.py currently.
            
            /!\ TBD: remove loop over relations from main.py
                     and put it here

            Note:
            We give the model the whole sentence but as [X] can be 
            multiple tokens we only keep the nll of the prompt!
            
            Returns:
                res_dict (dict) keys: 'nll'        values: (Tensor) shape [len(df), L]
                                      'perplexity'         (Tensor) shape [len(df), L]
                                      'tokens'             (dict) keys: 'tokens'  values: (List[str])
                                                                        'pos_x'           (int)
                                                                        'pos_y'           (int)
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

            # Tokenize
            input_ids, attention_mask = self.tokenizer.tokenize(
                                                templates = templates,
                                                sub_surfaces = sub_surfaces,
                                                obj_surfaces = obj_surfaces,
                                                token_sequence = kwargs['token_sequence'],
                                                method = method,
                                                prompt_attention_mask = True, # attention mask only focus on prompt's tokens
                                                remove_dot=True
                                                )
            
            mask_left_to_right = torch.zeros_like(input_ids).to(self.device)
            input_ids = input_ids.to(self.device)
            
            # Forward passes
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
                            utils.compute_perplexity(probs[:,k]).cpu() # will change when we'll do the batch-wise computation
                            )
                
            nlls = torch.vstack(nlls).t() # (BS, L) (L = input_ids.shape[1])   
            perplexities = torch.vstack(perplexities).t()                                 

            # Retrieve only prompt's tokens NLLs & Perplexities
            if method > 1:
                # Here nlls size depend on the number of token of [X] so we need to cut it out!
                full_nlls.append( nlls[(nlls * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1) ) # if method 3 or 4 shape of nll is prompt_size + 1
                                                                                                                           # because of [MASK] or obj (Not-zero BS, prompt length)
                full_perplexities.append(  perplexities[(perplexities * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1) )
            else:
                
                full_nlls.append( nlls[(nlls * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1)[0] ) # Method 1 each prompt is the same, only need the first one of the batch
                
                full_perplexities.append( perplexities[(perplexities * attention_mask) != 0].view(attention_mask.any(axis = 1).sum().item() ,-1)[0] )
                break
            
        # Stack lists of tensors into a big tensor
        if method > 1:
            full_nlls = torch.cat(full_nlls, axis = 0)
            full_perplexities = torch.cat(full_perplexities, axis = 0)
        else:
            full_nlls = torch.stack(full_nlls)
            full_perplexities = torch.stack(full_perplexities)       

        # Retrieve tokens from the template
        tokens, pos_x, pos_y = self.tokenizer.get_tokens_list(template = templates[0],
                                                              token_sequence = kwargs['token_sequence'])

        return {'nll': full_nlls,
                'perplexity': full_perplexities, 
                'tokens': {'pos_x': pos_x,
                           'pos_y': pos_y,
                           'tokens': tokens}}
        
    def _compute_lama_scores(self, 
                             df: pd.DataFrame, 
                             **kwargs) -> Tuple[dict[str, float], dict[str, dict[str, float]]]: # need to rework the model type thing in every method
        """
            Compute LAMA scores for one set of prompts (e.g. machine).
            
            It returns two quantities: the average P@k over the whole dataset and
            the average P@k rela by rela.
            
            Returns:
                scores (dict) keys: 'P@k'  value: (float) the computed P@k 
                                     k in {1,5,20,100}
                                     
                scores_by_rela (dict) key: (str) rela id (e.g. 'P276') value: (dict) key: (str) 'P@k' value: (float) P@k for the rela
        
        """
        
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

        num_eval = kwargs['num_eval'] if kwargs['num_eval'] != -1 else len(df)

        total_eval = 0
        
        for k in tqdm.tqdm(range(0, num_eval, kwargs['batch_size'])): # We'll do it by hand
        
            # Retrieve elem
            elems = df.iloc[k:k + kwargs['batch_size']]
            sub_surfaces = elems['sub_surface'] # [X]
            obj_surfaces = elems['obj_surface'] # [Y]
            templates = elems['template']
            relas = elems['predicate_id']

            # Create and tokenize template
            
            input_ids, attention_mask = self.tokenizer.tokenize(
                                                templates = templates,
                                                sub_surfaces = sub_surfaces,
                                                obj_surfaces = obj_surfaces,
                                                token_sequence = kwargs['token_sequence'],
                                                method = 3
                                                )
                
            mask_pos_i, mask_pos_j = torch.where(input_ids == self.tokenizer.tokenizer.mask_token_id)

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
            labels = self.tokenizer.tokenizer(obj_surfaces.tolist(), 
                                              padding = True, 
                                              return_tensors = 'pt')
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
        
        return ({k: v/total_eval for k,v in scores.items()}, 
                utils.compute_lama_scores_by_rela(relas_and_scores))
    
    
    
    def _compute_embeddings_analysis(self, 
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
        
        print(f"\t\tCosine Sim - avg: {np.round(results['cosim avg'],3)}; max: {np.round(results['cosim max'],3)}")
        
        
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
        
        pca_kmeans = KMeans(n_clusters = 2, n_init='auto').fit(embeds_pca)
        pca_kmeans_completeness = completeness_score(true_labels, pca_kmeans.labels_)
        pca_kmeans_homogeneity = homogeneity_score(true_labels, pca_kmeans.labels_)
        
        pca_spectral = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(embeds_pca)
        pca_spectral_completeness = completeness_score(true_labels, pca_spectral.labels_)
        pca_spectral_homogeneity = homogeneity_score(true_labels, pca_spectral.labels_)
        
        
        ### MDS ###
        
        if embeds3 is not None:
            pass
        else:
            full_embeds = np.concatenate(
                                    (embeds1.numpy(), 
                                    embeds2.numpy()), 
                                    axis = 0
                                    )
        
        mds = MDS(n_components=2, normalized_stress='auto')
        embeds_mds = mds.fit_transform(full_embeds.astype(np.float64)) # Shape [2*(N prompts), 2]
        
        # Clustering
        mds_kmeans = KMeans(n_clusters = 2, n_init='auto').fit(embeds_mds)
        mds_kmeans_completeness = completeness_score(true_labels, mds_kmeans.labels_)
        mds_kmeans_homogeneity = homogeneity_score(true_labels, mds_kmeans.labels_)
        
        mds_spectral = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(embeds_mds)
        mds_spectral_completeness = completeness_score(true_labels, mds_spectral.labels_)
        mds_spectral_homogeneity = homogeneity_score(true_labels, mds_spectral.labels_)
        
        # Store Results
        results['pca R2'] = reg_score
        results['pca 1-corr'] = (pearson.statistic, pearson.pvalue)
        
        results['pca kmeans completeness'] = pca_kmeans_completeness
        results['pca kmeans homogeneity'] = pca_kmeans_homogeneity
        results['pca spectral completeness'] = pca_spectral_completeness
        results['pca spectral homogeneity'] = pca_spectral_homogeneity
        
        results['pca embeds1'] = embeds_pca[:embeds1.shape[0]]
        results['pca embeds2'] = embeds_pca[embeds1.shape[0]:]
        
        
        results['mds kmeans completeness'] = mds_kmeans_completeness
        results['mds kmeans homogeneity'] = mds_kmeans_homogeneity
        results['mds spectral completeness'] = mds_spectral_completeness
        results['mds spectral homogeneity'] = mds_spectral_homogeneity
        
        results['mds embeds1'] = embeds_pca[:embeds1.shape[0]]
        results['mds embeds2'] = embeds_pca[embeds1.shape[0]:]
        
        
        # Print
        print(f"\t\tPCA - R²: {np.round(results['pca R2'],3)}; 1-corr: {np.round(results['pca 1-corr'][0])} (p = {np.round(results['pca 1-corr'][1])})")
        print(f"\t\tPCA Clustering\
                \n\t\t\tKMeans - completeness: {np.round(results['pca kmeans completeness'],3)}; homogeneity {np.round(results['pca kmeans homogeneity'],3)}\
                \n\t\t\tSpectral - completeness: {np.round(results['pca spectral completeness'],3)}; homogeneity {np.round(results['pca spectral homogeneity'],3)}")
        
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
    
    def _compute_curvatures_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            embeddings (Tensor) shape [Num Layers, Dataset Size, Length, Hidden Dim]
         
        Returns:
            curvatures (Tensor) shape [Dataset Size, Num Layers]
        """
        
        # First let's transpose the embeddings to [Dataset Size, Num Layers, Length, Hidden Dim]
        embeddings = embeddings.transpose(0,1) 
        
        # We'll do a loop on each elem of the dataset because otherwise it's too complicated
        # to get rid of the 0s from padding (and we don't want to compute the curvatures to 0)
        curvatures = []
        for k in tqdm.tqdm(range(embeddings.shape[0])):
            activations = embeddings[k] # Shape [Num Layers, Length, Hidden Dim]
            # Here we want to get rid of the padded 0: let's assume that an embedding value is never 0!
            try:
                pad_idx = torch.where(activations == 0)[1][0].item()
                activations = activations[:,:pad_idx,:]
            except:
                pass # Some tensors were not padded
            # store curvature layer by layer
            layersAvgCurvatures = []
            for layerActivation in activations:
                layersAvgCurvatures.append(self.avgCurvature(layerActivation))
            curvatures.append(layersAvgCurvatures)
        
        return torch.tensor(curvatures)
        
        
    @staticmethod
    def computeSeg(arr: torch.Tensor) -> torch.Tensor:
        """
            From an array of N points (ie. shape (N, d))
            returns N-1 segments:
            v_k = x_{k+1} - x_k
        """
        return arr[1:,:] - arr[:-1,:]

    @staticmethod
    def curvature(arr: torch.Tensor) -> torch.Tensor:
        """
            Compute curvature betweens vectors contained
            in arr (of shape (N, d)) as:
            c_k = arccos( <v_{k+1},v_k>/||v_{k+1}||.||v_k||) 
        """
        return torch.arccos(torch.sum(arr[1:,:] * arr[:-1,:], dim = 1)/(torch.norm(arr[1:, :], dim = 1) * torch.norm(arr[:-1,:], dim = 1)))

    @staticmethod
    def avgCurvature(arr: torch.Tensor) -> torch.Tensor:
        vs = MetricsWrapper.computeSeg(arr)
        curvs = MetricsWrapper.curvature(vs)
        return curvs.mean().item()
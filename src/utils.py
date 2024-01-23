

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad
from typing import List, Tuple, Dict


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
    
def compute_lama_scores_by_rela(raw_relas_and_scores: pd.core.frame.DataFrame) -> dict:
    """

        Compute LAMA scores but present the results rela by rela.

        Args:
            raw_relas_and_scores (pd Dataframe) columns: ['rela', 'filter', 'P@k']
                                                Contains the score for each row of the dataset.
                                                'rela' column stores the rela id of the row
                                                'filter' states whether (1) or not (0) [Y] was one token.
        Returns:
            scores (dict) key: (str) rela id (e.g. 'P276') value: (dict) key: (str) 'P@k' value: (float) P@k for the rela

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

def compute_perplexity(probs: torch.Tensor) -> torch.Tensor:
    """
        Here perplexity means according to us.
        
        Args:
            probs (tensor) shape (BS, vocab_size)
        
    """
    H = - (probs * torch.log(probs)).sum(axis = 1) # entropy base e
    return torch.exp(H) # shape BS

def select_subset(t: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    """
        Given a tensor of shape [BS, *] or [Num Layer, BS, *].
        Chose randomly a subset of size size according to dim.
    
    """
    assert dim in [0,1]
    
    if size > t.shape[dim]:
        return t
    
    idx = np.random.choice(t.shape[dim],
                           size,
                           replace=False)
    if dim == 0:
        return t[idx]
    elif dim == 1:
        return t[:,idx]


def process_data_to_classify(data_to_classify: str,
                             seeds: List[int],
                             lama_name: str) -> List[str]:
    """
        Small function that transforms 'lama-autoprompt' in [lama_name, f'autoprompt_seed{seed[0]}']
        for instance.
    
        Args:
            data_to_classify (str) e.g. 'lama-autoprompt'
            seeds (List of int) usually [0,1]
            lama_name (str)
        Returns:
            data_to_classify (List of str)
    """
    
    # Retrieve data to classify
    data_to_classify = data_to_classify.split('-')
    assert len(data_to_classify) == 2
    count_autoprompt = 0
    for k in range(2):
        if data_to_classify[k] == 'lama':
            data_to_classify[k] = lama_name
        elif data_to_classify[k] == 'autoprompt':
            data_to_classify[k] = f'autoprompt_seed{seeds[count_autoprompt]}'
            count_autoprompt += 1
    return data_to_classify

def token_sequence_or_not(data_to_clasify: List[str]) -> List[bool]:
    token_sequence_bools = {}
    for d in data_to_clasify:
        if 'lama' in d:
            # Only dataset that is not a Token sequence (for now) is LAMA derivatives
            token_sequence_bools[d] = False
        else:
            token_sequence_bools[d] = True
    return token_sequence_bools

def collapse_metrics_nlls_perplexities(res_dict: dict, 
                                       dataset_names: List[str]) -> dict:
    """
    
    Stack everytensor of a res_dict from metrics.compute_nlls_perplexities.
    
    Args:
        res_dict (dict) keys: 'nlls'         values: (dict) keys: (str) relas from LAMA  values: (dict) keys: (str) dataset name  values: (Tensor) 
                              'perplexities'         (dict) keys: (str) relas from LAMA  values: (dict) keys: (str) dataset name  values: (Tensor)
                              'tokens'
        dataset_names (list of str)
                              
    Returns:
        collapsed_res (dict) keys: dataset_name        values: (tuple of tensor) (nlls, perplexities)  
    
    """
    
    nlls = {d:[] for d in dataset_names}
    perplexities = {d:[] for d in dataset_names}
    for rela in res_dict['nlls'].keys():
        for d in dataset_names:
            nlls[d].append(res_dict['nlls'][rela][d])
            perplexities[d].append(res_dict['perplexities'][rela][d])
    maxLength = max([max([e.shape[1] for e in nlls[d]]) for d in dataset_names])
    for d in dataset_names:
        for k in range(len(nlls[d])):
            nlls[d][k] = pad(nlls[d][k], (0, maxLength - nlls[d][k].shape[1] + 1)) # The + 1 is intended to pad every tensor by at least 1 tensor
            perplexities[d][k] = pad(perplexities[d][k], (0, maxLength - perplexities[d][k].shape[1] + 1)) # The + 1 is intended to pad every tensor by at least 1 tensor
    
    return {d: (torch.cat(nlls[d]), torch.cat(perplexities[d])) for d in dataset_names}
    
    

### It was used to see if shuffling prompts resulted in a drop in perf. I removed it
#   for readability but if I need it one day it's here. And results are still there as well. ###

def shuffle_template(template: str, 
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
    

"""  
def average_rela_results(
                scores_by_uuid: Dict[str, Dict[str, float]], 
                probs_by_uuid: Dict[str, List[float]]
                ) -> Tuple[ Dict[str, float] ]:
    # Scores
    scores = {'P@1': 0,
              'P@5': 0,
              'P@20': 0,
              'P@100': 0}
    for uuid in scores_by_uuid.keys():
        for k, v in scores_by_uuid[uuid].items():
            scores[k] += v
    scores = {k: v/len(scores_by_uuid) for k,v in scores.items()}

    # Probs
    prob = 0.
    for v in probs_by_uuid.values():
        prob += sum(v)/len(v)
    prob /= len(probs_by_uuid)

    return scores, prob
"""
    
def compute_variation(
                probs_before: list, 
                probs_after: list
                ) -> float:
  # Scores
  prob_changes = []
  for uuid in probs_before.keys():
    _prob_changes_uuid = []
    for k in range(len(probs_before[uuid])):
      prob_change = (probs_after[uuid][k] - probs_before[uuid][k]) / probs_before[uuid][k]
      _prob_changes_uuid.append(prob_change)
    prob_changes.append( sum(_prob_changes_uuid)/len(_prob_changes_uuid) )

  return sum(prob_changes)/len(prob_changes)
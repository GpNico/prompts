
from datasets import load_dataset
import glob
import json
import numpy as np
import os
from pathlib import Path
import pandas as pd
import shutil
import stat
import tqdm
from transformers import PreTrainedTokenizer
from typing import Dict, List, Union

from src.utils import get_template_from_lama


def load_data_wrapper(datasets_to_load: list,
                      **kwargs) -> dict:
    """
        Load every dataset in datasets_to_load.
        
        rk: for autoprompt one needs model_name & seed
        
        Args:
            datasets_to_load (List[str]) names of the datasets: lama, autoprompt, random_prompts
    """
    
    datasets = {}
    
    if 'lama' in datasets_to_load:
        # This one is mandatory
        lama_dataset = load_lama(debug = kwargs['debug_flag'])
        datasets['lama'] = lama_to_pandas(lama_dataset)
        
    if 'mlama' in datasets_to_load:
        datasets['mlama'] = load_mlama(kwargs['langs_to_load'])
        
    if 'autoprompt' in datasets_to_load:
        for seed, path in zip(kwargs['seeds'], kwargs['autoprompt_paths']):
            autoprompt = load_autoprompt(path = path)
            if 'lama' in datasets.keys():
                datasets[f'autoprompt_seed{seed}'] = autoprompt_as_lama(
                                                            lama = datasets['lama'],
                                                            autoprompt = autoprompt
                                                            )
            elif 'mlama' in datasets.keys():
                datasets[f'autoprompt_seed{seed}'] = autoprompt_as_lama(
                                                            lama = datasets['mlama'],
                                                            autoprompt = autoprompt
                                                            )

    if 'random' in datasets_to_load:
        rd_prompts = create_random_prompts(
                            tokenizer = kwargs['tokenizer'],
                            n_tokens = kwargs['n_tokens'],
                            n_prompts = kwargs['n_random_prompts'],
                            model_type = kwargs['model_type']
                            )
        if 'lama' in datasets.keys():
            datasets['random'] = random_prompts_as_lama(
                                    lama = datasets['lama'],
                                    rd_prompts = rd_prompts
                                    )
        elif 'mlama' in datasets.keys():
            datasets['random'] = random_prompts_as_lama(
                                    lama = datasets['mlama'],
                                    rd_prompts = rd_prompts
                                    )
    
    return datasets


def load_lama(debug: bool = False):
    """
        Args:
            debug (bool) if set to True then we load only 10% of LAMA
    """
    if debug:
        return load_dataset("lama", "trex", split="train[:10%]")#keep_in_memory = True, num_proc = 8)
    else:
        return load_dataset("lama", "trex", split="train")

def load_mlama(langs_to_load: list = []):
    mlama_path = os.path.join('data', 'mlama1.1')
    
    langs = os.listdir(mlama_path)
    for lang in langs_to_load:
        assert lang in langs
    if len(langs_to_load) == 0:
        langs_to_load = langs 
    
    data = []
    for lang in langs_to_load:
        lang_path = Path(os.path.join(mlama_path, lang))
        # Load Templates
        templates_path =lang_path.joinpath('templates')
        templates = {}
        with open(os.path.join(templates_path, 'train.jsonl'), 'r') as f:
            for line in f:
                data_line = json.loads(line) # dict
                templates[data_line['relation']] = data_line['template']
        with open(os.path.join(templates_path, 'dev.jsonl'), 'r') as f:
            for line in f:
                data_line = json.loads(line) # dict
                templates[data_line['relation']] = data_line['template']

        # Load Relas 
        relas = lang_path.glob('P[0-9]*')
        for rela in relas:
            with open(os.path.join(rela, 'train.jsonl'), 'r') as f:
                for line in f:
                    data_line = json.loads(line) # dict
                    # renaming obj_label into obj surface to match lama template
                    data_line['obj_surface'] = data_line['obj_label']
                    del data_line['obj_label']
                    data_line['sub_surface'] = data_line['sub_label']
                    del data_line['sub_label']
                    # adding lang and rela as identifier & template
                    data_line['lang'] = lang
                    data_line['predicate_id'] = rela.name
                    data_line['template'] = templates[rela.name]
                    data.append(data_line)
    print(f"Loaded {len(data)} rows for {' '.join(langs_to_load)}")
    df = pd.DataFrame(data = data)

    return df
    
        

def lama_to_pandas(dataset,
                   clean: bool = True) -> pd.core.frame.DataFrame:
    print("Converting HuggingFace dataset to pandas...")
    if clean:
        return pd.DataFrame(dataset).drop(['uuid', 'obj_uri', 'sub_uri', 'obj_label', 'masked_sentence', 'template_negated', 'label', 'description', 'type'], axis = 1).drop_duplicates().reset_index(drop = True)
    else: 
        return pd.DataFrame(dataset)

def load_autoprompt(path: str) -> pd.core.frame.DataFrame:
    return pd.read_csv(path)

def autoprompt_as_lama(lama: pd.core.frame.DataFrame, 
                       autoprompt: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
        Create a dataset like lama (clean pandas form) but
        using the autoprompts as template.
        
        Args:
            lama (DataFrame) the clean pandas version of lama
            autoprompts (DataFrame) loaded from autoprompts save
    """ 
    print("Create AutoPrompt Dataset...")
    
    df_autoprompt = lama.copy()

    for k in tqdm.tqdm(range(len(lama))):
        elem = df_autoprompt.iloc[k]
        rela = elem['predicate_id']

        prompt = autoprompt[autoprompt['id'] == rela]['prompt'].iloc[0]
        prompt = '[X] ' + prompt + ' [Y] .'

        df_autoprompt['template'][k] = prompt

    return df_autoprompt

def create_random_prompts(tokenizer, 
                          n_tokens: int, 
                          n_prompts: int,
                          model_type: str) -> list:
    """
        Create n_prompts random prompts of size n_tokens
        using the model's vocabulary.
        
        Args:
            tokenizer (Huggingface's tokenizer) model's tokenizer
            n_tokens (int) size of the created prompts 
            n_prompts (int) ...
    
    """
    rd_prompts = []
    for k in range(n_prompts):
        if model_type == 'bert':
            # Token Id 0 is [PAD], from 1 to 99 it's [unusedX]
            # 100 [UNK], 101 [CLS], 102 [SEP], 103 [MASK]
            # from 104 to 998 it's [unusedXXX]
            ids = np.random.choice(
                        np.arange(999, tokenizer.vocab_size),
                        n_tokens, 
                        replace = True
                        )
        else:
            raise Exception(f'model_type {model_type} is not supported yet.')
        prompt = ""
        for id in ids:
            prompt += " " + tokenizer.decode(id)
        rd_prompts.append(prompt[1:])
    return rd_prompts

def random_prompts_as_lama(lama: pd.core.frame.DataFrame,
                           rd_prompts: list) -> pd.core.frame.DataFrame:
    """
        Create a dataset like lama (clean pandas form) but
        using the random_prompts instead of autoprompt as template.
    """
    # First step: put it undr autoprompts csv format
    ids = list(set(lama['predicate_id']))
    rd_prompts_df = pd.DataFrame(
                          data = {"id": [ids[k] for k in range(len(ids))],
                                  "prompt": rd_prompts[:len(ids)]} # here we limit the number of rd prompts: this will change 
                          )
    # Second step: we can use the autoprompt function
    return autoprompt_as_lama(lama, rd_prompts_df)


def load_pararel():
    """
        Load the ParaRel with the above framework. 
    """
    ...
    
def load_pararel_by_uuid(
                    tokenizer: PreTrainedTokenizer = None,
                    remove_multiple_tokens: bool = True,
                    autoprompt: bool = False,
                    lower: bool = False) -> Dict[str, Dict[str, Union[List[str], str]]]:
    """
        Load each JSON from pararel folder, create sentences,
        replace [X] by sub_label, [Y] by [MASK] and stores obj_label
        as Y.
        
        If autoprompt is set to True then it loads Autoprompt data
        the exact same way.
        
        If remove_multiple_tokens is set to True then each Y that is tokenized in
        more than two tokens will be skipped.
        
        If lower is set to True then we apply .lower() to each str.
        Thats MANDATORY when working with uncased models.
        (Well not realy because the rest of the code tackles this
        but I had issue with this so better safe than sorry).
        
        This function is the exception of this file as it is to be called on its 
        own i.e. not through the load_data_wrapper.
    """
    
    assert tokenizer or not(remove_multiple_tokens)
    
    if not(os.path.exists(os.path.join('data', 'pararel'))):
        # Clone ParaRel data
        original_working_directory = os.getcwd()
        try:
            os.chdir('data')
            github_repo_url = 'https://github.com/yanaiela/pararel'
            os.system(f'git clone {github_repo_url}')
            
            shutil.move(
                os.path.join('pararel', 'data'),
                'pararel_data'
                )
            # Change permissions of all files and directories in the given path
            for root, dirs, files in os.walk('pararel'):
                for path in dirs + files:
                    full_path = os.path.join(root, path)
                    os.chmod(full_path, stat.S_IWRITE)
            shutil.rmtree('pararel')
            os.rename('pararel_data', 'pararel')
        finally:
            os.chdir(original_working_directory)
    
    # Get relation predicates ids
    predicate_ids = list(
                        set(
                            [n[:-6] for n in os.listdir(os.path.join( 'data', 'pararel', 'trex_lms_vocab'))]
                            ).intersection(
                                    set(
                                        [n[:-6] for n in os.listdir(os.path.join('data', 'pararel', 'pattern_data', 'graphs_json'))]
                                        )
                                    )
                        )
    
    # Load Data
    dataset = {}
    for predicate_id in predicate_ids:
        # Load Prompts
        prompts = []
        if autoprompt:
            with open(os.path.join('data', 'autoprompt_jsonl', f'{predicate_id}.jsonl'), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data['pattern'])
        else:
            with open(os.path.join('data', 'pararel', 'pattern_data', 'graphs_json', f'{predicate_id}.jsonl'), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data['pattern'])

        # Load Xs, Ys
        trex_vocab = []
        with open(os.path.join('data', 'pararel', 'trex_lms_vocab', f'{predicate_id}.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                trex_vocab.append((data['sub_label'], data['obj_label'], data['uuid']))
                
        # Oraganize and Process Data
        num_prompts = len(prompts)
        if num_prompts < 4:
            print(f"Relation {predicate_id} skipped. Not enough prompts.")
            continue

        dataset_by_uuid = {}
        for X, Y, uuid in trex_vocab:
            if remove_multiple_tokens:
                input_ids = tokenizer(Y).input_ids[1:-1]
            if len(input_ids)>1:
                continue
            if lower:
                Y = Y.lower()
            dataset_by_uuid[uuid] = {"sentences": [], 'Y': Y, 'num_prompts': num_prompts}
            for prompt in prompts:
                if lower:
                    sentence = prompt.replace('[X]', X).lower()
                    sentence = sentence.replace('[y]', '[MASK]') # /!\
                else:
                    sentence = prompt.replace('[X]', X).replace('[Y]', '[MASK]')
                dataset_by_uuid[uuid]['sentences'].append(sentence)
        
        # Store Data
        dataset[predicate_id] = dataset_by_uuid
        
    return dataset



from datasets import load_dataset
import glob
import json
import numpy as np
import os
from pathlib import Path
import pandas as pd
import tqdm

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
        lama_dataset = load_lama()
        datasets['lama'] = lama_to_pandas(lama_dataset)
        lama_raw = get_template_from_lama(df = datasets['lama'])
        datasets['lama_raw'] = list(lama_raw.values())
    if 'mlama' in datasets_to_load:
        datasets['mlama'] = load_mlama(kwargs['langs_to_load'])
    if 'autoprompt' in datasets_to_load:
        autoprompt_raw = []
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
            autoprompt_raw += autoprompt['prompt'].tolist()
        datasets['autoprompt_raw'] = autoprompt_raw
    if 'random' in datasets_to_load:
        rd_prompts = create_random_prompts(
                            tokenizer = kwargs['tokenizer'],
                            n_tokens = kwargs['n_tokens'],
                            n_prompts = kwargs['n_random_prompts']
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
        datasets['random_raw'] = rd_prompts
    
    return datasets


def load_lama():
    return load_dataset("lama", "trex")

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
    if clean:
        return pd.DataFrame(dataset['train']).drop(['uuid', 'obj_uri', 'sub_uri', 'obj_label', 'masked_sentence', 'template_negated', 'label', 'description', 'type'], axis = 1).drop_duplicates().reset_index(drop = True)
    else: 
        return pd.DataFrame(dataset['train'])

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

def create_random_prompts(tokenizer, n_tokens: int, n_prompts: int) -> list:
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
        ids = np.random.choice(tokenizer.vocab_size, n_tokens, replace = True) 
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



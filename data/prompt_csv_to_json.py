import argparse
import json
import os
import pandas as pd

if __name__ == '__main__':
    
    # /!\ Needs to be executed from ./data folder /!\
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        default='bert-base-uncased')
    parser.add_argument('--path', 
                        type=str,
                        default = "vanilla")
    args = parser.parse_args()
    
    # get path
    full_path = os.path.join('data', 'autoprompt', args.path)
    
    
    # Get autoprompt files
    autoprompt_files = [n for n in os.listdir(full_path) if ('.csv' in n) and (args.model_name in n)]
    
    # Read prompt files and get autoprompts
    rela2prompts = {}
    for file_name in autoprompt_files:
        seed = file_name.split('_')[-1][:-4]
        df = pd.read_csv(os.path.join(full_path, file_name))
        for k in range(len(df)):
            elem = df.iloc[k]
            rela, prompt = elem['id'], elem['prompt']
            prompt = '[X] ' + prompt + ' [Y] .'
            if rela in rela2prompts.keys():
                rela2prompts[rela].append({"pattern": prompt,
                                           "seed": seed})
            else:
                rela2prompts[rela] = [{"pattern": prompt,
                                       "seed": seed}]
    
    # Write in jsonl files
    folder_name = os.path.join('data', 'autoprompt_jsonl', args.path)
    os.makedirs(folder_name, exist_ok=True)
    for rela in rela2prompts.keys():
        with open(os.path.join(folder_name, f'{rela}.jsonl'), 'w') as jsonl_file:
            for item in rela2prompts[rela]:
                json_line = json.dumps(item)
                jsonl_file.write(json_line + '\n')
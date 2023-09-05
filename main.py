
import argparse
import os
import pickle
import torch
import tqdm
from transformers import AutoTokenizer

from src.classifier import PromptClassifier
from src.data import load_data_wrapper
from src.metrics import MetricsWrapper
from src.models import ModelWrapper
from src.plots import plot_lama_scores, plot_rela_nll_perplexity


if __name__ == '__main__':
    
    ### Argparse ###
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', 
                        type=str, 
                        default='bert',
                        help='For now: bert, gpt, xlm.')
    parser.add_argument('--model_name', 
                        type=str, 
                        default='bert-base-uncased',
                        help="HuggingFace model's name.")
    parser.add_argument('--datasets', 
                        type=str, 
                        default='lama-autoprompt-random',
                        help="For now only lama is supported")
    parser.add_argument('--seeds', 
                        type=str, 
                        default='0-1',
                        help="Seeds for AutoPrompt.")
    parser.add_argument('--n_random_prompts', 
                        type=int, 
                        default=82,
                        help="Num of random prompts to generate.")
    parser.add_argument('--n_tokens', 
                        type=int, 
                        default=4,
                        help="Num tokens in the random prompts.")
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=16)
    parser.add_argument('--prompt_method', 
                        type=int, 
                        default=1,
                        help="How the prompt will be presented to the model to compute its NLL.")
    parser.add_argument('--cls', 
                        type=str, 
                        default='last_nll',
                        help="Classifier to use.")
    parser.add_argument('--lama_scores', 
                        action='store_true',
                        help="Compute LAMA scores.")
    parser.add_argument('--rela_nll', 
                        action='store_true',
                        help="Compute NLLs of each relation prompt.")
    parser.add_argument('--rela_perplexity', 
                        action='store_true',
                        help="Compute perplexity of each relation prompt.")
    parser.add_argument('--prompt_cls', 
                        action='store_true',
                        help="Compute proportion of prompts in model vocab.")
    args = parser.parse_args()
    
    
    ### Device, Model, Tokenizer ###
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    
    model = ModelWrapper(
                    model_type = args.model_type,
                    model_name = args.model_name,
                    device = device
                    )
    
    ### Load Dataset(s) ###
    
    datasets_to_load = args.datasets.split('-')
    
    # under the assumption that we load autoprompts, useless otherwise
    
    seeds = [int(seed) for seed in args.seeds.split('-')]
    
    autoprompt_paths = [os.path.join(
                            "data",
                            f"{args.model_name}_en_seed_{seed}.csv"
                            ) for seed in seeds]
        
    datasets = load_data_wrapper(
                            datasets_to_load = datasets_to_load,
                            autoprompt_paths = autoprompt_paths,
                            tokenizer = tokenizer,
                            n_tokens = args.n_tokens,
                            n_random_prompts = args.n_random_prompts,
                            seeds = seeds
                            )
    
    ### Compute Metrics ###
    
    metrics = MetricsWrapper(
                    model = model,
                    tokenizer = tokenizer,
                    device = device
                    )
    
    if args.lama_scores:
        
        print("Compute LAMA score...")
                
        lama_scores_lama = metrics.evaluate_on_lama(
                                    dataset = datasets['lama'],
                                    num_eval = -1,
                                    autoprompt = False,
                                    batch_size = args.batch_size
                                    )
        
        lama_scores_autoprompt = {}
        if 'autoprompt' in datasets_to_load:
            for seed in seeds:
                lama_scores_autoprompt[seed] = metrics.evaluate_on_lama(
                                                dataset = datasets[f'autoprompt_seed{seed}'],
                                                num_eval = -1,
                                                autoprompt = True,
                                                batch_size = args.batch_size
                                                )
                
        lama_scores_random = metrics.evaluate_on_lama(
                                    dataset = datasets['random'],
                                    num_eval = -1,
                                    autoprompt = True,
                                    batch_size = args.batch_size
                                    )
    
    if args.rela_nll or args.rela_perplexity:
        
        print("Compute NLL & Perplexity score...")
        
        relations_ids = list(set(datasets['lama']['predicate_id']))
        
        nlls = {}
        perplexities = {}
        tokens = {}
        
        for rela in tqdm.tqdm(relations_ids):
            
            rela_nlls = {}
            rela_perplexities = {}
            rela_tokens = {}
            
            # LAMA
            lama_res = metrics.compute_nll_perplexity(
                                    df = datasets['lama'][datasets['lama']['predicate_id'] == rela], 
                                    autoprompt = False, 
                                    method = args.prompt_method,
                                    batch_size = args.batch_size
                                    )
            rela_nlls['lama'] = lama_res['nll']
            rela_perplexities['lama'] = lama_res['perplexity']
            rela_tokens['lama'] = lama_res['tokens']
            
            # AutoPrompt
            for seed in seeds:
                autoprompt_res = metrics.compute_nll_perplexity(
                                            df = datasets[f'autoprompt_seed{seed}'][datasets[f'autoprompt_seed{seed}']['predicate_id'] == rela], 
                                            autoprompt = True, 
                                            method = args.prompt_method,
                                            batch_size = args.batch_size
                                            )
                rela_nlls[f'autoprompt_seed{seed}'] = autoprompt_res['nll']
                rela_perplexities[f'autoprompt_seed{seed}'] = autoprompt_res['perplexity']
                rela_tokens[f'autoprompt_seed{seed}'] = autoprompt_res['tokens']

            # Random
            random_res = metrics.compute_nll_perplexity(
                                        df = datasets['random'][datasets['random']['predicate_id'] == rela], 
                                        autoprompt = True, 
                                        method = args.prompt_method,
                                        batch_size = args.batch_size
                                        )
            rela_nlls['random'] = random_res['nll']
            rela_perplexities['random'] = random_res['perplexity']
            rela_tokens['random'] = random_res['tokens']
            
            # store
            nlls[rela] = rela_nlls
            perplexities[rela] = rela_perplexities
            tokens[rela] = rela_tokens
            
    if args.prompt_cls:
        
        # Init Classifier
        cls = PromptClassifier(
                            model = model,
                            model_name = args.model_name,
                            tokenizer = tokenizer,
                            device = device,
                            cls = args.cls
                        )
        
        # Compute NLLs
        random_prompts_list = datasets['random_raw']
        autoprompt_list = datasets['autoprompt_raw']
        
        random_prompts_tok = cls.tokenize(random_prompts_list)
        autoprompt_tok = cls.tokenize(autoprompt_list)
        
        if args.cls in ['last_nll', 'last_perplexity', 'linear_reg', 'logistic_reg']:
            dataset = {'0': random_prompts_tok,
                       '1': autoprompt_tok}
            cls.train(dataset = dataset)
            
        cls.compute_roc_curve(dataset = dataset)   
        
        exit(0)
    
    ### Plot & Save Results ###
    
    if args.lama_scores:
        
        os.makedirs(os.path.join('results', "lama_scores"), exist_ok=True)
        
        with open(os.path.join('results', 'lama_scores', f'{args.model_name}_lama_scores_lama.pickle'), 'wb') as f:
            pickle.dump(lama_scores_lama, f)
            
        if 'autoprompt' in datasets_to_load:
            for seed in seeds:
                with open(os.path.join('results', 'lama_scores', f'{args.model_name}_lama_scores_autoprompt_seed{seed}.pickle'), 'wb') as f:
                    pickle.dump(lama_scores_autoprompt[seed], f)
        
        plot_lama_scores(
            lama_scores_lama = lama_scores_lama,
            lama_scores_autoprompt = lama_scores_autoprompt,
            lama_scores_random = lama_scores_random,
            model_name = args.model_name,
            seeds = seeds
        )
        
    if args.rela_nll:
        
        os.makedirs(os.path.join('results', "nll", f"method_{args.prompt_method}"), exist_ok=True)
        
        with open(os.path.join('results', "nll", f"method_{args.prompt_method}", f'{args.model_name}_nlls_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(nlls, f)
        with open(os.path.join('results', "nll", f"method_{args.prompt_method}", f'{args.model_name}_tokens_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(tokens, f)
        
        for seed in seeds:
            for rela in relations_ids:
                plot_rela_nll_perplexity(
                    nlls[rela],
                    tokens[rela],
                    method = args.prompt_method,
                    model_name = args.model_name,
                    rela = rela,
                    seed = seed,
                    perplexity = False
                )
                
    if args.rela_perplexity:
        
        os.makedirs(os.path.join('results', "perplexity", f"method_{args.prompt_method}"), exist_ok=True)
        
        with open(os.path.join('results', "perplexity", f"method_{args.prompt_method}", f'{args.model_name}_perplexities_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(perplexities, f)
        with open(os.path.join('results', "perplexity", f"method_{args.prompt_method}", f'{args.model_name}_tokens_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(tokens, f)
        
        for seed in seeds:
            for rela in relations_ids:
                plot_rela_nll_perplexity(
                    perplexities[rela],
                    tokens[rela],
                    method = args.prompt_method,
                    model_name = args.model_name,
                    rela = rela,
                    seed = seed,
                    perplexity = True
                )
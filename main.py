
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
from src.plots import plot_lama_scores, plot_rela_nll_perplexity, plot_embeddings_dim_red


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
    parser.add_argument('--langs_to_load', 
                        type=str, 
                        default='',
                        help="If mlama selected, indicates the langs to load in mlama.")
    parser.add_argument('--seeds', 
                        type=str, 
                        default='0-1',
                        help="Seeds for AutoPrompt.")
    parser.add_argument('--n_random_prompts', 
                        type=int, 
                        default=128,
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
    parser.add_argument('--cls_what', 
                        type=str, 
                        default='prompt-random',
                        help="What to distinguish.")
    parser.add_argument('--lama_scores', 
                        action='store_true',
                        help="Compute LAMA scores.")
    parser.add_argument('--shuffle', 
                        action='store_true',
                        help="Shuffle prompts.")
    parser.add_argument('--rela_nll', 
                        action='store_true',
                        help="Compute NLLs of each relation prompt.")
    parser.add_argument('--rela_perplexity', 
                        action='store_true',
                        help="Compute perplexity of each relation prompt.")
    parser.add_argument('--prompt_cls', 
                        action='store_true',
                        help="Compute proportion of prompts in model vocab.")
    parser.add_argument('--embeds_analysis',
                        action="store_true",
                        help="Compute prompts embeddings and various metrics on them.")
    args = parser.parse_args()
    
    
    if not('bert' in args.model_name):
        raise Exception('Be careful mate, there is a lot of mask_token, sep_token references \
                        which will be a pain to solve!')
    
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
    if 'lama' in datasets_to_load:
        lama_name = 'lama'
    elif 'mlama' in datasets_to_load:
        lama_name = 'mlama'
    
    langs_to_load = args.langs_to_load.split('-')
    if langs_to_load == ['']:
        langs_to_load = []
    
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
                            seeds = seeds,
                            langs_to_load = langs_to_load
                            )
    
    ### Compute Metrics ###
    
    metrics = MetricsWrapper(
                    model = model,
                    tokenizer = tokenizer,
                    device = device
                    )
    
    if args.lama_scores:
        
        print("Compute LAMA score...")
                
        lama_scores_lama, scores_by_rela_lama = metrics.evaluate_on_lama(
                                    dataset = datasets[lama_name],
                                    num_eval = -1,
                                    autoprompt = False,
                                    batch_size = args.batch_size,
                                    shuffle = args.shuffle
                                    )
        
        lama_scores_autoprompt = {}
        scores_by_rela_autoprompt = {}
        if 'autoprompt' in datasets_to_load:
            for seed in seeds:
                lama_scores_autoprompt[seed], scores_by_rela_autoprompt[seed] = metrics.evaluate_on_lama(
                                                                    dataset = datasets[f'autoprompt_seed{seed}'],
                                                                    num_eval = -1,
                                                                    autoprompt = True,
                                                                    batch_size = args.batch_size,
                                                                    shuffle = args.shuffle
                                                                    )
                
        lama_scores_random, scores_by_rela_random = metrics.evaluate_on_lama(
                                        dataset = datasets['random'],
                                        num_eval = -1,
                                        autoprompt = True,
                                        batch_size = args.batch_size,
                                        shuffle = args.shuffle
                                        )
        
        
    if args.embeds_analysis or ('cluster' in args.cls):
        
        print("Compute Embeddings Anlysis...")
        
        embeds_lama, rela2embeds_lama = metrics.compute_embeddings(
                                    df = datasets[lama_name],
                                    autoprompt = False
                                    )
        
        embeds_autoprompt = {}
        rela2embeds_autoprompt = {}
        if 'autoprompt' in datasets_to_load:
            for seed in seeds:
                embeds_autoprompt[seed], rela2embeds_autoprompt[seed] = metrics.compute_embeddings(
                                    df = datasets[f'autoprompt_seed{seed}'],
                                    autoprompt = True
                                    )
                
        embeds_random, rela2embeds_random = metrics.compute_embeddings(
                                    prompt_list = datasets['random_raw'],  # Random Prompt do not require DataFrame
                                    autoprompt = True
                                    )
                
        print("Human vs AutoPrompt (seed 0)")
        embeds_analysis_lama_autoprompt0 = metrics.compute_embeddings_analysis(embeds_lama, embeds_autoprompt[0])
        embeds_analysis_lama_autoprompt0['label1'] = 'LAMA'
        embeds_analysis_lama_autoprompt0['label2'] = 'AutoPrompt (seed 0)'
        print("Human vs AutoPrompt (seed 1)")
        _ = metrics.compute_embeddings_analysis(embeds_lama, embeds_autoprompt[1])
        print("Human vs Random (Baseline)")
        embeds_analysis_lama_random = metrics.compute_embeddings_analysis(embeds_lama, embeds_random)
        embeds_analysis_lama_random['label1'] = 'LAMA'
        embeds_analysis_lama_random['label2'] = 'Random'
        print("AutoPrompt (seed 1) vs AutoPrompt (seed 0)")
        _ = metrics.compute_embeddings_analysis(embeds_autoprompt[0], embeds_autoprompt[1])
        print("Random vs AutoPrompt (seed 0)")
        embeds_analysis_autoprompt0_random = metrics.compute_embeddings_analysis(embeds_autoprompt[0], embeds_random)
        embeds_analysis_autoprompt0_random['label1'] = 'AutoPrompt (seed 0)'
        embeds_analysis_autoprompt0_random['label2'] = 'Random'
        
    
    
    if args.rela_nll or args.rela_perplexity:
        
        print("Compute NLL & Perplexity score...")
        
        relations_ids = list(set(datasets[lama_name]['predicate_id']))
        
        nlls = {}
        perplexities = {}
        tokens = {}
        
        for rela in tqdm.tqdm(relations_ids):
            
            rela_nlls = {}
            rela_perplexities = {}
            rela_tokens = {}
            
            # LAMA
            lama_res = metrics.compute_nll_perplexity(
                                    df = datasets[lama_name][datasets[lama_name]['predicate_id'] == rela], 
                                    autoprompt = False, 
                                    method = args.prompt_method,
                                    batch_size = args.batch_size
                                    )
            rela_nlls[lama_name] = lama_res['nll']
            rela_perplexities[lama_name] = lama_res['perplexity']
            rela_tokens[lama_name] = lama_res['tokens']
            
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
                            cls = args.cls,
                            cls_what = args.cls_what,
                            batch_size = args.batch_size
                        )
        
        # Compute NLLs
        cls_what = args.cls_what.split('-')
        assert len(cls_what) == 2 # TEMP
        tok_list = {}
        if 'random' in cls_what:
            random_prompts_list = datasets['random_raw']
            tok_list['random'] = cls.tokenize(random_prompts_list, 
                                              autoprompt=True)
        if 'prompt' in cls_what:
            autoprompt_list = datasets['autoprompt_raw']
            tok_list['prompt'] = cls.tokenize(autoprompt_list,
                                              autoprompt=True)
        if 'human' in cls_what:
            lama_list = datasets['lama_raw']
            tok_list['human'] = cls.tokenize(lama_list)
            
        
        if args.cls in ['last_nll', 'last_perplexity', 'logistic_reg', 'last_nll_reg']:
            # So dataset 0 is supposed to be random in the pairs human-random & prompt-random
            # and is supposed to be prompt in the pairs human-prompt
            if 'random' in cls_what:
                other_name = 'human' if 'human' in cls_what else 'prompt'
                dataset = {'0': tok_list['random'],
                           '1': tok_list[other_name]}
            else:
                dataset = {'0': tok_list['prompt'],
                           '1': tok_list['human']}
            cls.train(dataset = dataset)
            cls.chose_best_threshold(dataset = dataset)  
        elif args.cls in ['cluster-pca']:
            pass 
        
        if cls_what == ['prompt', 'random'] or cls_what == ['random', 'prompt']:
            cls.compute_prompt_proportion(
                n_tokens = args.n_tokens,
                n_eval = 2000
                )
    
    ### Plot & Save Results ###
    
    if args.lama_scores:
        
        os.makedirs(os.path.join('results', "lama_scores"), exist_ok=True)
        
        if args.shuffle:
            scores_name_lama = f'{args.model_name}_{lama_name}_scores_{lama_name}_shuffled.pickle'
            scores_by_rela_name_lama = f'{args.model_name}_{lama_name}_scores_by_rela_{lama_name}_shuffled.pickle'
            scores_name_autoprompt = args.model_name + f'_{lama_name}' + '_scores_autoprompt_seed{}_shuffled.pickle'
            scores_by_rela_name_autoprompt = args.model_name + f'_{lama_name}' + '_scores_by_rela_autoprompt_seed{}_shuffled.pickle'
        else:
            scores_name_lama = f'{args.model_name}_{lama_name}_scores_{lama_name}.pickle'
            scores_by_rela_name_lama = f'{args.model_name}_{lama_name}_scores_by_rela_{lama_name}.pickle'
            scores_name_autoprompt = args.model_name + f'_{lama_name}' + '_scores_autoprompt_seed{}.pickle'
            scores_by_rela_name_autoprompt = args.model_name + f'_{lama_name}' + '_scores_by_rela_autoprompt_seed{}.pickle'
        
        with open(os.path.join('results', 'lama_scores', scores_name_lama), 'wb') as f:
            pickle.dump(lama_scores_lama, f)
        
        with open(os.path.join('results', 'lama_scores', scores_by_rela_name_lama), 'wb') as f:
            pickle.dump(scores_by_rela_lama, f)
            
        if 'autoprompt' in datasets_to_load:
            for seed in seeds:
                with open(os.path.join('results', 'lama_scores', scores_name_autoprompt.format(seed) ), 'wb') as f:
                    pickle.dump(lama_scores_autoprompt[seed], f)
                    
                with open(os.path.join('results', 'lama_scores', scores_by_rela_name_autoprompt.format(seed)), 'wb') as f:
                    pickle.dump(scores_by_rela_autoprompt[seed], f)
        
        plot_lama_scores(
            lama_scores_lama = lama_scores_lama,
            lama_scores_autoprompt = lama_scores_autoprompt,
            lama_scores_random = lama_scores_random,
            model_name = args.model_name,
            seeds = seeds,
            shuffle = args.shuffle,
            lama_name = lama_name
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
                
    if args.embeds_analysis:
        
        os.makedirs(os.path.join('results', "embeddings_analysis", "PCA"), exist_ok=True)
        os.makedirs(os.path.join('results', "embeddings_analysis", "MDS"), exist_ok=True)
        
        
        plot_embeddings_dim_red(
                        embeds1 = embeds_analysis_lama_autoprompt0['pca embeds1'],
                        embeds2 = embeds_analysis_lama_autoprompt0['pca embeds2'],
                        embeds3 = None,
                        label1 = embeds_analysis_lama_autoprompt0['label1'],
                        label2 = embeds_analysis_lama_autoprompt0['label2'],
                        label3 = None,
                        annots1 = list(rela2embeds_lama.keys()),
                        annots2 = list(rela2embeds_lama.keys()),
                        annots3 = None,
                        dim_red_name = 'PCA',
                        model_name = args.model_name
                        )
        
        # Remark: Not neccessary to have as many random as lama
        plot_embeddings_dim_red(
                        embeds1 = embeds_analysis_lama_random['pca embeds1'],
                        embeds2 = embeds_analysis_lama_random['pca embeds2'],
                        embeds3 = None,
                        label1 = embeds_analysis_lama_random['label1'],
                        label2 = embeds_analysis_lama_random['label2'],
                        label3 = None,
                        annots1 = list(rela2embeds_lama.keys()),
                        annots2 = [],
                        annots3 = None,
                        dim_red_name = 'PCA',
                        model_name = args.model_name
                        )
        
        plot_embeddings_dim_red(
                        embeds1 = embeds_analysis_autoprompt0_random['pca embeds1'],
                        embeds2 = embeds_analysis_autoprompt0_random['pca embeds2'],
                        embeds3 = None,
                        label1 = embeds_analysis_autoprompt0_random['label1'],
                        label2 = embeds_analysis_autoprompt0_random['label2'],
                        label3 = None,
                        annots1 = list(rela2embeds_lama.keys()),
                        annots2 = [],
                        annots3 = None,
                        dim_red_name = 'PCA',
                        model_name = args.model_name
                        )
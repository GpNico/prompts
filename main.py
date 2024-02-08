
import argparse
import os
import pickle
import torch
import tqdm
from transformers import AutoTokenizer

from src.classifier import PromptClassifier
from src.data import load_data_wrapper, load_pararel_by_uuid
from src.metrics import MetricsWrapper
from src.models import ModelWrapper
from src.knowledge_neurons import KnowledgeNeurons
from src.plots import plot_lama_scores, plot_rela_nll_perplexity, plot_embeddings_dim_red, plot_clustering, plot_R_sq, plot_curvatures, plot_kns_surgery, plot_KNs_layer_distribution
from src.utils import process_data_to_classify

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
                        default=128)
    parser.add_argument('--prompt_method', 
                        type=int, 
                        default=1,
                        help="How the prompt will be presented to the model to compute its NLL.")
    parser.add_argument('--hidden_states',
                        action="store_true",
                        help="Compute embeddings of all layers.")
    parser.add_argument('--pooling', 
                        type=str, 
                        default='avg',
                        help="Pooling method to use when computing embeddings: 'avg' or 'sum'")
    parser.add_argument('--kns_compute',
                        action="store_true",
                        help="Compute knowledge neurons.")
    parser.add_argument('--kns_eval',
                        action="store_true",
                        help="Compute knowledge neurons surgery.")
    parser.add_argument('--kns_unmatch',
                        action="store_true",
                        help="If set to True use KNs computed on the other dataset for kns_eval.")
    parser.add_argument('--kns_overlap',
                        action="store_true",
                        help="Compute knowledge neurons overlap between ParaRel & Autoprompt.")
    
    
    ## ANALYSIS ##
    parser.add_argument('--lama_scores', 
                        action='store_true',
                        help="Compute LAMA scores.")
    parser.add_argument('--rela_nll', 
                        action='store_true',
                        help="Compute NLLs of each relation prompt.")
    parser.add_argument('--rela_perplexity', 
                        action='store_true',
                        help="Compute perplexity of each relation prompt.")
    parser.add_argument('--embedds_analysis',
                        action="store_true",
                        help="Compute prompts embeddings and various metrics on them.")
    parser.add_argument('--curvatures',
                        action="store_true",
                        help="Compute curvatures of prompts embeddings trajectory.")
    parser.add_argument('--knowledge_neurons',
                        action="store_true",
                        help="Deal with knowledge neurons.")
    
    
    ## CLASSIFICATION ##
    parser.add_argument('--prompt_cls', 
                        action='store_true',
                        help="Compute proportion of prompts in model vocab.")
    parser.add_argument('--cls', 
                    type=str, 
                    default='last_nll',
                    help="Classifier to use.")
    parser.add_argument('--data_to_classify', 
                        type=str, 
                        default='prompt-random',
                        help="What to distinguish.")
    
    # TEMP FLAG: used to debug (!) code by loading partially datasets
    parser.add_argument('--debug',
                        action="store_true")
    args = parser.parse_args()
    
    
    if not('bert' in args.model_name):
        raise Exception('Be careful mate, there is a lot of mask_token, sep_token references \
                        which will be a pain to solve!')
        
    if args.knowledge_neurons:
        print("WARNING: You are computing Knowledge Neurons hence only this will be done.")
        print("         If you want to compute other metrics please set knowledge_neurons")
        print("         to False.")
    
    ### Device, Model, Tokenizer ###
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    
    model = ModelWrapper(
                    model_type = args.model_type,
                    model_name = args.model_name,
                    device = device,
                    output_hidden_states = not(args.knowledge_neurons)
                    )
    
    ### Load Dataset(s) ###
    
    datasets_to_load = args.datasets.split('-')
    if 'lama' in datasets_to_load:
        lama_name = 'lama'
    elif 'mlama' in datasets_to_load:
        lama_name = 'mlama'
    elif 'pararel' in datasets_to_load:
        lama_name = 'pararel'
    else:
        lama_name = None
    
    langs_to_load = args.langs_to_load.split('-')
    if langs_to_load == ['']:
        langs_to_load = []
        
    if not(args.knowledge_neurons):
        # under the assumption that we load autoprompts, useless otherwise
        seeds = [int(seed) for seed in args.seeds.split('-')]
            
        datasets = load_data_wrapper(
                                datasets_to_load = datasets_to_load,
                                tokenizer = tokenizer,
                                n_tokens = args.n_tokens,
                                n_random_prompts = args.n_random_prompts,
                                seeds = seeds,
                                langs_to_load = langs_to_load,
                                model_type = args.model_type,
                                debug_flag = args.debug,
                                )
        
    ### Knowledge Neurons ###
    
    if args.knowledge_neurons:
        
        assert len(datasets_to_load) == 1
        assert datasets_to_load[0] in ['autoprompt', 'pararel']
        
        # Load data
        dataset = load_pararel_by_uuid(
                            tokenizer = tokenizer,
                            remove_multiple_tokens = True,
                            autoprompt = (datasets_to_load[0] == 'autoprompt'),
                            lower = True
                            )
        
        kns_path_raw = os.path.join('results', 'knowledge_neurons', args.model_name)
        if datasets_to_load[0] == 'pararel':
            kns_path = os.path.join(kns_path_raw, 'pararel')
        elif datasets_to_load[0] == 'autoprompt':
            kns_path = os.path.join(kns_path_raw, 'autoprompt')
        os.makedirs(kns_path, exist_ok=True)
        
        kn = KnowledgeNeurons(
            model = model.model,
            tokenizer = tokenizer,
            data = dataset,
            device = device,
            kns_path = kns_path
        )
        
        if args.kns_compute:
            kn.compute_knowledge_neurons()
        
        if args.kns_eval:
            relative_probs = kn.knowledge_neurons_surgery(kns_match = not(args.kns_unmatch))
            
            plot_kns_surgery(relative_probs, kns_path, kns_match = not(args.kns_unmatch))
            
        if args.kns_overlap:
            layer_kns_pararel, layer_kns_autoprompt, layer_overlap_kns = kn.compute_overlap()
            
            plot_KNs_layer_distribution(
                                layer_kns_pararel, 
                                num_layers = model.model.config.num_hidden_layers,
                                dataset = 'pararel',
                                overlap = False,
                                kns_path = kns_path_raw
                                )
            plot_KNs_layer_distribution(
                                layer_kns_autoprompt, 
                                num_layers = model.model.config.num_hidden_layers,
                                dataset = 'autoprompt',
                                overlap = False,
                                kns_path = kns_path_raw
                                )
            
            plot_KNs_layer_distribution(
                                layer_overlap_kns, 
                                num_layers = model.model.config.num_hidden_layers,
                                overlap = True,
                                kns_path = kns_path_raw
                                )
        
    
    ### Compute Metrics ###
    
    metrics = MetricsWrapper(
                    model = model,
                    tokenizer = tokenizer,
                    device = device
                    )
    
    if args.lama_scores:
        
        lama_scores_res_dict = metrics.compute_lama_scores(
                                            datasets = datasets,
                                            lama_name = lama_name,
                                            seeds = seeds,
                                            batch_size = args.batch_size
                                            )
        
        
    if args.embedds_analysis:
        
        embedds_analysis_res_dict = metrics.compute_embeddings_analysis(
                                            datasets=datasets,
                                            lama_name = lama_name,
                                            seeds = seeds,
                                            method = args.prompt_method,
                                            batch_size = args.batch_size,
                                            pooling = args.pooling,
                                            hidden_states = args.hidden_states,
                                            subset_size = 1000
                                        )
        
    if args.curvatures:
        
        curvatures_res_dict = metrics.compute_curvatures(
                                            datasets=datasets,
                                            lama_name = lama_name,
                                            seeds = seeds,
                                            batch_size = args.batch_size,
                                        )
        
    
    
    if args.rela_nll or args.rela_perplexity:
        
        nlls_perplexities_res_dict = metrics.compute_nlls_perplexities(
                                            datasets = datasets,
                                            seeds = seeds,
                                            method = args.prompt_method,
                                            batch_size = args.batch_size
                                            )
            
    if args.prompt_cls:
             
        # Init Classifier
        cls = PromptClassifier(
                            model = model,
                            model_name = args.model_name,
                            tokenizer = tokenizer,
                            metrics = metrics, 
                            device = device,
                            cls = args.cls,
                            data_to_classify = process_data_to_classify(args.data_to_classify, seeds, lama_name),
                            batch_size = args.batch_size,
                            seeds = seeds,
                            prompt_method = args.prompt_method,
                            pooling = args.pooling
                            ) 
        
        # Preprocess
        train_dataset = cls.preprocess(datasets)
        
        # Train
        cls.train(train_dataset)
        
        # Compute Optimal Threshold
        cls.chose_best_threshold(train_dataset)
        
        
    ### Plot & Save Results ###
    
    if args.lama_scores:
        
        os.makedirs(os.path.join('results', "lama_scores"), exist_ok=True)
        
        scores_name = f'{args.model_name}_scores_'+ '{}.pickle'
        scores_by_rela_name = f'{args.model_name}_scores_by_rela_' +'{}.pickle'
        for dataset_name in lama_scores_res_dict.keys():
            with open(os.path.join('results', 'lama_scores', scores_name.format(dataset_name)), 'wb') as f:
                pickle.dump(lama_scores_res_dict[dataset_name]['scores'], f)
            with open(os.path.join('results', 'lama_scores', scores_by_rela_name.format(dataset_name)), 'wb') as f:
                pickle.dump(lama_scores_res_dict[dataset_name]['scores_by_rela'], f)
        
        
        plot_lama_scores(
            lama_scores = lama_scores_res_dict,
            model_name = args.model_name,
            seeds = seeds,
            lama_name = lama_name
        )
        
    if args.rela_nll:
        
        os.makedirs(os.path.join('results', "nll", f"method_{args.prompt_method}"), exist_ok=True)
        
        with open(os.path.join('results', "nll", f"method_{args.prompt_method}", f'{args.model_name}_nlls_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(nlls_perplexities_res_dict['nlls'], f)
        with open(os.path.join('results', "nll", f"method_{args.prompt_method}", f'{args.model_name}_tokens_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(nlls_perplexities_res_dict['tokens'], f)
            
        relations_ids = list(set(datasets[lama_name]['predicate_id']))
        
        for seed in seeds:
            for rela in relations_ids:
                plot_rela_nll_perplexity(
                    nlls_perplexities_res_dict['nlls'][rela],
                    nlls_perplexities_res_dict['tokens'][rela],
                    method = args.prompt_method,
                    model_name = args.model_name,
                    rela = rela,
                    seed = seed,
                    perplexity = False
                )
                
    if args.rela_perplexity:
        
        os.makedirs(os.path.join('results', "perplexity", f"method_{args.prompt_method}"), exist_ok=True)
        
        with open(os.path.join('results', "perplexity", f"method_{args.prompt_method}", f'{args.model_name}_perplexities_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(nlls_perplexities_res_dict['perplexities'], f)
        with open(os.path.join('results', "perplexity", f"method_{args.prompt_method}", f'{args.model_name}_tokens_method_{args.prompt_method}.pickle'), 'wb') as f:
            pickle.dump(nlls_perplexities_res_dict['tokens'], f)
        
        for seed in seeds:
            for rela in relations_ids:
                plot_rela_nll_perplexity(
                    nlls_perplexities_res_dict['perplexities'][rela],
                    nlls_perplexities_res_dict['tokens'][rela],
                    method = args.prompt_method,
                    model_name = args.model_name,
                    rela = rela,
                    seed = seed,
                    perplexity = True
                )
                
    if args.curvatures:
        
        curvatures_path = os.path.join('results', "curvatures")
        os.makedirs(curvatures_path, exist_ok=True)
        
        plot_curvatures(
            curvatures_res_dict,
            model_name = args.model_name,
            dir_path = curvatures_path
            )
                
    if args.embedds_analysis:
        
        pca_path = os.path.join('results', "embeddings_analysis", "PCA", f"{args.model_name}-{args.prompt_method}-{args.pooling}")
        mds_path = os.path.join('results', "embeddings_analysis", "MDS", f"{args.model_name}-{args.prompt_method}-{args.pooling}")
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(mds_path, exist_ok=True)
        
        data_names = [lama_name] + \
                     [f'autoprompt_seed{seed}' for seed in seeds] + \
                     ['random'] # Not that clean because we should use the data_to_load param but you know..
        
        # Save PCA
        
        if args.hidden_states:
            for l in range(embedds_analysis_res_dict[f'{data_names[0]}-{data_names[1]}']['num_layers']):
                for i in range(len(data_names)):
                    for j in range(i+1, len(data_names)):
                        plot_embeddings_dim_red(
                                        embeds1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}'][f'pca embeds1 layer{l}'],
                                        embeds2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}'][f'pca embeds2 layer{l}'],
                                        embeds3 = None,
                                        label1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label1'],
                                        label2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label2'],
                                        label3 = None,
                                        annots1 = [], # Supposed to be relations ids but it's hard for nothing big so anyway
                                        annots2 = [], # Same
                                        annots3 = None,
                                        dim_red_name = 'PCA',
                                        dir_path = pca_path,
                                        layer_num=l
                                        )
                        plot_embeddings_dim_red(
                                        embeds1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}'][f'mds embeds1 layer{l}'],
                                        embeds2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}'][f'mds embeds2 layer{l}'],
                                        embeds3 = None,
                                        label1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label1'],
                                        label2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label2'],
                                        label3 = None,
                                        annots1 = [], # Supposed to be relations ids but it's hard for nothing big so anyway
                                        annots2 = [], # Same
                                        annots3 = None,
                                        dim_red_name = 'MDS',
                                        dir_path = mds_path,
                                        layer_num=l
                                        )
                        
            for i in range(len(data_names)):
                    for j in range(i+1, len(data_names)):
                        plot_clustering(
                            res_dict = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}'], 
                            dir_path = pca_path,
                            dim_red_name = 'pca'
                            )
                        plot_R_sq(
                            res_dict = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}'], 
                            dir_path = pca_path,
                            dim_red_name = 'pca'
                            )
                        plot_clustering(
                            res_dict = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}'], 
                            dir_path = mds_path,
                            dim_red_name = 'mds'
                            )
                        
        else:
            for i in range(len(data_names)):
                for j in range(i+1, len(data_names)):
                    plot_embeddings_dim_red(
                                    embeds1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['pca embeds1'],
                                    embeds2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['pca embeds2'],
                                    embeds3 = None,
                                    label1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label1'],
                                    label2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label2'],
                                    label3 = None,
                                    annots1 = [], # Supposed to be relations ids but it's hard for nothing big so anyway
                                    annots2 = [], # Same
                                    annots3 = None,
                                    dim_red_name = 'PCA',
                                    dir_path = pca_path,
                                    )
                    plot_embeddings_dim_red(
                                    embeds1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['mds embeds1'],
                                    embeds2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['mds embeds2'],
                                    embeds3 = None,
                                    label1 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label1'],
                                    label2 = embedds_analysis_res_dict[f'{data_names[i]}-{data_names[j]}']['label2'],
                                    label3 = None,
                                    annots1 = [], # Supposed to be relations ids but it's hard for nothing big so anyway
                                    annots2 = [], # Same
                                    annots3 = None,
                                    dim_red_name = 'MDS',
                                    dir_path = mds_path,
                                    )
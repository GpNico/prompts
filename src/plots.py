
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.patches as mpatches
import numpy as np
import os
import torch
from typing import Tuple, Dict, List



def plot_lama_scores(lama_scores: dict,
                     **kwargs) -> None:
    """
        Args:
            lama_scores (dict) keys: dataset_name values: dict    keys: scores, _   values: {'P@1': 0.26,
                                                                                            'P@5': 0.45,
                                                                                            'P@20': 0.62,
                                                                                            'P@100': 0.79}
    """
    fig, ax = plt.subplots(1,1,figsize = (8, 4)) 
    
    
    for dataset_name in lama_scores.keys():
        xs = [int(k[2:]) for k in lama_scores[dataset_name]['scores'].keys()]
        ax.plot(xs, lama_scores[dataset_name]['scores'].values(), linestyle = '--', marker = 'x', label = dataset_name)

    ax.set_ylim([0,1])
    ax.legend()

    ax.grid(True)
    ax.set_xscale('log')

    ax.set_xlabel('k')
    ax.set_ylabel('P@k')
    ax.set_title(f"{kwargs['model_name']} P@k for LAMA & AutoPrompt")
    
    fig_name = f"{kwargs['model_name']}_"
    for dataset_name in lama_scores.keys():
        fig_name += dataset_name + '_'
    fig_name += "scores.png"

    plt.savefig(
        os.path.join(
            "results",
            'lama_scores',
            fig_name
        )
    )
    plt.close()
    
def plot_rela_nll_perplexity(rela_scores: dict, 
                             rela_tokens: dict,
                             method: int,
                             **kwargs):
    """
    
    """
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax2 = ax.twiny()

    
    # LAMA
    dataset_rela_scores_err = rela_scores['lama'].std(axis = 0)/np.sqrt(rela_scores['lama'].shape[0])

    x_dataset = np.arange(rela_scores['lama'].shape[1])
    l2 = ax2.plot(
            x_dataset, 
            rela_scores['lama'].mean(axis = 0), 
            marker = 'x', 
            color = 'tab:orange', 
            label = 'LAMA')
    ax2.fill_between(
                x_dataset, 
                rela_scores['lama'].mean(axis = 0) - dataset_rela_scores_err, 
                rela_scores['lama'].mean(axis = 0) + dataset_rela_scores_err,
                color = 'tab:orange',
                alpha = 0.3
                )
    
    # AutoPrompt
    autoprompts_rela_scores_err = rela_scores[f"autoprompt_seed{kwargs['seed']}"].std(axis = 0)/np.sqrt(rela_scores[f"autoprompt_seed{kwargs['seed']}"].shape[0])

    x_autoprompts = np.arange(rela_scores[f"autoprompt_seed{kwargs['seed']}"].shape[1])
    l1 = ax.plot(
            x_autoprompts, 
            rela_scores[f"autoprompt_seed{kwargs['seed']}"].mean(axis = 0), 
            marker = 'x', 
            label = 'AutoPrompt')
    ax.fill_between(
                x_autoprompts, 
                rela_scores[f"autoprompt_seed{kwargs['seed']}"].mean(axis = 0) - autoprompts_rela_scores_err, 
                rela_scores[f"autoprompt_seed{kwargs['seed']}"].mean(axis = 0) + autoprompts_rela_scores_err,
                alpha = 0.3
                )

    # Random
    rd_rela_scores_err = rela_scores['random'].std(axis = 0)/np.sqrt(rela_scores['random'].shape[0])

    l3 = ax.plot(
            x_autoprompts, 
            rela_scores['random'].mean(axis = 0), 
            marker = 'x', 
            color = 'tab:green', 
            label = 'random')
    ax.fill_between(
                x_autoprompts, 
                rela_scores['random'].mean(axis = 0) - rd_rela_scores_err, 
                rela_scores['random'].mean(axis = 0) + rd_rela_scores_err,
                color = 'tab:green',
                alpha = 0.3)
    
    name = 'nll'
    if kwargs['perplexity']:
        name = 'perplexity'


    pos_x = rela_tokens[f"autoprompt_seed{kwargs['seed']}"]['pos_x']
    pos_y = rela_tokens[f"autoprompt_seed{kwargs['seed']}"]['pos_y']
    x_autoprompts = list(x_autoprompts)
    
    if method in [3,4]:
        x_autoprompts =  x_autoprompts[:pos_x] + [pos_x - 0.5] + x_autoprompts[pos_x:]
    elif method in [1,2]:
        x_autoprompts = x_autoprompts[:pos_x] + [pos_x - 0.5] + x_autoprompts[pos_x:pos_y - 1] + [pos_y - 1.5] + x_autoprompts[pos_y - 1:]
    
    ax.set_xticks(x_autoprompts, rela_tokens[f"autoprompt_seed{kwargs['seed']}"]['tokens'])
    colors = ['tab:blue']*len(x_autoprompts)
    colors[pos_x] = 'blue'
    colors[pos_y] = 'red'
    
    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)

    pos_x = rela_tokens['lama']['pos_x']
    pos_y = rela_tokens['lama']['pos_y']
    x_dataset = list(x_dataset)
    
    if method in [3,4]:
        x_dataset = x_dataset[:pos_x] + [pos_x - 0.5] + x_dataset[pos_x:]
    elif method in [1,2]:
        x_dataset = x_dataset[:pos_x] + [pos_x - 0.5] + x_dataset[pos_x:pos_y - 1] + [pos_y - 1.5] + x_dataset[pos_y - 1:]
        
    ax2.set_xticks(x_dataset, rela_tokens['lama']['tokens'])
    colors = ['tab:orange']*len(x_dataset)
    colors[pos_x] = 'blue'
    colors[pos_y] = 'red'
        
    for xtick, color in zip(ax2.get_xticklabels(), colors):
        xtick.set_color(color)

    ax.set_xlabel('token')
    ax.set_ylabel(name.upper())

    plt.title(f"{name.upper()} of {kwargs['model_name']} on {kwargs['rela']}")

    ls = l1 + l2 + l3
    labs = [l.get_label() for l in ls] 
    ax.legend(ls, labs)

    plt.savefig(
        os.path.join(
            "results",
            name,
            f"method_{method}",
            f"{kwargs['model_name']}_{kwargs['rela']}_{name}_{kwargs['seed']}.png"
        )
    )
    plt.close()
    
    
def plot_embeddings_dim_red(embeds1, 
                            embeds2,
                            embeds3 = None,
                            label1: str = '1',
                            label2: str = '2',
                            label3: str = '3',
                            annots1: list = [],
                            annots2: list = [],
                            annots3: list = [],
                            dim_red_name: str = "PCA",
                            layer_num: int = None,
                            **kwargs
                            ):
    fig = plt.figure(figsize = (15,10))

    plt.scatter(
        embeds1[:, 0],
        embeds1[:, 1],
        label = label1
        )

    if len(annots1) > 0:
        for k, annot in enumerate(annots1):
            plt.annotate(
                annot,
                xy = (embeds1[k,0],
                    embeds1[k,1])
                )

    plt.scatter(
        embeds2[:, 0],
        embeds2[:, 1],
        label = label2
        )

    if len(annots2) > 0:
        for k, annot in enumerate(annots2):
            plt.annotate(
                annot,
                xy = (embeds2[k,0],
                    embeds2[k,1])
                )
        
    if embeds3 is not None:
        plt.scatter(
            embeds3[:, 0],
            embeds3[:, 1],
            label = label3
            )
        
        if len(annots3) > 0:
            for k, annot in enumerate(annots3):
                plt.annotate(
                    annot,
                    xy = (embeds3[k,0],
                        embeds3[k,1])
                    )

    if layer_num is None:
        plt.title(f"{label1} Prompts vs {label2} Prompts Embeddings {dim_red_name.upper()}")
    else:
        plt.title(f"{label1} Prompts vs {label2} Prompts Embeddings {dim_red_name.upper()} - Layer {layer_num}")
    plt.xlabel(f'{dim_red_name.upper()} 1')
    plt.ylabel(f'{dim_red_name.upper()} 2')
    plt.legend()
    
    if layer_num is None:
        plt.savefig(
            os.path.join(
                kwargs['dir_path'],
                f"{label1}_vs_{label2}.png"
            )
        )
    else:
        plt.savefig(
            os.path.join(
                kwargs['dir_path'],
                f"{label1}_vs_{label2}_layer_{layer_num}.png"
            )
        )
    plt.close()
    
def plot_clustering(res_dict: dict, 
                    **kwargs):
    """
        Plot the clustering metrics (homogeneity, completeness, etc.) for each layer of the model.
        
        Args:
            res_dict (dict) res dict of the considered pair.
                            names of the pair are under the keys 'label1' 'label2'
                            clustering metrics are under:
                                'pca kmeans completeness'
                                'pca kmeans homogeneity'
                                'pca spectral completeness'
                                'pca spectral homogeneity'

    """
    
    layers = [k for k in range(res_dict['num_layers'])]
    kmeans_completeness = [res_dict[f'{kwargs["dim_red_name"]} kmeans completeness layer{l}'] for l in range(res_dict['num_layers'])]
    kmeans_homogeneity = [res_dict[f'{kwargs["dim_red_name"]} kmeans homogeneity layer{l}'] for l in range(res_dict['num_layers'])]
    spectral_completeness = [res_dict[f'{kwargs["dim_red_name"]} spectral completeness layer{l}'] for l in range(res_dict['num_layers'])]
    spectral_homogeneity = [res_dict[f'{kwargs["dim_red_name"]} spectral homogeneity layer{l}'] for l in range(res_dict['num_layers'])]
    
    fig, axs = plt.subplots(1, 2 , figsize = (20,10))

    axs[0].plot(
        layers,
        kmeans_completeness,
        marker = '+',
        label = 'completeness',
        linestyle = '--'
        )
    axs[0].plot(
        layers,
        kmeans_homogeneity,
        marker = 'x',
        label = 'homogeneity',
        linestyle = '--'
        )
    axs[1].plot(
        layers,
        spectral_completeness,
        marker = '+',
        label = 'completeness',
        linestyle = '--'
        )
    axs[1].plot(
        layers,
        spectral_homogeneity,
        marker = 'x',
        label = 'homogeneity',
        linestyle = '--'
        )

    axs[0].set_title(f"K-Means")
    axs[0].set_xlabel(f'Model Layer')
    axs[0].set_ylabel(f'Clustering Metrics')
    axs[0].set_ylim([0,1])
    axs[0].legend()
    
    axs[1].set_title(f"Spectral")
    axs[1].set_xlabel(f'Model Layer')
    axs[1].set_ylabel(f'Clustering Metrics')
    axs[1].set_ylim([0,1])
    axs[1].legend()
    
    plt.suptitle(f"{res_dict['label1']} Prompts vs {res_dict['label2']} Prompts Embeddings {kwargs['dim_red_name'].upper()} Clustering Metrics")

    plt.savefig(
        os.path.join(
            kwargs['dir_path'],
            f"{res_dict['label1']}_vs_{res_dict['label2']}_{kwargs['dim_red_name']}_clustering_metrics.png"
        )
    )

    plt.close()
    
def plot_R_sq(res_dict: dict, 
              **kwargs):
    """
        Plot the R² of the linear regressions between embedds1 and embedds2 (after
        dim reduction).
        
        Args:
            res_dict (dict) res dict of the considered pair.
                            names of the pair are under the keys 'label1' 'label2'
                            R² is under:
                                'pca R2'

    """
    
    layers = [k for k in range(res_dict['num_layers'])]
    R_sq_s = [res_dict[f'{kwargs["dim_red_name"]} R2 layer{l}'] for l in range(res_dict['num_layers'])]
    
    fig = plt.figure(figsize = (15,10))

    plt.plot(
        layers,
        R_sq_s,
        marker = '+',
        label = 'R²',
        linestyle = '--'
        )
    
    plt.title(f"Spectral")
    plt.xlabel(f'Model Layer')
    plt.ylabel(f'Clustering Metrics')
    plt.ylim([0,1])
    plt.legend()
    
    plt.suptitle(f"{res_dict['label1']} Prompts vs {res_dict['label2']} Prompts Embeddings {kwargs['dim_red_name'].upper()} R²")

    plt.savefig(
        os.path.join(
            kwargs['dir_path'],
            f"{res_dict['label1']}_vs_{res_dict['label2']}_{kwargs['dim_red_name']}_R_sq.png"
        )
    )

    plt.close()
    

def plot_curvatures(res_dict: dict,
                    **kwargs) -> None:
    """
    
    
    Args:
        res_dict (dict) keys: dataset names values: (tensor) shape [Dataset Size, Num Layers]
    
    """

    for dataset_name, curvatures in res_dict.items():
        layers = np.arange(curvatures.shape[1])
        # Put it in deg
        curvatures = curvatures*(180/np.pi)
        # Plot
        curvatures_change = curvatures - curvatures[:, 0][:,None]
        mean_scores = torch.mean(curvatures_change, dim=0)
        std_dev_scores = torch.std(curvatures_change, dim=0)
        plt.errorbar(layers[:curvatures.shape[1]], mean_scores, yerr=std_dev_scores, fmt='o-', label=dataset_name)

    plt.title(f'Curvatures Change for {kwargs["model_name"]} Along Machine & Human Prompts')
    plt.xlabel('Model Layer')
    plt.ylabel('Curvature Change (deg)')
    plt.legend()
    
    plt.savefig(
        os.path.join(
            kwargs['dir_path'],
            f"curvatures_{kwargs['model_name']}.png"
        )
    )

    plt.close()
    
    
def plot_kns_surgery(relative_probs: Dict[str, Dict[str, float]], 
                     kns_path: str,
                     kns_match: bool = True) -> None:
    rela_names = list(relative_probs['wo_kns'].keys())

    # Params
    n_k = 1
    n_relas = len(rela_names)
    n_bars_per_rela = n_k * 2
    colors = ['cornflowerblue', 'navy']

    fig = plt.figure(figsize = (0.5*n_relas, 5))
    for i in range(n_relas):
        width = 1/(n_bars_per_rela + 1)
        plt.bar(i + 1*width + width/2, relative_probs['wo_kns'][rela_names[i]], width = width, color=colors[0])
        plt.bar(i + 2*width + width/2, relative_probs['db_kns'][rela_names[i]], width = width, color=colors[1])

    plt.hlines(0, xmin=0, xmax=n_relas, color='black')
    plt.xlim((0,n_relas))
    #plt.ylim((-5,5))
    plt.xticks(np.arange(n_relas) + 0.5, rela_names)

    # Legend
    wo_patch = mpatches.Patch(facecolor=colors[0], label='w/o KNs')
    db_patch = mpatches.Patch(facecolor=colors[1], label='db KNs')
    plt.legend(handles=[wo_patch, db_patch])
    if kns_match:
        plt.title('Relative Probs - Without KNs, Doubling KNs')
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery.png"
            )
        )
    else:
        plt.title('Relative Probs - Without KNs, Doubling KNs - UnMatched KNs')
        plt.savefig(
            os.path.join(
                kns_path,
                f"kns_surgery_unmatched.png"
            )
        )
    plt.show()
    plt.close()
    
    
def plot_KNs_layer_distribution(layer_kns: Dict[str,int], **kwargs) -> None:

    num_neurons = 3072
    layer_scores = [0]*kwargs['num_layers']
    for layer, count in layer_kns.items():
        layer_scores[layer] += count

    layer_scores = [s*100/num_neurons for s in layer_scores]

    plt.bar(np.arange(len(layer_scores)), layer_scores)
    plt.xlabel('Layer')
    plt.ylabel('Percentage')
    if kwargs['overlap']:
        plt.title(f'Knowledge Neurons Overlap Layer Distribution')
        plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                f"kns_overlap_layer_distrib.png"
            )
        )
    else:
        plt.title(f'Knowledge Neurons Layer Distribution ({kwargs["dataset"]})')
        plt.savefig(
            os.path.join(
                kwargs['kns_path'],
                kwargs['dataset'],
                f"kns_layer_distrib.png"
            )
        )
    plt.close()
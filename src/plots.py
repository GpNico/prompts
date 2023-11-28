
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os



def plot_lama_scores(lama_scores_lama: dict,
                     lama_scores_autoprompt: dict,
                     lama_scores_random: dict,
                     **kwargs) -> None:
    """
        Args:
            lama_scores_lama (dict) {'P@1': 0.26,
                                     'P@5': 0.45,
                                     'P@20': 0.62,
                                     'P@100': 0.79}
            lama_scores_autoprompt (dict of dict) dict of autoprompts scores (values)
                                                  for different seeds (keys)
    """
    fig, ax = plt.subplots(1,1,figsize = (8, 4)) 

    xs = [int(k[2:]) for k in lama_scores_lama.keys()]

    ax.plot(xs, lama_scores_lama.values(), linestyle = '--', marker = 'x', label = 'LAMA')
    
    if len(lama_scores_autoprompt) > 0: 
        for seed in kwargs['seeds']:
            ax.plot(xs, 
                    lama_scores_autoprompt[seed].values(), 
                    linestyle = '--', 
                    marker = 'x', 
                    label = f'AutoPrompt (seed {seed})')
            
    ax.plot(xs, lama_scores_random.values(), linestyle = '--', marker = 'x', label = 'random')


    ax.set_ylim([0,1])
    ax.legend()

    ax.grid(True)
    ax.set_xscale('log')

    ax.set_xlabel('k')
    ax.set_ylabel('P@k')
    ax.set_title(f"{kwargs['model_name']} P@k for LAMA & AutoPrompt")
    
    if kwargs['shuffle']:
        fig_name = f"{kwargs['model_name']}_{kwargs['lama_name']}_scores_shuffled.png"
    else:
        fig_name = f"{kwargs['model_name']}_{kwargs['lama_name']}_scores.png"

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
                            **kwargs
                            ):
    fig = plt.figure(figsize = (15,10))

    plt.scatter(
        embeds1[:, 0],
        embeds1[:, 1],
        label = label1
        )

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

        for k, annot in enumerate(annots3):
            plt.annotate(
                annot,
                xy = (embeds3[k,0],
                      embeds3[k,1])
                )

    plt.title(f"{label1} Prompts vs {label2} {kwargs['model_name']}s Embeddings {dim_red_name.upper()}")
    plt.xlabel(f'{dim_red_name.upper()} 1')
    plt.ylabel(f'{dim_red_name.upper()} 2')
    plt.legend()
    
    plt.savefig(
        os.path.join(
            "results",
            "embeddings_analysis",
            f"{dim_red_name.upper()}",
            f"{kwargs['model_name']}_{label1}_vs_{label2}.png"
        )
    )
    plt.close()
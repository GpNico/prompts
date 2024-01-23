
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
import torch
from torchmetrics.classification import BinaryStatScores
import tqdm
from typing import List

from src.metrics import MetricsWrapper
import src.utils as utils
from src.embedder import Embedder

class PromptClassifier:
    
    def __init__(self, 
                 model,
                 model_name: str,
                 tokenizer,
                 metrics: MetricsWrapper,
                 device: str,
                 batch_size: int,
                 prompt_method: str,
                 cls: str,
                 data_to_classify: List[str],
                 seeds: List[int],
                 pooling: str):
        """
            Args:
                model (HuggingFace model) model used to compute NLL & Perplexity
                                          and AutoPrompt
                model_name (str) e.g. bert-base-uncased, ...
                tokenizer (HuggingFace tokenizer) 
                metrics (MetricsWrapper) used to compute usefull content for classification
                device (str) useful for NLL & Perplexity and training a linear classifier.
                cls (str) which classifier to use. Use only last nll, use NN, logistic regression, etc.
                data_to_classify (str) e.g. ['lama', 'autoprompt_seed0']
        """
        assert cls in ['last_nll', 
                       'last_perplexity',
                       'linear_nn',
                       'nlls_reg',
                       'last_nll_reg',
                       'last_layer_cluster_pca',
                       'all_layers_cluster_pca'
                       ]
        self.cls = cls
        self.data_to_classify = data_to_classify
        self.prompt_method = prompt_method
        self.seeds = seeds
        self.pooling = pooling
        
        # init threshold
        self.threshold = 0.

        # Args
        self.device = device
        self.model = model
        self.model_name = model_name
        
        self.tokenizer = tokenizer
        
        self.predScores = BinaryStatScores()
        
        self.metrics = metrics
        
        self.batch_size = batch_size
        
    
    def compute_roc_curve(self, dataset):
        
        
        thresholds = np.linspace(0,1,40)
        
        tprs, fprs = [], []
        for thresh in tqdm.tqdm(thresholds, total = len(thresholds)):
            self.threshold = thresh
            
            scores = self.evaluate(dataset = dataset)
            
            if scores['tp'] + scores['fn'] != 0:
                tpr = scores['tp']/(scores['tp'] + scores['fn'])
            else:
                tpr = 0
            if scores['fp'] + scores['tn'] != 0:
                fpr = scores['fp']/(scores['fp'] + scores['tn'])
            else:
                fpr = 0
            
            tprs.append(tpr)
            fprs.append(fpr)
            
        
        plt.plot(thresholds, thresholds, linestyle = '--', color = 'black', alpha = 0.7, label = 'random classifier')
        plt.plot(fprs, tprs, marker = 'x', label = 'model')
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.title(f"CLS {self.cls} {'-'.join(self.data_to_classify)} ROC Curve {self.model_name}")
        plt.legend()
        
        os.makedirs(os.path.join(
                "results",
                "cls",
                f"{self.cls}",
                ), exist_ok=True)
        
        plt.savefig(
            os.path.join(
                "results",
                "cls",
                f"{self.cls}",
                f"{self.model_name}_{'_'.join(self.data_to_classify)}_method_{self.prompt_method}_roc_curve.png"
            )
        )
        plt.close()
        return list(thresholds), fprs, tprs
    
    def chose_best_threshold(self, 
                             dataset,
                             tpr_coeff: float = 0.5) -> None:
        """
            Compute ROC Curve and set self.threshold
            to optimal one.
            
            The choice of a threshold depends on the importance of TPR and 
            FPR classification problem. For example, if your classifier will 
            decide which criminal suspects will receive a death sentence, 
            false positives are very bad (innocents will be killed!). Thus 
            you would choose a threshold that yields a low FPR while keeping 
            a reasonable TPR (so you actually catch some true criminals). If 
            there is no external concern about low TPR or high FPR, one option 
            is to weight them equally by choosing the threshold that maximizes: 
            TPR - FPR
            
            This is why we have tpr_coeff from 0 to 1. Equal importance for TPR and FPR
            is tpr_coeff = 0.5
            tpr_coeff * TPR - (1 - tpr_coeff) * FPR
            
            tpr_coeff = 1 -> more importance on TPR (high SENSITIVITY)
            tpr_coeff = 0 -> more importance on - FPR, we want the smallest
                             FPR possible (high SPECIFICITY) 
                             
            Args:
                dataset (dict) key: (str) '0' '1'    values: (tensor) nature depends on the self.cls
.
        """
        thresholds, fprs, tprs = self.compute_roc_curve(dataset = dataset)
        optimal_value = - np.inf
        for k, thresh in enumerate(thresholds):
            val = tpr_coeff * tprs[k] - (1 - tpr_coeff) * fprs[k]
            if optimal_value < val:
                self.threshold = thresh
                self.tpr = tprs[k]
                self.fpr = fprs[k]
                optimal_value = val
                
        print(f"Optimal Threshold: {np.round(self.threshold, 3)}; \
                TP Rate: {np.round(self.tpr, 3)}; \
                FP Rate: {np.round(self.fpr, 3)}")
        
    def compute_prompt_proportion(self, n_tokens: int, n_eval: int) -> None:
        """
            We sample prompts of size n_tokens
            and run them through the classifier.
            This will give a proportion of prompts.
            
            Args:
                n_tokens (int)
                n_eval (int) number of prompts we will evaluate
        """
        
        def promptIdsGen(max = 0):
            n = 0
            while n < max:
                ids = np.random.choice(
                            self.tokenizer.vocab_size, # I don't like that we can't skip special_tokens
                            (self.batch_size,n_tokens), 
                            replace = True
                            )
                yield torch.tensor(
                        np.concatenate(
                            (self.tokenizer.cls_token_id * np.ones(self.batch_size)[:,None], 
                            ids, 
                            self.tokenizer.sep_token_id * np.ones(self.batch_size)[:,None]), 
                            axis = 1
                            ),
                        dtype = torch.long
                        )
                n+=1
                
        num_prompts = 0.
        num_total_eval = 0.
        for prompt_tok in tqdm.tqdm(promptIdsGen(max = n_eval), total = n_eval):
            preds = self.predict(prompt_tok)
            num_prompts += preds.sum().item()
            num_total_eval += preds.shape[0] # a priori useless as the total will be n_eval*batch_size
                                             # but you know...
            
        print(f"Classified {num_total_eval} Random Tokens Sequences.")
        print(f"TP Rate: {np.round(self.tpr, 3)}; \
                FP Rate: {np.round(self.fpr, 3)}")
        print("Prompt Proportion Found: ", np.round(num_prompts/num_total_eval, 4))
        
    def preprocess(self, datasets: dict) -> dict:
        """
            Preprocess data before training.
            This preprocessing depends on the classification method and is stand-alone
            here, which means that we do not assume prior processing.
        
            Args:
                datasets (dict of DataFrames)
        """
        
        if self.cls in ['last_nll', 
                       'last_perplexity',
                       'linear_nn_nll',
                       'nlls_reg',
                       'last_nll_reg']:
            # We need to compute nlls and/or perplexities
            raw_res = self.metrics.compute_nlls_perplexities(
                                            datasets = {d: datasets[d] for d in self.data_to_classify},
                                            batch_size = self.batch_size,
                                            method = self.prompt_method,
                                            seeds = self.seeds
                                            )
            nlls_perplexities = utils.collapse_metrics_nlls_perplexities(raw_res, self.data_to_classify)
            # Rq: The nlls do not contain [X], nor [CLS] nor [SEP]
            
            if self.cls in ['linear_nn_nll',
                            'nlls_reg']:
                # Here we need full nlls
                # Rq: the nlls do not contain the [X] values
                return {'0': nlls_perplexities[self.data_to_classify[0]][0],
                        '1': nlls_perplexities[self.data_to_classify[1]][0]}
            elif self.cls in ['last_nll', 
                              'last_nll_reg']:
                # Here we only need last nll
                # /!\ due to padding there's a trick to use /!\
                # Rq: this code works because we padded every tensor by at least 1 zero
                train_dataset = {}
                for k in range(2):
                    row_idx, col_idx = torch.where(nlls_perplexities[self.data_to_classify[k]][0] == 0)
                    _, unique_idx = np.unique(row_idx, return_index=True)
                    row_idx = row_idx[unique_idx]
                    col_idx = col_idx[unique_idx]
                    train_dataset[str(k)] = nlls_perplexities[self.data_to_classify[k]][0][row_idx, col_idx - 1] # get only the last token NLL
                return train_dataset
            elif self.cls in ['last_perplexity']:
                # Here we only need last nll
                # /!\ due to padding there's a trick to use /!\
                # Rq: this code works because we padded every tensor by at least 1 zero
                train_dataset = {}
                for k in range(2):
                    row_idx, col_idx = torch.where(nlls_perplexities[self.data_to_classify[k]][1] == 0)
                    _, unique_idx = np.unique(row_idx, return_index=True)
                    row_idx = row_idx[unique_idx]
                    col_idx = col_idx[unique_idx]
                    train_dataset[str(k)] = nlls_perplexities[self.data_to_classify[k]][1][row_idx, col_idx - 1] # get only the last token NLL
                return train_dataset
        elif self.cls in ['last_layer_cluster_pca',
                          'all_layers_cluster_pca']:
            embedder = Embedder(
                    model=self.model,
                    tokenizer=self.metrics.tokenizer # /!\ We use the wrapper
                    )
            
            # Get wheter data_to_classify are token_sequence or not
            tokens_sequence_bools = utils.token_sequence_or_not(self.data_to_classify)
        
            if self.pooling == 'mask':
                assert self.prompt_method == 3
                
            print('Dataset size ', len(datasets[self.data_to_classify[0]]))
                
            # Embedds Dataset 0
            embedds_0 = embedder.embed(
                                    df = datasets[self.data_to_classify[0]],
                                    method = self.prompt_method,
                                    pooling= self.pooling,
                                    output_hidden_states= (self.cls == 'all_layers_cluster_pca'),
                                    batch_size = self.batch_size,
                                    token_sequence = tokens_sequence_bools[self.data_to_classify[0]]
                                    )
            # Embedds Dataset 0
            embedds_1 = embedder.embed(
                                    df = datasets[self.data_to_classify[1]],
                                    method = self.prompt_method,
                                    pooling= self.pooling,
                                    output_hidden_states= (self.cls == 'all_layers_cluster_pca'),
                                    batch_size = self.batch_size,
                                    token_sequence = tokens_sequence_bools[self.data_to_classify[1]]
                                    )
            
            # Compute PCA
            full_embedds = np.concatenate((embedds_0.numpy(), embedds_1.numpy()), axis = 0)
            
            pca = PCA(n_components=2)
            embeds_pca = pca.fit_transform(full_embedds) # Shape [2*(N prompts), 2]
            
            return {'0': embeds_pca[:embedds_0.shape[0]], 
                    '1': embeds_pca[embedds_0.shape[0]:]}
        
        return
    
    def train(self, dataset: dict) -> None:
        """
            Args:
                dataset (dict) key: (str) '0' '1'    values: (tensor) nature depends on the self.cls
            
        """
        
        if self.cls in ['last_nll', 'last_perplexity']:
            # So we select Dataset 0 to compute the last_nll_mean
            # dataset values here is a 1d tensor containing last_nll 
            self.last_nll_mean = dataset['0'].mean().item()
            print("\tlast_nll_mean ", self.last_nll_mean)
            return
        elif self.cls in ['nlls_reg', 'last_nll_reg']:
            
            self.reg = LogisticRegression()
                
            nlls_0 = dataset['0'] # shape [Dataset Size, L0] (nlls_reg), [Dataset Sie] (last_nll_reg)
            nlls_1 = dataset['1'] # shape [Dataset Size, L1] (nlls_reg), [Dataset Sie] (last_nll_reg)
            
            # No need to pad as L1 = L0 (cf. utils.collaps_metrics_nlls_perplexities)
            X = torch.cat((nlls_0, nlls_1)) # Shape [2*Dataset Size, L] (nlls_reg), Shape [2*Dataset Size] (last_nll_reg)
            y = torch.cat((torch.zeros(nlls_0.shape[0]), 
                           torch.ones(nlls_1.shape[0])))
            
            self.reg = self.reg.fit( X.numpy(), y.numpy() )
            print("Logistic Regression Training R^2: ", self.reg.score(X,y))
        elif self.cls in ['last_layer_cluster_pca']:
            self.pca_kmeans = KMeans(n_clusters = 2, n_init='auto').fit(
                                                np.concatenate(
                                                        (dataset['0'], 
                                                         dataset['1']), 
                                                        axis = 0
                                                        )
                                                )
        
        return
    
    def evaluate(self, dataset: dict) -> dict:
        """
            Compute True Positive, False Positive, True Negative, False Negative for dataset.
        
            Args:
                dataset (dict) key: (str) '0' '1'    values: (tensor) nature depends on the self.cls
            Returns:
                (dict) keys: (str) tp, fp, tn, fn
            
        """
        
        if self.cls in ['last_nll', 'last_perplexity', 'nlls_reg', 'last_nll_reg', 'last_layer_cluster_pca']:
            # Dataset 0
            preds0 = self.predict(x = dataset['0']) # shape (DS)
            targets0 = torch.zeros_like(preds0)
            
            # Dataset 1
            preds1 = self.predict(x = dataset['1']) # shape (DS)
            targets1 = torch.ones_like(preds1)
            
            # Cat
            preds =  torch.cat((preds0, preds1))
            targets = torch.cat((targets0, targets1))
            
            # metrics
            res = self.predScores(preds, targets)
            return {"tp": res[0].item(), "fp": res[1].item(), "tn": res[2].item(), "fn": res[3].item()}
    
    def predict(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
            
            Args:
                x (torch.Tensor) inputs can be a lot of things:
                                    if last_nll: 1d tensor containing last_nll
                                    if last_perplexity: 1d tensor containing last_perplexity
            Returns:
                preds (torch.Tensor) 0 if Dataset 0, 1 if Dataset 1
        """
        
        if self.cls in ['last_nll', 'last_perplexity']:
            # Dataset 0 has been used to compute self.last_nll_mean
            preds = torch.zeros(x.shape[0])
            if ('lama' in self.data_to_classify[0]) or ('random' == self.data_to_classify[1]):
                # If lama is dataset 0 we classify as 1 sequence with last_nll above
                # Same goes if random is dataset 1
                preds[torch.where( x > 3*(1 - self.threshold)*self.last_nll_mean )[0]] = 1 # We put a 3* to allow to go 3x above lama last_nll mean
            else:
                # Opposite scenario
                preds[torch.where( x < 3*(1 - self.threshold)*self.last_nll_mean )[0]] = 1 
            return preds

        elif self.cls in ['nlls_reg', 'last_nll_reg']:
            
            preds = self.reg.predict_proba( x.numpy() ) # preds shape [N_data, N_classes], prob of being from that class
            # So we can retrieve the probs of being 1 like this: 
            preds = preds[:,1]
            # Not sure it is useful as preds[:,0] + preds[:,1] = 1...
            return 1.*(torch.tensor(preds) >= self.threshold)
        
        elif self.cls in ['last_layer_cluster_pca']:
            # Here x is shape [Dataset Size, 2]
            preds = self.pca_kmeans.predict(x) # preds shape [Dataset Size]
            return preds
            
        
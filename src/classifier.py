
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
import torch
from torchmetrics.classification import BinaryStatScores
import tqdm



class PromptClassifier:
    
    def __init__(self, 
                 model,
                 model_name: str,
                 tokenizer,
                 device: str,
                 cls: str):
        """
            Args:
                model (HuggingFace model) model used to compute NLL & Perplexity
                                          and AutoPrompt
                tokenizer (HuggingFace tokenizer) 
                device (str) useful for NLL & Perplexity and training a linear classifier.
                cls (str) which classifier to use. Use only last nll, use NN, linear regression, etc.
        """
        assert cls in ['last_nll', 
                       'last_perplexity',
                       'linear_nn',
                       'linear_reg',
                       'logistic_reg']
        self.cls = cls
        
        # init threshold
        self.threshold = 0.
        
        # 
        self.device = device
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.metric = BinaryStatScores()
        
        # Useful 
        self.token2id = {} # The idea is that when tokenizing ##s for example (which is a token)
                           # the tokenizr will treat it as # - # - s which is unfortunate...
        for id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode(id)
            self.token2id[token] = id
    
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
        plt.title(f"CLS {self.cls} ROC Curve {self.model_name}")
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
                f"{self.model_name}_roc_curve.png"
            )
        )
        plt.close()
    
    def chose_best_threshold(self):
        """
            Compute ROC Curve and set self.threshold
            to optimal one.
        """
        pass
    
    def train(self, dataset) -> None:
        """
            dataset is purposedly not more specified as its nature 
            may depend on the classifier type cls.
        """
        
        if self.cls in ['last_nll', 'last_perplexity']:
            # here dataset is supposed to be a set of 
            # random prompts to compute the average of non-prompt
            # nll or perplexity
            
            res = self._compute_nll_perplexity(input_ids = dataset['0']) # shape (dataset length, prompt size + 2) cls and sep
            
            self.last_nll_mean = res['nll'][:,-2].mean().item()
            
            print("last_nll_mean ", self.last_nll_mean)
            return
        elif self.cls in ['linear_reg', 'logistic_reg']:
            if self.cls == 'linear_reg':
                self.reg = LinearRegression()
            elif self.cls == 'logistic_reg':
                self.reg = LogisticRegression()
                
            input_ids0 = dataset['0']
            input_ids1 = dataset['1']
            
            res0 = self._compute_nll_perplexity(input_ids = input_ids0)
            res1 = self._compute_nll_perplexity(input_ids = input_ids1)
            
            X = torch.cat((res0['nll'][:,1:-1], res1['nll'][:,1:-1]))
            y = torch.cat((torch.zeros(input_ids0.shape[0]), 
                           torch.ones(input_ids1.shape[0])))
            
            self.reg = self.reg.fit( X.numpy(), y.numpy() )
            print("Linear Regression Training R^2: ", self.reg.score(X,y))
        
        return
    
    def evaluate(self, dataset) -> dict:
        """
            dataset is purposedly not more specified as its nature 
            may depend on the classifier type cls.
            
            Compute tp, fp, etc...
        """
        
        if self.cls in ['last_nll', 'last_perplexity', 'linear_reg', 'logistic_reg']:
            # non-prompt
            preds0 = self.predict(input_ids = dataset['0']) # shape (DS)
            targets0 = torch.zeros_like(preds0)
            
            # prompt
            preds1 = self.predict(input_ids = dataset['1']) # shape (DS)
            targets1 = torch.ones_like(preds1)
            
            # Cat
            preds =  torch.cat((preds0, preds1))
            targets = torch.cat((targets0, targets1))
            
            # metrics
            res = self.metric(preds, targets)
            return {"tp": res[0].item(), "fp": res[1].item(), "tn": res[2].item(), "fn": res[3].item()}
            
        pass
    
    def predict(self, 
                input_ids: torch.Tensor) -> torch.Tensor:
        """
            Take negative log-likelihoods (nlls) and returns 0 if not
            a prompt 
        """
        
        if self.cls in ['last_nll', 'last_perplexity']:
            
            res = self._compute_nll_perplexity(input_ids = input_ids)
            
            preds = torch.zeros(input_ids.shape[0])
            
            if self.cls == 'last_nll':
                preds[torch.where( res['nll'][:,-2] < (1 - self.threshold)*self.last_nll_mean )[0]] = 1
            elif self.cls == 'last_perplexity':
                preds[torch.where( res['perplexity'][:,-2] < (1 - self.threshold)*self.last_nll_mean )[0]] = 1
            return preds
        
        elif self.cls == 'linear_reg':
            
            res = self._compute_nll_perplexity(input_ids = input_ids)
            
            preds = self.reg.predict( res['nll'][:,1:-1].numpy() )
            
            return 1.*(torch.tensor(preds) >= self.threshold)
        
        elif self.cls == 'logistic_reg':
            
            res = self._compute_nll_perplexity(input_ids = input_ids)
            
            preds = self.reg.predict_proba( res['nll'][:,1:-1].numpy() )
            
            return 1.*(torch.tensor(preds) >= self.threshold)
        
        
        pass
    
    def _compute_nll_perplexity(self, input_ids: torch.Tensor) -> dict:
        """
            Same as MetricsWrapper.compute_nll_perplexity but simpler 
            because here we compute nlls of sentences of same size.
            
            Args:
                input_ids (tensor) each sentence must be tokenize
                                   in the same number of tokens.
                                   shape: (BS, N_tok)
            Returns:
                res (dict) keys: nll, perplexity
                           values: torch.Tensor
        """
        input_ids = input_ids.to(self.device)
        mask_left_to_right = torch.zeros_like(input_ids).to(self.device)
        
        nlls = []
        perplexities = []
        for k in range(input_ids.size(1)):
            mask_left_to_right[:, k] = 1.
            with torch.no_grad():
                logits = self.model(
                                input_ids = input_ids, 
                                attention_mask = mask_left_to_right
                                )
            probs = logits.softmax(-1)
            
            ids = input_ids[:,k] # shape (BS)
            ps = probs[torch.arange(input_ids.shape[0]), k, ids] # shape (BS)
            
            nlls.append(-torch.log(ps).cpu())
            
            perplexities.append(
                        self._compute_perplexity(probs[:,k]).cpu() # will change when we'll do the batch-wise computation
                        )
        
        nlls = torch.vstack(nlls).t() # (BS, L) (L = input_ids.shape[1])   
        perplexities = torch.vstack(perplexities).t()    
            
        return {"nll": nlls,
                "perplexity": perplexities}
    
    def _compute_perplexity(self, probs: torch.Tensor) -> torch.Tensor:
        """
            Here perplexity means according to us.
            
            Args:
                probs (tensor) shape (BS, vocab_size)
            
        """
        H = - (probs * torch.log(probs)).sum(axis = 1) # entropy base e
        return torch.exp(H) # shape BS
    
    def tokenize(self, sentences: list) -> torch.Tensor:
        sentences_tok = [
                            [self.token2id['[CLS]']] + [self.token2id[tok] for tok in sentence.split(' ')] + [self.token2id['[SEP]']]\
                            for sentence in sentences
                        ]
        return torch.tensor(sentences_tok)
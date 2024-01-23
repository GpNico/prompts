
from captum.attr import IntegratedGradients, LayerActivation
import json
import numpy as np
import os
from pathlib import Path
import torch
import tqdm
from typing import Dict, List, Tuple, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils import compute_variation

class KnowledgeNeurons:
    """
        Compute Knowledge Neurons (KNs) of HuggingFace Models
        based on this paper:
         https://arxiv.org/abs/2104.08696
         
    """
    
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 data: Dict[str, Dict[str, Union[List[str], str]]],
                 device: str,
                 kns_path: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.device = device
        self.kns_path = kns_path
        
    def compute_knowledge_neurons(self) -> None:
        
        for k, predicate_id in enumerate(self.data.keys()):
            print(f"Computing {predicate_id} ({k+1}/{len(self.data)})")

            kns_rela = self.compute_knowledge_neurons_by_uuid(
                                predicate_id=predicate_id,
                                )
            if kns_rela:
                with open(os.path.join(self.kns_path, f'{predicate_id}.json'), 'w') as f:
                    json.dump(kns_rela, f)
                    
                    
    def knowledge_neurons_surgery(self,
                                  kns_match: bool = True) -> Dict[str, Dict[str, float]]:
        """
        
            If kns_match = True the KNs corresponding to self.data are loaded.
            Otherwise the other KNs are load (for example, self.data is autoprompt, KNs ParaRel). 
        
        """
        scores = {
                'vanilla': {},
                'wo_kns': {},
                'db_kns': {}
                }
        probs = {
                'vanilla': {},
                'wo_kns': {},
                'db_kns': {}
                }
        relative_probs = {
                        'wo_kns': {},
                        'db_kns': {}
                        }
        
        # KNs path
        if kns_match:
            kns_path = self.kns_path
            rela_names = [n[:-5] for n in os.listdir(kns_path) if '.json' in n]
        else:
            last_directory_name = Path(self.kns_path).parts[-1]
            parent_folder_path = Path(self.kns_path).parent
            # So rela were ignored by ParaRel
            rela_names = list(
                        set(
                            [n[:-5] for n in os.listdir(os.path.join(parent_folder_path, 'autoprompt')) if '.json' in n]
                            ).intersection(
                                set(
                                    [n[:-5] for n in os.listdir(os.path.join(parent_folder_path, 'pararel')) if '.json' in n]
                                    )
                            )
                        )
            if last_directory_name == 'autoprompt':
                kns_path = os.path.join(parent_folder_path, 'pararel')
            elif last_directory_name == 'pararel':
                kns_path = os.path.join(parent_folder_path, 'autoprompt')
            else:
                raise
        
        for i, rela in enumerate(rela_names):
            try:
                 # Getting KNs
                with open(os.path.join(kns_path, rela + '.json'), 'r') as f:
                    rela_kns = json.load(f)
            except:
                print(f"Error with {rela}. Skipped.")
                continue
            print("Compuing Rela ", rela, f" ({i+1}/{len(rela_names)})")
            
            # Vanilla
            vanilla_raw_res = self.eval_one_rela_by_uuid(
                    predicate_id = rela,
                    mode = None,
                    rela_kns = None
                    )
            if vanilla_raw_res:
                #res = average_rela_results(*vanilla_raw_res)
                #scores['vanilla'][rela] = res[0]
                #probs['vanilla'][rela] = res[1]
                pass
            
            # Erase
            wo_raw_res = self.eval_one_rela_by_uuid(
                    predicate_id = rela,
                    mode = 'wo',
                    rela_kns = rela_kns
                    )
            if wo_raw_res:
                #res = average_rela_results(*wo_raw_res)
                #scores['wo_kns'][rela] = res[0]
                #probs['wo_kns'][rela] = res[1]
                relative_probs['wo_kns'][rela] = compute_variation(vanilla_raw_res[1], wo_raw_res[1])
                
            # Enhance
            db_raw_res = self.eval_one_rela_by_uuid(
                    predicate_id = rela,
                    mode = 'db',
                    rela_kns = rela_kns
                    )
            if db_raw_res:
                #res = average_rela_results(*db_raw_res)
                #scores['db_kns'][rela] = res[0]
                #probs['db_kns'][rela] = res[1]
                relative_probs['db_kns'][rela] = compute_variation(vanilla_raw_res[1], db_raw_res[1])
                    
        return relative_probs
    
    def compute_overlap(self) -> None:
        
        # Result path
        res_path = Path(self.kns_path).parent.absolute()
        # ParaRel path
        pararel_path = os.path.join(res_path, 'pararel')
        # Autoprompt path
        autoprompt_path = os.path.join(res_path, 'autoprompt')
        
        # Get relations names
        rela_names = list(
                        set(
                            [n[:-5] for n in os.listdir(pararel_path) if '.json' in n]
                            ).intersection(
                                set(
                                    [n[:-5] for n in os.listdir(autoprompt_path) if '.json' in n]
                                    )
                            )
                        )
        
        # scores
        overlap_by_rela = {}
        
        # Used for global overlap
        autoprompt_kns = {}
        pararel_kns = {}
        print("Intra Relation Overlap:")
        for i, rela in enumerate(rela_names):
            # Getting KNs by rela for rela overlap
            with open(os.path.join(autoprompt_path, rela + '.json'), 'r') as f:
                autoprompt_rela_kns = json.load(f)
            autoprompt_kns.update(autoprompt_rela_kns)
            with open(os.path.join(pararel_path, rela + '.json'), 'r') as f:
                pararel_rela_kns = json.load(f)
            pararel_kns.update(pararel_rela_kns)
            
            # Compute rela overlap
            overlap_by_rela[rela] = KnowledgeNeurons._overlap_metrics(pararel_rela_kns, 
                                                                      autoprompt_rela_kns)
            print(f"\t{rela}: {np.round(overlap_by_rela[rela]['overlap'], 2)}  (on {np.round(overlap_by_rela[rela]['num_kns_1'] ,2)} ParaRel KNs & {np.round(overlap_by_rela[rela]['num_kns_2'] ,2)} Autoprompt KNs)")
            
        # Global overlap
        global_overlap = KnowledgeNeurons._overlap_metrics(pararel_kns, 
                                                           autoprompt_kns)
        print(f"Total Overlap: {np.round(global_overlap['overlap'], 2)}  (on {np.round(global_overlap['num_kns_1'] ,2)} ParaRel KNs & {np.round(global_overlap['num_kns_2'] ,2)} Autoprompt KNs)")
            
                            
    def compute_knowledge_neurons_by_uuid(
                        self,
                        predicate_id: str,
                        t_thresh: float = 0.3, # In github 0.3, in the paper 0.2
                        p_thresh: float = 0.3  # In github 0.5, in the paper 0.7
                        ) -> Dict[str, List[Tuple[float, float]]]:
        """
            Compute KNs for one relation from ParaRel (e.g. P264).
            Note that parameters are chosen based on this implementation: 
            https://github.com/EleutherAI/knowledge-neurons
            So here are the differences with the paper:
                t = in github 0.3, in the paper 0.2
                p = in github 0.5, in the paper 0.7
            We agreed to lower the threshold as we are already being to selective (i.e.
            fewer KNs than Github)
            
            TBD: Add adaptative Refining by increasing/decreasing p by 0.05 if the
                 number of KNs is not within [3,5].
                 
            Args:
                predicate_id (str) ParaRel relation id e.g. P1624
                t_thresh (float) t threshold from the paper
                p_thresh (float) p threshold from the paper
                
            Returns:
                kns (dict) keys: uuid      values: list of KNs in the format (layer, neuron) 
                           ex: {'uuid1': [(11, 2045), (12, 751)], 'uuid2': [(5, 3014)]}
        
        """
        
        # Get Dataset
        dataset = self.data[predicate_id]
        
        if dataset is None:
            return None

        # Compute IG attributions
        kns = {} # key uuid
        for uuid in tqdm.tqdm(dataset.keys(), total = len(dataset)):
            sentences = dataset[uuid]['sentences'] # List[str]
            Y = dataset[uuid]['Y'] # str
            # Tokenization
            target = self.tokenizer([Y]*len(sentences), return_tensors = 'pt')['input_ids'][:,1:-1].to(self.device) # No need to pas
            encodings = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device) # No batch for know
            target_pos = torch.where(encodings.input_ids == self.tokenizer.mask_token_id)[1]
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask

            ### Attr ###
            # little trick to avoid Cuda out of memory
            k, batch_size = 0, len(sentences)
            fail_flag = False # Possible that even with only one sentence it crashes.
            uuid_attr_lst = []
            while k < len(sentences):
                _input_ids = input_ids[k:k + batch_size, :]
                _attention_mask = attention_mask[k:k + batch_size, :]
                _target = target[:, k:k + batch_size] # /!\
                _target_pos = target_pos[k:k + batch_size]
                try:
                    uuid_attr = []
                    for l in range(self.model.config.num_hidden_layers):
                        attr = self.IG_batch(
                                input_ids = _input_ids,
                                attention_mask = _attention_mask,
                                target = _target,
                                target_pos = _target_pos,
                                layer_num = l
                                )
                        # attr shape [BS, num_neurons]
                        uuid_attr.append(attr.unsqueeze(1)) # create layer dim (shape [BS, 1, num neurons])
                    k += batch_size
                    uuid_attr_lst.append(torch.cat(uuid_attr, dim=1)) # cat along layer dim (shape [BS, num_layers, num neurons])
                except RuntimeError as e:
                    print(e)
                    print("error!")
                    if "CUDA out of memory" in str(e):
                        if batch_size == 1:
                            print("Couldn't load this uuid on GPU. Skipping it.")
                            fail_flag = True
                            break
                        print(f"Reducing batch size from {batch_size} to {batch_size // 2}")
                        batch_size //= 2
                        k = 0 # reseting batch count
                        uuid_attr_lst = []
                    else:
                        raise e
                if fail_flag:
                    continue

            uuid_attr = torch.vstack(uuid_attr_lst)

            ## Refining

            # for each prompt, retain the neurons with attribution scores greater than the attribution threshold t, obtaining the coarse set of knowledge neurons
            # BUT the max is computed for each prompt, not for all prompts at onec /!\
            max_attr = uuid_attr.max(dim = -1).values.max(dim = -1).values
            max_attr = max_attr.unsqueeze(1).unsqueeze(2) # to match uuid_attr shape
            threshold = t_thresh*max_attr # t of the paper

            prompts_idx, layer_idx, neuron_idx = torch.where(uuid_attr > threshold)

            neuron2prompt = {}
            for k in range(len(prompts_idx)):
                layer, neuron = layer_idx[k].item(), neuron_idx[k].item()
                if (layer, neuron) in neuron2prompt.keys():
                    neuron2prompt[(layer, neuron)] += 1
                else:
                    neuron2prompt[(layer, neuron)] = 1

            # considering all the coarse sets together, retain the knowledge neurons shared by more than p% prompts.
            uuid_kns = [k for k,v in neuron2prompt.items() if v >= dataset[uuid]['num_prompts']*p_thresh]

            kns[uuid] = uuid_kns

        return kns
        
    def IG_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        target_pos: torch.Tensor,
        layer_num: int,
        n_steps: int = 20
        ) -> torch.Tensor:
        """
            Compute Integrated Gradients of the second layer of
            the model's layer_num^th encoder FFN.
            This where KNs are supposed to hide.
            
            This code is using the Captum library that does not provide
            ways of attributing specific layer activations.
            
            For those who care, Captum provides ways of attributing the 
            INPUT with respect to a specific layer activation or neuron
            activation with the LayerIntegratedGradients and 
            NeuronIntegratedGradients.
            
            Returns:
                attr (Tensor) shape [batch_size, layer_size]
        """
        
        def forward_pos_func(hidden_states: torch.Tensor) -> torch.Tensor:
            """
                This function is a trick to evaluate a Pytorch model
                from a particular layer. 
                In this case the second Linear layer from the FFN of
                a particular encoder.
                
                To do this we register a hook at the desired layer that
                will change the output by the hidden_states tensor.
                
                It must have a way of doing this more easily and efficiently
                as we are passing a tensor through the entire model instead 
                of just layer->output, but hey it works!
            """
        
            def _custom_hook(
                    module: torch.nn.Module, 
                    input: torch.Tensor, 
                    output: torch.Tensor) -> None:
                """
                    Pytorch hook that change the output by a
                    tensor hidden_states.
                    Note that passing through inplace operations
                    is mandatory to enable gradients flow.
                """
                output -= output
                output += hidden_states

            # Works only for BERT right now
            hook_handle = self.model.bert.encoder.layer[layer_num].intermediate.register_forward_hook(_custom_hook)

            res = self.model(
                    input_ids = input_ids.repeat(n_steps, 1),
                    attention_mask = attention_mask.repeat(n_steps, 1)
                    ) # fact has a new first dim which is the discretization to compute the int
            
            hook_handle.remove()

            outputs = res.logits[torch.arange(res.logits.shape[0]),
                                 target_pos.repeat(n_steps),
                                 target.flatten().repeat(n_steps)] # Get only attribution of the target_pos
            return outputs

        # Gradients will be used so ensure they are 0
        self.model.zero_grad()
        
        # Get the activation of the desired layer
        la = LayerActivation(self.model, self.model.bert.encoder.layer[layer_num].intermediate)
        hidden_states = la.attribute(
                            input_ids,
                            additional_forward_args = attention_mask
                            ) # Shape [Batch_size, L, d]

        # This one is pure security as LayerActivation shall not modify gradient
        self.model.zero_grad()
        
        # Compute Integrated Gradient using the forward_pos_func defined above
        # and attributiong the target layer activations
        ig = IntegratedGradients(forward_pos_func)
        attr = ig.attribute(hidden_states, n_steps = n_steps)
        
        return attr[torch.arange(target_pos.shape[0]), target_pos, :]
    
    
    def eval_one_rela_by_uuid(
                        self,
                        predicate_id: str,
                        mode: str = None,
                        rela_kns: dict[str, List[Tuple[int, int]]] = None
                        ) -> Tuple[ Dict[str, Dict[str, float]], Dict[str, List[float]]]:
        """

            ...

        """
        self.model.eval()

        assert not(mode) or rela_kns

        # Get Dataset
        dataset = dataset = self.data[predicate_id]
        
        if dataset is None:
            return None

        # Compute T-Rex Scores
        scores = {}
        probs = {}
        softmax = torch.nn.Softmax(dim=1)
        for uuid in tqdm.tqdm(dataset.keys(), total=len(dataset)):
            sentences = dataset[uuid]['sentences']
            Y = dataset[uuid]['Y']
            # Tokenization
            target = self.tokenizer([Y]*len(sentences), return_tensors = 'pt')['input_ids'][:,1:-1].to(self.device) # No need to pas
            encodings = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device) # No batch for know
            target_pos_i, target_pos_j = torch.where(encodings.input_ids == self.tokenizer.mask_token_id)
            # Register hooks
            if mode:
                hook_handles = self.register_nks_hooks(
                                            rela_kns[uuid],
                                            mode = mode,
                                            mask_pos = (target_pos_i, target_pos_j),
                                            batch_size = encodings.input_ids.shape[0],
                                            sequence_length = encodings.input_ids.shape[1]
                                            )

            # Forward pass
            with torch.no_grad():
                logits = self.model(**encodings).logits

            # Remove hooks
            if mode:
                KnowledgeNeurons.remove_hooks(hook_handles)

            _, ids = torch.topk(logits[target_pos_i, target_pos_j], k = 100)
            ids = ids.cpu()
            target = target.cpu()

            scores[uuid] = {'P@1': 0,
                            'P@5': 0,
                            'P@20': 0,
                            'P@100': 0}
            scores[uuid]['P@1'] = ((target[:] == ids[:,:1]).any(axis = 1)).sum().item()/len(sentences)
            scores[uuid]['P@5'] = ((target[:] == ids[:,:5]).any(axis = 1)).sum().item()/len(sentences)
            scores[uuid]['P@20'] = ((target[:] == ids[:,:20]).any(axis = 1)).sum().item()/len(sentences)
            scores[uuid]['P@100'] = ((target[:] == ids[:,:100]).any(axis = 1)).sum().item()/len(sentences)

            output_probs = softmax(logits[target_pos_i, target_pos_j].cpu())
            probs[uuid] = output_probs[torch.arange(output_probs.shape[0]), target.flatten()].tolist()

        return scores, probs
    
    
    @staticmethod
    def remove_hooks(hook_handles: List[torch.utils.hooks.RemovableHandle]) -> None:
        """
            Remove hooks from the model.
        """
        for hook_handle in hook_handles:
            hook_handle.remove()

    def register_nks_hooks(
                        self,
                        kns: List[Tuple[int,int]],
                        mode: str,
                        mask_pos: Tuple[torch.Tensor, torch.Tensor],
                        batch_size: int,
                        sequence_length: int,
                        num_neurons: int = 3072) -> List[torch.utils.hooks.RemovableHandle]:
        """
            Register hooks in the second layer of some transfomer's encoder FFN.
            
            The encoders l that will be affected are defined in kns parameters that contains the
            knowledge neurons.
            
            This hook multiplies the activations of n^th neuron of this layer by 2 if mode = 'db'
            and by 0 if mode = 'wo'. n here is defined in kns.
            
            Overall kns = [..., (l,n), ...]
            
            TBD: get num_neurons from the model parameters directly.
            
            Args:
                kns (list) contains the knowledge neurons like (layer num, neuron num)
                mode (str) enhance (db) or erase (wo) knowledge
                mask_pos (tuple) this activation modification only applies at the mask pos
                batch_size (int) usefull to define fact
                squence_length (int) usefull to define fact
                num_neurons (int) cf. TBD
            Returns:
                hook_handles (list) needed to remove hooks, otherwise everything is broken!
        
        """
        assert mode in ['wo', 'db']

        # Get neurons by layer
        layer2neurons = {}
        for kn in kns:
            layer, neuron = kn
            if layer in layer2neurons.keys():
                layer2neurons[layer].append(neuron)
            else:
                layer2neurons[layer] = [neuron]

        # Get by what we will multiply KNs activations
        layer2fact = {}
        for layer in layer2neurons.keys():
            fact = 1.*torch.ones(batch_size, sequence_length, num_neurons)
            for neuron in layer2neurons[layer]:
                if mode == 'wo':
                    fact[mask_pos[0], mask_pos[1], neuron] = 0.
                elif mode == 'db':
                    fact[mask_pos[0], mask_pos[1], neuron] = 2.
                layer2fact[layer] = fact.to(self.device)

        # Create Custom Hooks
        def _hook_template(
                        module: torch.nn.Module, 
                        input: torch.Tensor, 
                        output: torch.Tensor, 
                        fact: torch.Tensor
                        ) -> None:
            """
                Modify the output by multiplying them by fact.
                
                Note that a Pytorch hook can only takes module,
                input and output as arguments so we used a trick
                here by defining this "hook template" and by 
                registering the hook using a lambda function that
                get rid of the fact argument.
            """
            output *= fact
        hook_handles = []
        for layer in layer2fact.keys():
            hook_handle = self.model.bert.encoder.layer[layer].intermediate.register_forward_hook(
                                        lambda module,input,output,fact=layer2fact[layer]: _hook_template(module, input, output, fact)
                                        )
            hook_handles.append(hook_handle)

        return hook_handles
    
    @staticmethod
    def _overlap_metrics(kns_dict_1: Dict[str, List[ Tuple[float, float]]],
                         kns_dict_2: Dict[str, List[ Tuple[float, float]]]) -> Dict[str, float]:
        """
            For each uuid get its KNs and compute the overlap.
            
            Returns the average.
        
        """
        overlap = 0.
        prop_overlap = 0. # TBD
        num_kns_1 = 0.
        num_kns_2 = 0.
        for uuid in kns_dict_1.keys():
            assert uuid in kns_dict_2.keys()
            
            kns1, kns2 = kns_dict_1[uuid], kns_dict_2[uuid]
            kns1 = [(e[0], e[1]) for e in kns1]
            kns2 = [(e[0], e[1]) for e in kns2]
            
            # num kns
            num_kns_1 += len(kns1)
            num_kns_2 += len(kns2)
            # overlap
            kns_overlap = set(kns1).intersection(set(kns2))
            
            # Here we compute the size of the overlap
            # We could also compute the proportion of KNs shared
            overlap += len(kns_overlap)
            
        return {'overlap': overlap/len(kns_dict_1),
                'num_kns_1': num_kns_1/len(kns_dict_1),
                'num_kns_2': num_kns_2/len(kns_dict_2)}
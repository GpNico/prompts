"""
Dataset of prompts based on random relations (same relations as above though)
    - Should we get paraphrases by doing auto-prompt training with the objective to output similar probabilities
        (i) (X, Answer from Random Prompt but constrained to Y)
        (ii) (X, Shuffle Y) shuffled within a relation  P1
             Note to self: Xs are coherent, Ys are coherent, X and Y related, (X,Y)s are not coherent
        (iii) (X, Y’) shuffled, with X coming from one relation, Y’ coming from another relation P2
              Note to self: Xs are coherent, Ys are coherent, X and Y not related, (X,Y)s are not coherent
        (iv) (X, Y’) sample X and Y’ from all the X’s and Y’s from all relations P3
             Note to self: Xs are not coherent, Ys are not coherent, X and Y not related, (X,Y)s are not coherent)
        (v) (X,Y) sample X and Y from all possible tokens

Currently doing: (ii), (iii) & (iv)

Questions:
    1. Is Autoprompt able to map any X to any Y?
    2. If yes which Knowledge Neuron is accessed while doing so?

Expected template of Autoprompt:

{"index":509,
 "sub_uri":"Q3300448",
 "obj_uri":"Q96",
 "obj_label":"Mexico",
 "sub_label":"Ozumba",
 "lineid":510,
 "uuid":"5a2d6d86-4086-4d6f-b64c-e9fe3afdd355"}

BUT: "index" optionnal, "lineid" optionnal
AND: call, "sub_uri" and "obj_uri" useless too.

One line per uuid (or training example): at least two files, train.jsonl & dev.jsonl

"""

import argparse
import json
import os
import random
from typing import List, Dict


TRAIN_DEV_SPLIT = 0.8

def shuffle2(rela_dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
        Xs and Ys same relation BUT Ys is shuffled.
    """
    Ys = []
    for elem in rela_dataset:
       Ys.append(elem['obj_label'])
    random.shuffle(Ys)
    
    for k in range(len(rela_dataset)):
        rela_dataset[k]['obj_label'] = Ys[k]

    return rela_dataset

def shuffle3(dataset: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    """
        Xs from one relation, Ys from another.
    """
    predicate_ids = list(dataset.keys())
    
    # One simple way to do this is to take Ys from Xs predicate_id's next 
    
    # need to save Ys for the zero^th elem
    elem_zero_Ys = [elem['obj_label'] for elem in dataset[predicate_ids[0]]]
    
    for k, predicate_id in enumerate(predicate_ids):
        Ys_predicate_id = predicate_ids[(k+1)%len(predicate_ids)]
        length = min(len(dataset[predicate_id]), len(dataset[Ys_predicate_id]))    
        for i in range(length):
            if k+1 < len(predicate_ids):
                dataset[predicate_id][i]['obj_label'] = dataset[Ys_predicate_id][i]['obj_label']
            else:
                dataset[predicate_id][i]['obj_label'] = elem_zero_Ys[i]
    return dataset

if __name__ == '__main__':
    
    ### Get Params ###
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffling_method', 
                        type=int, 
                        default=2)
    args = parser.parse_args()
    
    ### Open T-Rex vocab ###
    
    # Get predicate ID
    predicate_ids = [n[:-6] for n in os.listdir(os.path.join('data', 'pararel', 'trex_lms_vocab'))]
    
    # Load vocab
    dataset = {}
    for predicate_id in predicate_ids:
        rela_dataset = []
        with open(os.path.join('data', 'pararel', 'trex_lms_vocab', f'{predicate_id}.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                rela_dataset.append({'sub_label': data['sub_label'], 
                                     'obj_label': data['obj_label'], 
                                     'uuid': data['uuid']})
        dataset[predicate_id] = rela_dataset
        
        
    ### Write Dataset ###
    if args.shuffling_method == 3:
        dataset = shuffle3(dataset)
    
    for predicate_id in predicate_ids:
        # shuffle data before split
        random.shuffle(dataset[predicate_id])
        
        # Create directory
        rela_path = os.path.join(
                        'data',
                        'shuffled_trex', 
                        str(args.shuffling_method), 
                        predicate_id
                        )
        os.makedirs(rela_path, exist_ok=True)
        
        # Shuffle
        if args.shuffling_method == 2:
            rela_dataset = shuffle2(dataset[predicate_id])
        else:
            rela_dataset = dataset[predicate_id]
        
        # Save file
        num_train_items = int(TRAIN_DEV_SPLIT*len(rela_dataset))
        
        with open(os.path.join(rela_path, 'train.jsonl'), "w") as jsonl_file:
            for item in rela_dataset[:num_train_items]:
                jsonl_file.write(json.dumps(item) + "\n")
                
        with open(os.path.join(rela_path, 'dev.jsonl'), "w") as jsonl_file:
            for item in rela_dataset[num_train_items:]:
                jsonl_file.write(json.dumps(item) + "\n")
            
        
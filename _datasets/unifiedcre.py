import os
import re
import logging
from ast import literal_eval
from collections import defaultdict
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from datasets import Features, Value, ClassLabel


# Currently available datasets
data_dir = "data"
available_datasets = [
    'altlex','because','ctb','esl','esl2','pdtb',
    'semeval2010t8','cnc','causenet','causenetm'
    ]

# Test and Dev splits by doc_id. The ones not mentioned are for Train.
# Dev is used for Testing by Default.
# We pre-split the dataset and save under data_dir in "splits" folder
splits = {
    'altlex': {
        'test': ['altlex_dev.tsv']
        },
    'because': {
        'test': ['Article247_327.ann','wsj_22(.)+','wsj_23(.)+','wsj_00(.)+','wsj_01(.)+','wsj_24(.)+']
        }, #10% random doc & follow PDTB
    'ctb': {
        'test': ['ea980120.1830.0456.tml','APW19980227.0494.tml','PRI19980306.2000.1675.tml',
        'APW19980213.1320.tml','APW19980501.0480.tml', 'PRI19980205.2000.1998.tml',
        'wsj_22(.)+','wsj_23(.)+','wsj_00(.)+','wsj_01(.)+','wsj_24(.)+']
        }, #10% random doc & follow PDTB
    'esl': {
        'test': ['37_(.)+','41_(.)+']
        },
    'esl2': {
        'test': ['37_(.)+','41_(.)+']
        },
    'pdtb': {
        'test': ['wsj_22(.)+','wsj_23(.)+'], 
        'dev': ['wsj_00(.)+','wsj_01(.)+','wsj_24(.)+']
        },
    'semeval2010t8': {
        'test': ['test.json']
        },
    'cnc': {
        'dev': ['train_10_(.)+'],
        'test': ['test(.)+']
    },
    'causenet': {
        'test': [] # under 'causenet-test-doc_id.txt'
    },
    'causenetm': {
        'dev': ['dev(.)+'],
        'test': ['test(.)+']
    }
}

# Tasks that the dataset supports. We might not have access to all mentioned.
# All examples should be eligible for seq_clf: given text, clf causal [requires deduplication]
tasks = {
    'argument': ['pdtb','cnc','altlex','because','fincausal2021','causenetm'], # find causal spans [requires grouping]
    'pair_clf': ['semeval2010t8','ctb','pdtb','cnc','altlex','because','causenet','causenetm'], # given pair, clf causal [as is]
}

# For all datasets
ft = {
    'corpus': Value('string'),
    'doc_id': Value('string'), 
    'sent_id': Value('string'),
    'eg_id': Value('int64'),
    'index': Value('string'),
    'text': Value('string'),
    'text_w_pairs': Value('string'),
    'context': Value('string'),
    'seq_label': ClassLabel(names=['No','Yes']),
    'pair_label': ClassLabel(names=['No','Yes']),
    'num_sents': Value('int64')
}
# For grouped spans
ft2 = {
    'causal_text_w_pairs': Value('string'),
    'num_rs': Value('int64')
}


def clean_tok(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub('</*[A-Z]+\d*>','',tok) 


def get_BIO(text_w_pairs):
    tokens = []
    ce_tags = []
    next_tag = tag = 'O'
    for tok in text_w_pairs.split(' '):

        # Replace if special
        if '<ARG0>' in tok:
            tok = re.sub('<ARG0>','',tok)
            tag = 'B-C'
            next_tag = 'I-C'
        elif '</ARG0>' in tok:
            tok = re.sub('</ARG0>','',tok)
            tag = 'I-C'
            next_tag = 'O'
        elif '<ARG1>' in tok:
            tok = re.sub('<ARG1>','',tok)
            tag = 'B-E'
            next_tag = 'I-E'
        elif '</ARG1>' in tok:
            tok = re.sub('</ARG1>','',tok)
            tag = 'I-E'
            next_tag = 'O'

        tokens.append(clean_tok(tok))
        ce_tags.append(tag)
        tag = next_tag
    
    return tokens, ce_tags


def mass_append(eg, dict_to_append_to, text_column=None, label_column=None, text_column_name='text', label_column_name='seq_label'):
    dict_to_append_to['corpus'].append(eg['corpus'])
    dict_to_append_to['index'].append(eg['index'])
    if text_column is not None:
        dict_to_append_to['text'].append(text_column)
    else:
        dict_to_append_to['text'].append(eg[text_column_name])
    if label_column is not None:
        dict_to_append_to['label'].append(label_column)
    else:
        dict_to_append_to['label'].append(eg[label_column_name])


def get_data_files_dict(list_of_dataset_names:list, ddir:str, do_train_val:bool):
    data_files = defaultdict(list)
    for d in list_of_dataset_names:
        data_files['train'].append(os.path.join(ddir,f'{d}_train.csv'))
        if 'dev' in splits[d].keys():
            if do_train_val:
                data_files['validation'].append(os.path.join(ddir,f'{d}_dev.csv'))
            else:
                data_files['train'].append(os.path.join(ddir,f'{d}_dev.csv'))
        if do_train_val:
            data_files['validation'].append(os.path.join(ddir,f'{d}_test.csv'))
        else:
            data_files['train'].append(os.path.join(ddir,f'{d}_test.csv'))
    return dict(data_files)


def load_span_dataset_ungrouped(dataset_name:list, do_train_val:bool, span_files:dict={}, data_dir:str=data_dir):
    # Sanity checks
    datasets_for_span = []
    if dataset_name is not None:
        for d in dataset_name:
            if d not in available_datasets:
                raise ValueError(f'"{d}" dataset is unavailable.')
            if d in tasks['argument']:
                datasets_for_span.append(d) # take as is
            else:
                logging.info(f'"{d}" dataset is ignored because it is unsuitable for span detection task.')
    
    # Process
    if len(datasets_for_span)>0:
        data_files = get_data_files_dict(datasets_for_span, os.path.join(data_dir,'splits'), do_train_val)
    else:
        data_files = {'train':[],'validation':[]}
    if len(span_files)>0:
        for k,v in span_files.items():
            data_files[k].append(v)
    data_files2 = {}
    for k,v in data_files.items():
        if len(v)>0:
            data_files2[k]=v
    data_files = data_files2
    del(data_files2)

    span_dataset = load_dataset('csv', data_files=data_files, features=Features(ft))

    sdataset = DatasetDict()

    for s in list(data_files.keys()): # train, validation

        posi_span = {'corpus':[], 'index':[],'text':[],'label':[],'ce_tags':[]}

        for eg in span_dataset[s]:
            if int(eg['seq_label'])==1 and int(eg['pair_label'])==1 and eg['text_w_pairs'] is not None:
                # positive spans
                tokens, ce_tags = get_BIO(eg['text_w_pairs'])
                mass_append(eg, posi_span, text_column=tokens, label_column_name='pair_label')
                posi_span[f'ce_tags'].append(ce_tags)
            else:
                # we ignore not-causal examples, or examples with no pair annotations
                # model tends to perform worse having to decipher negative from positive examples 
                # while doing span detection
                pass
    
        sdataset[f'span_{s}'] = Dataset.from_dict(posi_span)

    # print(sdataset)
    # print(sdataset['span_validation'][0])

    return sdataset


def get_args(text_w_pairs):
    return [
        re.search(r'<ARG0>(.*?)</ARG0>', text_w_pairs).group(1).strip()+'.',
        re.search(r'<ARG1>(.*?)</ARG1>', text_w_pairs).group(1).strip()+'.'
    ]


def load_cre_dataset(dataset_name:list, do_train_val:bool, \
    also_add_span_sequence_into_seq:bool=False, span_augment:bool=False,
    span_files:dict={}, seq_files:dict={}, do_train:bool=True, data_dir:str=data_dir):
    # Sanity checks
    datasets_for_span, datasets_for_others = [], []
    if dataset_name is not None:
        for d in dataset_name:
            if d not in available_datasets:
                raise ValueError(f'"{d}" dataset is unavailable.')
            if d in tasks['argument']:
                datasets_for_span.append(d) # requires grouped sources
            else:
                datasets_for_others.append(d) # take as is
    logging.debug(f'Files for all three tasks: {datasets_for_span}')
    logging.debug(f'Files for non-span det tasks: {datasets_for_others}')

    # Process
    all_splits = []
    if len(datasets_for_span)>0:
        data_files = get_data_files_dict(datasets_for_span, os.path.join(data_dir,'grouped','splits'), do_train_val)
        if 'train' in data_files.keys() and not do_train:
            del(data_files['train'])
    else:
        data_files = {}
    data_files = defaultdict(list,data_files)
    if len(span_files)>0:
        for k,v in span_files.items():
            data_files[k].append(v)
    if len(data_files)>0:
        logging.debug(f'Files for span_dataset: {data_files}')
        span_dataset = load_dataset('csv', data_files=data_files, features=Features({**ft,**ft2}))
        all_splits.extend(list(data_files.keys()))
    else:
        span_dataset = None
    
    if len(datasets_for_others)>0:
        data_files = get_data_files_dict(datasets_for_others, os.path.join(data_dir,'splits'), do_train_val)
        if 'train' in data_files.keys() and not do_train:
            del(data_files['train'])
    else:
        data_files = {}
    data_files = defaultdict(list,data_files)
    if len(seq_files)>0:
        for k,v in seq_files.items():
            data_files[k].append(v)
    if len(data_files)>0:
        logging.debug(f'Files for main_dataset: {data_files}')
        main_dataset = load_dataset('csv', data_files=data_files, features=Features(ft))
        all_splits.extend(list(data_files.keys()))
    else:
        main_dataset = None
    
    dataset = DatasetDict()
    sdataset = DatasetDict()

    logging.debug(f"span_dataset: {span_dataset}")
    logging.debug(f"main_dataset: {main_dataset}")

    for s in set(all_splits): # train, validation

        posi_span = {'corpus':[], 'index':[],'text':[],'label':[],'ce_tags':[],'ce_tags1':[],'ce_tags2':[]}
        seq = {'corpus':[], 'index':[],'text':[],'label':[]}
        pair = {'corpus':[], 'index':[],'text':[],'label':[]} #text_w_pairs

        if span_dataset is not None:
            for eg in span_dataset[s]:
                if int(eg['seq_label'])==1 and int(eg['pair_label'])==1:
                    if eg['causal_text_w_pairs'] is None:
                        if eg['corpus'] in tasks['pair_clf'] and eg['text_w_pairs'] is not None:
                            mass_append(eg, pair, text_column_name='text_w_pairs', label_column_name='pair_label')
                        mass_append(eg, seq, text_column_name='text', label_column_name='seq_label')
                    else:
                        # positive spans
                        ce_tags = []
                        for text_w_pairs in literal_eval(eg['causal_text_w_pairs']):
                            if (eg['corpus'] in tasks['pair_clf']) or (eg['corpus'] not in available_datasets):
                                mass_append(eg, pair, text_column=text_w_pairs, label_column_name='pair_label')
                            tokens, _ce_tags = get_BIO(text_w_pairs)
                            ce_tags.append(_ce_tags)
                            if span_augment and s=='train':
                                for t in get_args(text_w_pairs):
                                    mass_append(eg, seq, text_column=t, label_column=0)
                        if (eg['corpus'] in tasks['argument']) or (eg['corpus'] not in available_datasets):
                            mass_append(eg, posi_span, text_column=tokens, label_column_name='seq_label')
                            missing = int(3-len(ce_tags))
                            if missing==0:
                                pass # do nothing
                            elif missing==1  or missing==2:
                                ce_tags = ce_tags+[['O']*len(ce_tags[0])]*missing
                            elif missing<0:
                                logging.debug(eg['causal_text_w_pairs'])
                                logging.debug(ce_tags)
                                raise ValueError('There should not be more than 3 causal relations per example.')
                            else: # missing>3 (or strange decimals)
                                logging.debug(eg['causal_text_w_pairs'])
                                logging.debug(ce_tags)
                                raise ValueError('There should not be 0 causal relations per example.')
                            for ct, tags in enumerate(sorted(ce_tags)):
                                if ct==0:
                                    ct='' # replace
                                posi_span[f'ce_tags{ct}'].append(tags)
                        if also_add_span_sequence_into_seq: # Warning: This might lead to duplicates if using both span_ and seq_ datasets
                            mass_append(eg, seq, text_column_name='text', label_column_name='seq_label')
                elif int(eg['seq_label'])==0 and int(eg['pair_label'])==1:
                    raise ValueError('There should be no such examples. Preprocessing error!')
                elif int(eg['seq_label'])==1 and int(eg['pair_label'])==0:
                    if ((eg['corpus'] in tasks['pair_clf']) or (eg['corpus'] not in available_datasets)) and \
                        (eg['text_w_pairs'] is not None):
                        mass_append(eg, pair, text_column_name='text_w_pairs', label_column_name='pair_label')
                else: # i.e. eg['seq_label']==0 and eg['pair_label']==0:
                    # negative seqs
                    if ((eg['corpus'] in tasks['pair_clf']) or (eg['corpus'] not in available_datasets)) and \
                        (eg['text_w_pairs'] is not None):
                        mass_append(eg, pair, text_column_name='text_w_pairs', label_column_name='pair_label')
                    if int(eg['eg_id'])==0: # deduplication
                        mass_append(eg, seq, text_column_name='text', label_column_name='seq_label')

            del(span_dataset[s])

        if main_dataset is not None:
            for eg in main_dataset[s]:
                if eg['eg_id']==0: # deduplication
                    mass_append(eg, seq, text_column_name='text', label_column_name='seq_label')
                if ((eg['corpus'] in tasks['pair_clf']) or (eg['corpus'] not in available_datasets)) and \
                    (eg['text_w_pairs'] is not None):
                    mass_append(eg, pair, text_column_name='text_w_pairs', label_column_name='pair_label')

            del(main_dataset[s])

        # Add to dataset
        dataset[f'seq_{s}'] = Dataset.from_dict(seq)
        dataset[f'pair_{s}'] = Dataset.from_dict(pair)
        sdataset[f'span_{s}'] = Dataset.from_dict(posi_span)
    
    # Get Train lengths
    seq_len = len(dataset["seq_train"]) if "seq_train" in dataset.keys() else 0
    pair_len = len(dataset["pair_train"]) if "pair_train" in dataset.keys() else 0
    span_len = len(sdataset["span_train"]) if "span_train" in sdataset.keys() else 0
    lowest_denom = max(min(span_len, pair_len, seq_len),1)
    pspan = int(span_len/lowest_denom)
    apair = int(pair_len/lowest_denom)
    aseq = int(seq_len/lowest_denom)
    min_batch_size = apair + aseq + pspan
    
    return sdataset, dataset, (min_batch_size, pspan, apair, aseq)
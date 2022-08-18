import logging
import os
import re
import sys
import tempfile
import pandas as pd
from itertools import product
from .files import make_dir, save_json, open_json
from datetime import datetime

# get unique tmp file per run
tmp_file_path = f'outs/tmp/{next(tempfile._get_candidate_names())}.json'
make_dir(tmp_file_path)


def get_logger(logname, no_stdout=True, set_level=0, datefmt='%d/%m/%Y %H:%M:%S'):
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt=datefmt
        )
    logger = logging.getLogger()
    logger.setLevel(set_level)

    handler = logging.StreamHandler(open(logname, "a"))
    handler.setLevel(set_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if no_stdout:
        logger.removeHandler(logger.handlers[0])

    return logger


def save_params(args, save_results=True, tmp_file_path=tmp_file_path, datefmt='%d/%m/%Y %H:%M:%S'):

    model_args, data_args, training_args = args

    if save_results:
        res_summary = {
            'starttime': datetime.now().strftime(datefmt),
            'model_args': str(model_args),
            'data_args': str(data_args),
            'training_args': str(training_args)
            }
        save_json(res_summary, tmp_file_path)


def extend_res_summary(additional_res, tmp_file_path=tmp_file_path):
    res_summary = open_json(tmp_file_path, data_format=dict)
    res_summary.update(additional_res)
    save_json(res_summary, tmp_file_path)


def get_average(df, filter_by):
    """
    df [pd.DataFrame]
    filter_by [list or tuples] : list of strings to filter column names for averaging across folds
    E.g. "Train_K3_Micro_F1" can be found via ["Train", "Micro_F1"]
    Does edits in place!
    """
    keep_cols = [col for col in df.columns if all(
        [fil in col for fil in filter_by])]
    df['AVG_'+'_'.join(filter_by)] = df[keep_cols].mean(axis=1)


def save_results_to_csv(save_file_path, append=True, tmp_file_path=tmp_file_path, datefmt='%d/%m/%Y %H:%M:%S'):
    """
    Takes res_summary of current run (in json format) and appends to main results frame (in csv format)
    """
    # load tmp results
    res_summary = open_json(tmp_file_path, data_format=pd.DataFrame)

    # calculate average scores
    combis = list(product(
        ['CV', 'Val'], 
        ['precision', 'recall', 'f1', 'exact match', 'loss', 
        'precision_CE', 'recall_CE', 'f1_CE', 'exact match_CE']
        ))
    for combi in combis:
        get_average(res_summary, combi)

    # calculate end time
    end = datetime.now()
    res_summary['endtime'] = end.strftime(datefmt)
    res_summary['timetaken'] = end - \
        datetime.strptime(res_summary['starttime'][0], datefmt)

    if append and os.path.isfile(save_file_path):
        # load old file
        old_summary = pd.read_csv(save_file_path)
        # append below
        res_summary = pd.concat([old_summary, res_summary], axis=0)

    # save final and delete tmp file
    res_summary.to_csv(save_file_path, index=False)
    os.remove(tmp_file_path)


def clean_up_midlogs(save_file_path, folder_to_clean, ext='.log'):
    """
    open python in cmd line from root folder, run the following:

    from src.utils.logger import clean_up_midlogs
    clean_up_midlogs('outs/results.csv', 'outs')
    """

    success_files = list(pd.read_csv(save_file_path)['logfile'])

    files = [os.path.join(path, name) for path, subdirs, files in os.walk(folder_to_clean) for name in files] 
    files_to_remove = [f for f in files if re.sub('\\\\', '/', f)  not in success_files and f[-4:]==ext]
    print(files_to_remove)

    approval = input('Approve deletion? Yes (y) or No (n)')
    if approval.lower() in ['y', 'yes']:
        for f in files_to_remove:
            os.remove(f)
    


import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add missing columns required by UniCausal format."
    )
    parser.add_argument(
        "--input_csv_path", type=str, default=None, help="A csv or a json file containing the data."
    )
    parser.add_argument(
        "--suffix", type=str, default='_formatted', help="Suffix to save the new data."
    )
    args = parser.parse_args()
    return args


def add_dummy_columns():
    args = parse_args()

    data = pd.read_csv(args.input_csv_path)
    dummy_values = {
        'corpus': 'dcorpus',
        'doc_id': 'ddocid',
        'sent_id': range(len(data)),
        'eg_id': 0,
        'index': [f'dcorpus_ddocid_{i}_0' for i in range(len(data))],
        'context': '',
        'num_sents': 0
    }
    tricky_defaults = {
        'text': '',
        'text_w_pairs': '',
        'seq_label': 1,
        'pair_label': 1
    }

    missing_columns = [c for c in list(dummy_values.keys()) if c not in data.columns]
    for col in missing_columns:
        print(f'Adding col "{col}"...')
        data[col] = dummy_values[col]

    missing_columns = [c for c in list(tricky_defaults.keys()) if c not in data.columns]
    for col in missing_columns:
        print(f'You are missing a key col "{col}", we try our best to add defaults, but you should check the column naming first.')
        if col=='text_w_pairs' and 'text' in data.columns:
            data[col] = data['text']
        else:
            data[col] = tricky_defaults[col]

    file_name, ext = os.path.splitext(args.input_csv_path)
    data.to_csv(file_name+args.suffix+ext, index=False)
    print('Complete!')


if __name__ == "__main__":
    add_dummy_columns()
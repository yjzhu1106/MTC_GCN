import os
import pickle
import argparse
import numpy as np
import pandas as pd

from dataloading import RAF_DB_annotation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle images annotations')

    parser.add_argument('--csv_dir',
                        default= '/root/autodl-tmp/RAF-DB/data',
                        help='path to annotations of the dataset', required=True)

    args = parser.parse_args()

    df_train = pd.read_csv(os.path.join(args.csv_dir, 'raf_train.csv'))

    df_train = df_train[df_train['expression'] < 7]

    collumns = ["subDirectory_filePath", "expression", "valence", "arousal"]
    df_val = pd.read_csv(os.path.join(
        args.csv_dir, 'raf_test.csv'))
    df_val = df_val[df_val['expression'] < 7]

    data_train = []
    data_val = []

    for index, row in df_train.iterrows():

        sample = RAF_DB_annotation(
            row['subDirectory_filePath'], row['expression'], row['valence'], row['arousal'])
        data_train.append(sample)

    for index, row in df_val.iterrows():

        sample = RAF_DB_annotation(
            row['subDirectory_filePath'], row['expression'], 0, 0)
        data_val.append(sample)

    data = {'train': data_train, 'val': data_val}

    with open('/root/autodl-tmp/RAF-DB/data/data_raf_db.pkl', "wb") as w:
        pickle.dump(data, w)

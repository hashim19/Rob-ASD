import numpy as np
from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, arange, meshgrid, ceil, linspace
import math
import pandas as pd
import soundfile as sf
import librosa
import os
import pickle
import pickle_blosc
from itertools import repeat
from multiprocessing.pool import Pool
from functools import partial

import sys
sys.path.append("../")

import config as config

from .feature_functions import extract_cqcc, extract_lfcc, extract_mfcc


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def extract_features(file, features, data_type='train', data_label='bonafide', feat_root='Features', cached=False):

    def get_feats():
        # data, samplerate = sf.read(file)

        print(file)
        data, samplerate = librosa.load(file, sr=None)

        print(data.shape)

        print(samplerate)

        if features == 'cqcc':
            return extract_cqcc(data, samplerate)

        if features == 'lfcc':
            return extract_lfcc(data, samplerate)

        if features == 'mfcc': 
            return extract_mfcc(data, samplerate) 

        else:
            return None

    if cached:

        if data_type == 'train':

            feat_dir = os.path.join(feat_root, features + '_features', data_type, data_label)
        
        else:

            feat_dir = os.path.join(feat_root, features + '_features', data_type)

        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        feat_file = file.split('.')[0].split('/')[-1] + '.pkl'

        feat_fullfile = feat_dir + '/' + feat_file

        # print(feat_fullfile)

        if not os.path.exists(feat_fullfile):

            feat_data = get_feats()

            with open(feat_fullfile, 'wb') as f:
                pickle.dump(feat_data, f)

        else:

            with open(feat_fullfile, 'rb') as f:
                feat_data = pickle.load(f)
        
        return feat_data

    return get_feats()


if __name__ == "__main__":

    db_type = config.db_type
    db_folder = config.db_folder
    data_names = config.data_names

    laundering_type = config.laundering_type
    laundering_param = config.laundering_param
    protocol_filenames = config.protocol_filenames

    # if db_type == 'in_the_wild':
    #     data_dir = [os.path.join(db_folder, 'release_in_the_wild')]
    # elif db_type == 'asvspoof_eval_laundered':
    #     data_dir = [os.path.join(db_folder, 'flac')]
    # elif db_type == 'asvspoof_train_laundered':
    
    # data_dirs = [os.path.join(db_folder, dt_name) for dt_name in data_names]

    # print(data_dirs)

    # protocol_paths = [os.path.join(db_folder, 'protocols', pf) for pf in protocol_filenames] 

    # print(protocol_paths)

    Feat_dir = os.path.join(config.feat_dir, laundering_type, laundering_param)

    features = config.feature_type

    audio_ext = config.audio_ext
    data_types = config.data_types
    data_labels = config.data_labels


    # extract features and save them

    for data_name, protocol_filename in zip(data_names, protocol_filenames):

        protocol_path = os.path.join(db_folder, 'protocols', protocol_filename)

        print(protocol_path)

        for data_type in data_types:

            if data_type == 'train':
                data_dir = os.path.join(db_folder, data_name, 'flac')

                for data_label in data_labels:

                    if data_name == 'ASVspoof2019_LA_train' or data_name == 'ASVspoof2019_LA_dev':
                        df = pd.read_csv(protocol_path, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_For_LA", "SYSTEM_ID", "KEY"])
                        df = df.loc[df['KEY'] == data_label]
                        files = df["AUDIO_FILE_NAME"].values

                    elif data_name == 'ASVSpoofData_2019_train_10_percent_laundered':
                        df = pd.read_csv(protocol_path, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
                        df = df.loc[df['KEY'] == data_label]
                        files = df["AUDIO_FILE_NAME"].values
                    
                    print(df.head())

                    print("{} data size is {}".format(data_label, files.shape))

                    args_iter = list(repeat(data_dir + '/' + files + audio_ext, 1))

                    with Pool(processes=6) as pool:

                        for result in pool.imap(partial(extract_features, features=features, data_type=data_type, data_label=data_label, feat_root=Feat_dir, cached=True), *args_iter, chunksize=1000):

                            print(f'Got result: {result.shape}', flush=True)

                    # for nf, file in enumerate(files):
                    #     Tx = extract_features(data_dir + file + audio_ext, features=features, data_label=data_label,
                    #         data_type=data_type, feat_root=Feat_dir, cached=True)
                    #     print(Tx.shape)

            if data_type == 'eval':

                data_dir = os.path.join(db_folder, 'flac')

                print(protocol_path)

                if db_type == 'in_the_wild':
                    df = pd.read_csv(protocol_path, sep=',', names=["AUDIO_FILE_NAME", "Speaker_Id", "KEY"])
                    df = df.iloc[1:,:]
                    files = df["AUDIO_FILE_NAME"].values
                    print(df)
                    print(len(files))

                    args_iter = list(repeat(data_dir + '/' + files, 1))

                elif db_type == 'asvspoof_eval_laundered':
                    df = pd.read_csv(protocol_path, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
                    df = df[df["Laundering_Param"] == laundering_param]
                    files = df["AUDIO_FILE_NAME"].values
                    print(df)
                    print(len(files))

                    args_iter = list(repeat(data_dir + '/' + files + audio_ext, 1))
                
                print(*args_iter)

                with Pool(processes=6) as pool:

                    # starmap_with_kwargs(pool, extract_features, args_iter, kwargs_iter)

                    # results = pool.imap(partial(extract_features, features=features, data_type=data_type, feat_root=Feat_dir, cached=True), *args_iter, chunksize=10)

                    for result in pool.imap(partial(extract_features, features=features, data_type=data_type, feat_root=Feat_dir, cached=True), *args_iter, chunksize=1000):

                        print(f'Got result: {result.shape}', flush=True)

                    # print(results)
                    





            
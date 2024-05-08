import numpy as np
from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, arange, meshgrid, ceil, linspace
import math
import pandas as pd
import soundfile as sf
import librosa
import os
import pickle
from itertools import repeat
from multiprocessing.pool import Pool
from functools import partial

import config as config

from feature_functions import extract_cqcc, extract_lfcc, extract_mfcc


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

    db_folder = config.db_folder

    laundering_type = config.laundering_type
    laundering_param = config.laundering_param
    protocol_pth = config.protocol_filename

    data_dir = os.path.join(db_folder, 'flac')
    protocol_path = os.path.join(db_folder, 'protocols', protocol_pth)

    Feat_dir = os.path.join(config.feat_dir, laundering_type, laundering_param)

    features = config.feature_type

    audio_ext = config.audio_ext
    data_types = config.data_types
    data_labels = config.data_labels


    # extract features and save them
    for data_type in data_types:

        if data_type == 'train':

            for data_label in data_labels:

                df = pd.read_csv(protocol_path, sep=' ', header=None)
                files = df[df[4] == data_label][1]
                print("{} data size is {}".format(data_label, files.shape))

                for nf, file in enumerate(files):
                    Tx = extract_features(data_dir + file + audio_ext, features=features, data_label=data_label,
                        data_type=data_type, feat_root=Feat_dir, cached=True)
                    print(Tx.shape)

        if data_type == 'eval':

            print(protocol_path)

            df = pd.read_csv(protocol_path, sep=' ', header=None)
            # df = pd.read_csv(protocol_path, sep=',', header=None)
            # df = df.drop(0)
            # print(df)
            df = df[df[5] == laundering_param]
            print(df)
            files = df[1].values

            print(len(files))
            args_iter = list(repeat(data_dir + '/' + files + audio_ext, 1))
            
            print(*args_iter)

            with Pool(processes=6) as pool:

                # starmap_with_kwargs(pool, extract_features, args_iter, kwargs_iter)

                # results = pool.imap(partial(extract_features, features=features, data_type=data_type, feat_root=Feat_dir, cached=True), *args_iter, chunksize=10)

                for result in pool.imap(partial(extract_features, features=features, data_type=data_type, feat_root=Feat_dir, cached=True), *args_iter, chunksize=1000):

                    print(f'Got result: {result.shape}', flush=True)

                # print(results)
                    





            
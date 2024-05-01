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

    db_folder = '/data/Data/'        # '/home/alhashim/Data/AsvSpoofData_2019/train/'

    # laundering_type = 'Noise_Addition/'
    # laundering = 'AsvSpoofData_2019_street_10_10_5/'
    # protocol_pth = 'street_10_10_5_protocol.txt'

    # laundering_type = 'Reverberation/'
    # laundering = 'AsvSpoofData_2019_RT_0_9/'
    # protocol_pth = 'Protocol_ASV_RT_0_9.txt'

    # laundering_type = 'Recompression/'
    # laundering = 'recompression_320k/'
    # protocol_pth = 'recompression_protocol_320k.txt'

    # laundering_type = 'Resampling/'
    # laundering = 'resample_11025/'
    # protocol_pth = 'resample_11025.txt'

    laundering_type = 'Filtering/'
    laundering = 'low_pass_filt_7000/'
    protocol_pth = 'low_pass_filt_7000_protocol.txt'

    # laundering_type = 'Transcoding/'
    # laundering = 'Asvspoof19_40_audio_facebook/'
    # protocol_pth = 'Asvspoof19_40_protocol.csv'

    data_dirs = [db_folder + laundering_type + laundering]  # [db_folder + 'LA/ASVspoof2019_LA_train/flac/', db_folder + 'LA/ASVspoof2019_LA_dev/flac/'] db_folder + 'ASVspoof2019_LA_eval/flac/'
    protocol_paths = [db_folder + 'AsvSpoofData_2019_protocols/' + protocol_pth]  # [db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', db_folder + 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trn.txt'] 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    # protocol_paths = [db_folder + laundering_type + 'ASVspoof2019_LA_cm_protocols/' + protocol_pth]

    # Feat_dir = 'features_out'
    Feat_dir = os.path.join('/data/Features/', laundering_type, laundering)
    # Feat_dir = os.path.join('/data/Features/', laundering)

    audio_ext = '.wav'

    data_types = ['eval']

    data_labels = ['bonafide', 'spoof']
    # data_labels = ['bonafide']

    features = 'cqcc'


    # extract features and save them
    for k, protocol_path in enumerate(protocol_paths):

        for data_type in data_types:

            if data_type == 'train':

                for data_label in data_labels:

                    df = pd.read_csv(protocol_path, sep=' ', header=None)
                    files = df[df[4] == data_label][1]
                    print("{} data size is {}".format(data_label, files.shape))

                    for nf, file in enumerate(files):
                        Tx = extract_features(data_dirs[k] + file + audio_ext, features=features, data_label=data_label,
                            data_type=data_type, feat_root=Feat_dir, cached=True)
                        print(Tx.shape)

            if data_type == 'eval':

                df = pd.read_csv(protocol_path, sep=' ', header=None)
                # df = pd.read_csv(protocol_path, sep=',', header=None)
                # df = df.drop(0)
                print(df)
                files = df[1].values

                # for nf, file in enumerate(files):
                #     print(file)
                #     # if not file == 'LA_E_8070617_RT_0_6':
                #     # if not ( file == 'LA_E_2800292_RT_0_9' or file == 'LA_E_1573988_RT_0_9' ):
                #     Tx = extract_features(data_dirs[k] + file + audio_ext, features=features,
                #         data_type=data_type, feat_root=Feat_dir, cached=True)
                #     print(Tx.shape)

                print(len(files))
                args_iter = list(repeat(data_dirs[k] + files + audio_ext, 1))
                
                print(*args_iter)
                # print(len(*args_iter))

                # kwargs_iter = repeat(dict(features=features, data_type=data_type, feat_root=Feat_dir, cached=True), len(files))
                # kwargs_iter = dict(features=features, data_type=data_type, feat_root=Feat_dir, cached=True)
                # print(len(kwargs_iter))

                # print(*kwargs_iter)

                with Pool(processes=6) as pool:

                    # starmap_with_kwargs(pool, extract_features, args_iter, kwargs_iter)

                    # results = pool.imap(partial(extract_features, features=features, data_type=data_type, feat_root=Feat_dir, cached=True), *args_iter, chunksize=10)

                    for result in pool.imap(partial(extract_features, features=features, data_type=data_type, feat_root=Feat_dir, cached=True), *args_iter, chunksize=1000):

                        print(f'Got result: {result.shape}', flush=True)

                    # print(results)
                    





            
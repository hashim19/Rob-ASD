import os
import pandas as pd
import librosa
import numpy as np
from scipy.signal import lfilter
import pickle

from LFCC_pipeline import lfcc


def Deltas(x, width=3):
    hlen = int(np.floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = np.tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = np.tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]

def extract_lfcc(audio_data, sr, num_ceps=20, order_deltas=2, low_freq=None, high_freq=None, win_len=20, NFFT = 512, no_of_filter=20):

    lfccs = lfcc(sig=audio_data,
                 fs=sr,
                 num_ceps=num_ceps,
                 win_len=win_len/1000,
                 nfft=NFFT).T
    
    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = np.vstack(feats)

  
    return lfccs


if __name__ == "__main__":

    ################### For Asvspoof 2019 Original Data ###############
    #  set here the experiment to run (access and feature type)
    # access_type = 'LA'
    # feature_type = 'LFCC'

    # #  set paths to the wave files and protocols
    # pathToASVspoof2019Data = '/data/Data/'
    # pathToFeatures = os.path.join('/data/Features', 'AsvSpoofData_2019_RT_0_3', 'lfcc_features_new')

    # # pathToDatabase = os.path.join(pathToASVspoof2019Data, access_type)
    # pathToDatabase = os.path.join(pathToASVspoof2019Data, )
    # trainProtocolFile = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_cm_protocols', 'ASVspoof2019.' + access_type + '.cm.train.trn.txt')
    # devProtocolFile = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_cm_protocols', 'ASVspoof2019.' + access_type + '.cm.dev.trl.txt')
    # evalProtocolFile = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_cm_protocols', 'ASVspoof2019.' + access_type + '.cm.eval.trl.txt')

    # print(pathToDatabase)
    # print(trainProtocolFile)
    # print(devProtocolFile)
    # print(evalProtocolFile)

    ############ For post processed Data ###########
    db_folder = '/data/Data/'  # put your database root path here

    # laundering_type = 'Noise_Addition/'
    # laundering = 'AsvSpoofData_2019_WN_20_20_5'
    # protocol_pth = 'white_20_20_5_protocol.txt'
    

    # laundering_type = 'Reverberation/'
    # laundering = 'AsvSpoofData_2019_RT_0_9'
    # protocol_pth = 'Protocol_ASV_RT_0_9.txt'

    # laundering_type = 'Resampling/'
    # laundering = 'resample_44100/'
    # protocol_pth = 'resample_44100.txt'

    laundering_type = 'Filtering/'
    laundering = 'low_pass_filt_7000/'
    protocol_pth = 'low_pass_filt_7000_protocol.txt'
    
    pathToDatabase = db_folder + laundering_type + laundering + '/'
    evalProtocolFile = db_folder + 'AsvSpoofData_2019_protocols/' + protocol_pth
    pathToFeatures = os.path.join('/data/Features/', laundering_type, laundering, 'lfcc_features_airasvspoof')


    # # read train protocol
    # trainprotocol = pd.read_csv(trainProtocolFile, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])
    # trainfilelist = trainprotocol["AUDIO_FILE_NAME"].to_list()

    # # read dev protocol
    # devprotcol = pd.read_csv(devProtocolFile, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])
    # devfilelist = devprotcol["AUDIO_FILE_NAME"].to_list()

    # read eval protocol
    evalprotcol = pd.read_csv(evalProtocolFile, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])
    evalfilelist = evalprotcol["AUDIO_FILE_NAME"].to_list()

    ############ Feature extraction for training data ##############

    # # extract features for training data and store them
    # print('Extracting features for training data...')

    # LFCC_sav_dir = os.path.join(pathToFeatures, 'train')

    # if not os.path.exists(LFCC_sav_dir):
    #     os.makedirs(LFCC_sav_dir)

    # lfcc_features_ls = []
    # for file in trainfilelist:
    #     print(file)

    #     audio_file = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_train/flac', file + '.flac')

    #     x, fs = librosa.load(audio_file)
        
    #     lfcc_featues = extract_lfcc(x, fs)

    #     print(lfcc_featues.shape)

    #     LFCC_filename = os.path.join(LFCC_sav_dir, 'LFCC_' + file + '.pkl')

    #     if not os.path.exists(LFCC_filename):

    #         with open(LFCC_filename, 'wb') as f:
    #             pickle.dump(lfcc_featues, f)

    # print("Done")


    # ############ Feature extraction for Development data ##############

    # # extract features for development data and store them
    # print('Extracting features for development data...')

    # LFCC_sav_dir = os.path.join(pathToFeatures, 'dev')

    # if not os.path.exists(LFCC_sav_dir):
    #     os.makedirs(LFCC_sav_dir)

    # lfcc_features_ls = []
    # for file in devfilelist:

    #     audio_file = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_dev/flac', file + '.flac')

    #     x, fs = librosa.load(audio_file)
        
    #     lfcc_featues = extract_lfcc(x, fs)

    #     print(lfcc_featues.shape)

    #     LFCC_filename = os.path.join(LFCC_sav_dir, 'LFCC_' + file + '.pkl')

    #     if not os.path.exists(LFCC_filename):

    #         with open(LFCC_filename, 'wb') as f:
    #             pickle.dump(lfcc_featues, f)

    # print("Done")


    ############ Feature extraction for Evaluation data ##############

    # extract features for evaluation data and store them
    print('Extracting features for evaluation data...')

    LFCC_sav_dir = os.path.join(pathToFeatures, 'eval')

    if not os.path.exists(LFCC_sav_dir):
        os.makedirs(LFCC_sav_dir)

    lfcc_features_ls = []
    for file in evalfilelist:

        # audio_file = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_eval/flac', file + '.flac')
        audio_file = os.path.join(pathToDatabase, file + '.wav')

        x, fs = librosa.load(audio_file)
        
        lfcc_featues = extract_lfcc(x, fs)

        print(lfcc_featues.shape)

        LFCC_filename = os.path.join(LFCC_sav_dir, file + '.pkl')

        if not os.path.exists(LFCC_filename):

            with open(LFCC_filename, 'wb') as f:
                pickle.dump(lfcc_featues, f)

    print("Done")







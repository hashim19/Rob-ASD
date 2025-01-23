import os
import pandas as pd
import librosa
import numpy as np
from scipy.signal import lfilter
import pickle
import h5py

import sys
sys.path.append("../")
import config as config
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
    db_folder = config.db_folder  # put your database root path here
    db_type = config.db_type
    data_names = config.data_names

    laundering_type = config.laundering_type
    laundering_param = config.laundering_param
    protocol_filenames = config.protocol_filenames

    Feat_dir = os.path.join(config.feat_dir, laundering_type, laundering_param, 'lfcc_features_airasvspoof')

    features = config.feature_type

    audio_ext = config.audio_ext
    data_types = config.data_types
    data_labels = config.data_labels
    
    # if db_type == 'in_the_wild':
    #     pathToDatabase = os.path.join(db_folder, 'release_in_the_wild')
    # elif db_type == 'asvspoof':
    #     pathToDatabase = os.path.join(db_folder, 'flac')

    # evalProtocolFile = os.path.join(db_folder, 'protocols', protocol_filenames)
    # pathToFeatures = os.path.join(config.feat_dir, laundering_type, laundering_param, 'lfcc_features_airasvspoof')

    audio_ext = config.audio_ext

    for data_name, protocol_filename, data_type in zip(data_names, protocol_filenames, data_types):

        print(data_name)
        print(protocol_filename)
        print(data_type)

        # read protocol file
        if data_type == 'eval':

            evalProtocolFile = os.path.join(db_folder, 'protocols', protocol_filename)

            # read eval protocol
            if db_type == 'in_the_wild':
                pathToDatabase = os.path.join(db_folder, 'release_in_the_wild')

                evalprotcol = pd.read_csv(evalProtocolFile, sep=',', names=["AUDIO_FILE_NAME", "Speaker_Id", "KEY"])
                filelist = evalprotcol["AUDIO_FILE_NAME"].to_list()

            elif db_type == 'asvspoof_eval_laundered':
                pathToDatabase = os.path.join(db_folder, 'flac')

                evalprotcol = pd.read_csv(evalProtocolFile, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
                
                # create a temporary protocol file, this file will be used by test.py
                evalprotcol_tmp = evalprotcol.loc[evalprotcol['Laundering_Param'] == laundering_param]
                evalprotcol_tmp = evalprotcol_tmp[["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY"]]
                evalprotcol_tmp.to_csv(os.path.join(db_folder, 'protocols', protocol_filename.split('.')[0] + '_' 'tmp.txt'), header=False, index=False, sep=" ")

                filelist = evalprotcol_tmp["AUDIO_FILE_NAME"].to_list()

            elif db_type == 'asvspoof_eval':
                pathToDatabase = os.path.join(db_folder, 'flac')

                evalprotcol = pd.read_csv(evalProtocolFile, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])
                
                filelist = evalprotcol["AUDIO_FILE_NAME"].to_list()

            elif db_type == 'asvspoof5_eval_laundered':
                pathToDatabase = os.path.join(db_folder, data_name, 'flac')

                evalprotcol = pd.read_csv(evalProtocolFile, sep=' ', names=["AUDIO_FILE_NAME", "KEY"])
                
                filelist = evalprotcol["AUDIO_FILE_NAME"].to_list()


        elif data_type == 'train' or data_type == 'dev':

            pathToDatabase = os.path.join(db_folder, data_name, 'flac')
            # pathToDatabase = os.path.join(db_folder, 'flac')

            protocol_file_path = os.path.join(db_folder, 'protocols', protocol_filename)

            if db_type == 'asvspoof5_train_laundered':
                protocol_df = pd.read_csv(protocol_file_path, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Gender", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
            
            else:
                protocol_df = pd.read_csv(protocol_file_path, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])

            print(protocol_df)

            filelist = protocol_df["AUDIO_FILE_NAME"].to_list()

        ############ Feature extraction for Evaluation data ##############

        # extract features for evaluation data and store them
        print('Extracting features for {} data...'.format(data_type))

        LFCC_sav_dir = os.path.join(Feat_dir, data_type)

        # for h5py file
        h5_file = Feat_dir + '.h5'

        print(h5_file)

        if not os.path.exists(LFCC_sav_dir):
            os.makedirs(LFCC_sav_dir)

        if not os.path.exists(h5_file):

            lfcc_features_ls = []
            for file in filelist:

                LFCC_filename = os.path.join(LFCC_sav_dir, str(file) + '.pkl')

                if not os.path.exists(LFCC_filename):

                    # audio_file = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_eval/flac', file + '.flac')
                    audio_file = os.path.join(pathToDatabase, str(file) + audio_ext)

                    x, fs = librosa.load(audio_file)
                    
                    lfcc_featues = extract_lfcc(x, fs)

                    print(lfcc_featues.shape)

                    with open(LFCC_filename, 'wb') as f:
                        pickle.dump(lfcc_featues, f)

                else:

                    print("Feature file {} already extracted".format(file))

        else:
            with h5py.File(h5_file, 'r') as h5f:
                # for filename, feature_mat in h5f.items():
                    # print(filename)
                
                # print("Keys: %s" % h5f.keys())
                key = list(h5f.keys())[0]
                print(key)
                data = h5f[key][()]
                print(data.shape)
                # lfcc_features = h5f['LA_E_1524908_RT_0_3']
                # print(lfcc_featues)
            print("Features were already extracted and were saved in {} ".format(h5_file))

        print("Done")



    # # read train protocol
    # trainprotocol = pd.read_csv(trainProtocolFile, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])
    # trainfilelist = trainprotocol["AUDIO_FILE_NAME"].to_list()

    # # read dev protocol
    # devprotcol = pd.read_csv(devProtocolFile, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])
    # devfilelist = devprotcol["AUDIO_FILE_NAME"].to_list()

    # # read eval protocol
    # if db_type == 'in_the_wild':
    #     evalprotcol = pd.read_csv(evalProtocolFile, sep=',', names=["AUDIO_FILE_NAME", "Speaker_Id", "KEY"])
    #     evalfilelist = evalprotcol["AUDIO_FILE_NAME"].to_list()

    # elif db_type == 'asvspoof':
    #     evalprotcol = pd.read_csv(evalProtocolFile, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
        
    #     # create a temporary protocol file, this file will be used by test.py
    #     evalprotcol_tmp = evalprotcol.loc[evalprotcol['Laundering_Param'] == laundering_param]
    #     evalprotcol_tmp = evalprotcol_tmp[["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY"]]
    #     evalprotcol_tmp.to_csv(os.path.join(db_folder, 'protocols', protocol_pth.split('.')[0] + '_' 'tmp.txt'), header=False, index=False, sep=" ")

    #     evalfilelist = evalprotcol_tmp["AUDIO_FILE_NAME"].to_list()

    # ############ Feature extraction for training data ##############

    # # # extract features for training data and store them
    # # print('Extracting features for training data...')

    # # LFCC_sav_dir = os.path.join(pathToFeatures, 'train')

    # # if not os.path.exists(LFCC_sav_dir):
    # #     os.makedirs(LFCC_sav_dir)

    # # lfcc_features_ls = []
    # # for file in trainfilelist:
    # #     print(file)

    # #     audio_file = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_train/flac', file + '.flac')

    # #     x, fs = librosa.load(audio_file)
        
    # #     lfcc_featues = extract_lfcc(x, fs)

    # #     print(lfcc_featues.shape)

    # #     LFCC_filename = os.path.join(LFCC_sav_dir, 'LFCC_' + file + '.pkl')

    # #     if not os.path.exists(LFCC_filename):

    # #         with open(LFCC_filename, 'wb') as f:
    # #             pickle.dump(lfcc_featues, f)

    # # print("Done")


    # # ############ Feature extraction for Development data ##############

    # # # extract features for development data and store them
    # # print('Extracting features for development data...')

    # # LFCC_sav_dir = os.path.join(pathToFeatures, 'dev')

    # # if not os.path.exists(LFCC_sav_dir):
    # #     os.makedirs(LFCC_sav_dir)

    # # lfcc_features_ls = []
    # # for file in devfilelist:

    # #     audio_file = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_dev/flac', file + '.flac')

    # #     x, fs = librosa.load(audio_file)
        
    # #     lfcc_featues = extract_lfcc(x, fs)

    # #     print(lfcc_featues.shape)

    # #     LFCC_filename = os.path.join(LFCC_sav_dir, 'LFCC_' + file + '.pkl')

    # #     if not os.path.exists(LFCC_filename):

    # #         with open(LFCC_filename, 'wb') as f:
    # #             pickle.dump(lfcc_featues, f)

    # # print("Done")


    # ############ Feature extraction for Evaluation data ##############

    # # extract features for evaluation data and store them
    # print('Extracting features for evaluation data...')

    # LFCC_sav_dir = os.path.join(pathToFeatures, 'eval')

    # if not os.path.exists(LFCC_sav_dir):
    #     os.makedirs(LFCC_sav_dir)

    # lfcc_features_ls = []
    # for file in evalfilelist:

    #     # audio_file = os.path.join(pathToDatabase, 'ASVspoof2019_' + access_type + '_eval/flac', file + '.flac')
    #     audio_file = os.path.join(pathToDatabase, str(file) + audio_ext)

    #     x, fs = librosa.load(audio_file)
        
    #     lfcc_featues = extract_lfcc(x, fs)

    #     print(lfcc_featues.shape)

    #     LFCC_filename = os.path.join(LFCC_sav_dir, str(file) + '.pkl')

    #     if not os.path.exists(LFCC_filename):

    #         with open(LFCC_filename, 'wb') as f:
    #             pickle.dump(lfcc_featues, f)

    # print("Done")







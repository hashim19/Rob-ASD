import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
import pandas as pd
from torch.utils.data.dataloader import default_collate
import h5py

torch.set_default_tensor_type(torch.FloatTensor)

import sys
sys.path.append("../")

import config as config

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, path_to_protocol, part='train', feature='LFCC',
                 genuine_only=False, feat_len=750, padding='repeat'):
        self.access_type = access_type
        # self.ptd = path_to_database
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        # self.ptf = path_to_features
        # self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        if self.part == 'train':
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trn.txt')
        else:
            # protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
            protocol = path_to_protocol
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        # speaker, filename, tag, label = self.all_info[idx]
        # try:
        #     with open(self.ptf + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
        #         feat_mat = pickle.load(feature_handle)
        # except:
        #     # add this exception statement since we may change the data split
        #     def the_other(train_or_dev):
        #         assert train_or_dev in ["train", "dev"]
        #         res = "dev" if train_or_dev == "train" else "train"
        #         return res
        #     with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
        #         feat_mat = pickle.load(feature_handle)

        try:
            # with open(self.ptf + '/'+ self.feature + '_' + filename + '.pkl', 'rb') as feature_handle:
            #     feat_mat = pickle.load(feature_handle)
            with open(self.ptf + '/' + filename + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            def the_other(train_or_dev):
                assert train_or_dev in ["train", "dev"]
                res = "dev" if train_or_dev == "train" else "train"
                return res
            with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)

        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return feat_mat, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)


class InTheWild(Dataset):
    def __init__(self, path_to_features, path_to_protocol, part='train', feature='LFCC',
                 genuine_only=False, feat_len=750, padding='repeat'):
        
        # self.ptd = path_to_database
        self.access_type = 'LA'
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        # self.ptf = path_to_features
        # self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        protocol = path_to_protocol

        self.label = {"spoof": 1, "bonafide": 0}
        self.tag = {"-": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split(',') for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)
    

    def __getitem__(self, idx):
        # speaker, filename, _, tag, label = self.all_info[idx]
        filename, speaker, label = self.all_info[idx]

        try:
            # with open(self.ptf + '/'+ self.feature + '_' + filename + '.pkl', 'rb') as feature_handle:
            #     feat_mat = pickle.load(feature_handle)
            with open(self.ptf + '/' + filename + '.pkl', 'rb') as feature_handle:
                feat_mat = pickle.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            print("Can't load feature file {}".format(filename))

        
        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return feat_mat, filename, self.tag['-'], self.label[label]
    

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoofLaundered(Dataset):
    def __init__(self, path_to_features, path_to_protocol, protocol_filenames, part='train', feature='LFCC',
                 genuine_only=False, feat_len=750, padding='repeat', feature_format='.pkl', num_columns=7):
        
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.path_to_protocol = path_to_protocol
        self.protocol_filenames = protocol_filenames
        self.padding = padding
        self.feature_format = feature_format
        self.num_columns = num_columns

        protocol_paths = []
        if config.db_type == 'asvspoof_train_laundered':
            for idx, pf in enumerate(self.protocol_filenames):
                if config.data_types[idx] == self.part:
                    protocol_paths.append(os.path.join(self.path_to_protocol, pf))

        elif config.db_type == 'asvspoof_eval_laundered':
            protocol_paths = [self.path_to_protocol]


        # protocol_paths = [os.path.join(self.path_to_protocol, pf) for idx, pf in enumerate(self.protocol_filenames) if config.data_types[idx] == self.part]

        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                    "A19": 19}
        
        self.label = {"spoof": 1, "bonafide": 0}

        # print(protocol_paths)
        
        protocol_df_ls = []

        for protocol in protocol_paths:

            if config.db_type == 'asvspoof_train_laundered':
                protocol_df = pd.read_csv(protocol, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
            
            elif config.db_type == 'asvspoof_eval_laundered':
                protocol_df = pd.read_csv(protocol, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])

            if genuine_only:
                assert self.part in ["train", "dev"]
                protocol_df = protocol_df.loc[protocol_df['KEY'] == 'bonafide']
            # print(protocol_df.shape)
            protocol_df_ls.append(protocol_df)

        self.protocol_df = pd.concat(protocol_df_ls)


    def __len__(self):
        return len(self.protocol_df)
    

    def __getitem__(self, idx):
        # speaker, filename, _, tag, label = self.all_info[idx]
        if config.db_type == 'asvspoof_train_laundered':
            speaker, filename, _, tag, label, ld, lp = self.protocol_df.iloc[idx]
        
        elif config.db_type == 'asvspoof_eval_laundered':
            speaker, filename, tag, label, ld, lp = self.protocol_df.iloc[idx]

        try:
            if self.feature_format == 'h5f':
                h5_file = self.path_to_features + '.h5'
                
                with h5py.File(h5_file, 'r') as h5f:
                    feat_mat = h5f[filename][()]

            elif self.feature_format == 'pkl':
                print(self.ptf + '/' + filename + '.pkl')
                with open(self.ptf + '/' + filename + '.pkl', 'rb') as feature_handle:
                    feat_mat = pickle.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            print("Can't load feature file {}".format(filename))

        
        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        return feat_mat, filename, self.tag['-'], self.label[label]
    

    def collate_fn(self, samples):
        return default_collate(samples)
    

class ASVspoof5Laundered(Dataset):
    def __init__(self, path_to_features, path_to_protocol, protocol_filenames, part='train', feature='LFCC',
                 genuine_only=False, feat_len=750, padding='repeat', feature_format='pkl', num_columns=7):
        
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.genuine_only = genuine_only
        self.feat_len = feat_len
        self.feature = feature
        self.path_to_protocol = path_to_protocol
        self.protocol_filenames = protocol_filenames
        self.padding = padding
        self.feature_format = feature_format
        self.num_columns = num_columns

        protocol_paths = []
        if config.db_type == 'asvspoof5_train_laundered':
            for idx, pf in enumerate(self.protocol_filenames):
                if config.data_types[idx] == self.part:
                    protocol_paths.append(os.path.join(self.path_to_protocol, pf))

        elif config.db_type == 'asvspoof5_eval_laundered':
            protocol_paths = [self.path_to_protocol]


        # protocol_paths = [os.path.join(self.path_to_protocol, pf) for idx, pf in enumerate(self.protocol_filenames) if config.data_types[idx] == self.part]

        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                    "A19": 19}
        
        self.label = {"spoof": 1, "bonafide": 0}

        # print(protocol_paths)
        
        protocol_df_ls = []

        for protocol in protocol_paths:

            if config.db_type == 'asvspoof5_train_laundered':
                protocol_df = pd.read_csv(protocol, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "Gender", "Not_Used_For_LA", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
            
            elif config.db_type == 'asvspoof5_eval_laundered':
                protocol_df = pd.read_csv(protocol, sep=' ', names=["AUDIO_FILE_NAME", "KEY"])

            if genuine_only:
                assert self.part in ["train", "dev"]
                protocol_df = protocol_df.loc[protocol_df['KEY'] == 'bonafide']
            # print(protocol_df.shape)
            protocol_df_ls.append(protocol_df)

        self.protocol_df = pd.concat(protocol_df_ls)

        # print(self.protocol_df)


    def __len__(self):
        return len(self.protocol_df)
    

    def __getitem__(self, idx):
        # speaker, filename, _, tag, label = self.all_info[idx]
        if config.db_type == 'asvspoof5_train_laundered':
            speaker, filename, _, _, tag, label, ld, lp = self.protocol_df.iloc[idx]
        
        elif config.db_type == 'asvspoof5_eval_laundered':
            filename, label = self.protocol_df.iloc[idx]

        try:
            if self.feature_format == 'h5f':
                h5_file = self.path_to_features + '.h5'
                
                with h5py.File(h5_file, 'r') as h5f:
                    feat_mat = h5f[filename][()]

            elif self.feature_format == 'pkl':
                # print(self.ptf + '/' + filename + '.pkl')
                with open(self.ptf + '/' + filename + '.pkl', 'rb') as feature_handle:
                    feat_mat = pickle.load(feature_handle)
        except:
            # add this exception statement since we may change the data split
            print("Can't load feature file {}".format(filename))

        
        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len-self.feat_len)
            feat_mat = feat_mat[:, startp:startp+self.feat_len]
        if this_feat_len < self.feat_len:
            if self.padding == 'zero':
                feat_mat = padding(feat_mat, self.feat_len)
            elif self.padding == 'repeat':
                feat_mat = repeat_padding(feat_mat, self.feat_len)
            else:
                raise ValueError('Padding should be zero or repeat!')

        if config.db_type == 'asvspoof5_eval_laundered':
            return feat_mat, filename
        else:
            return feat_mat, filename, self.tag['-'], self.label[label]
    

    def collate_fn(self, samples):
        return default_collate(samples)
    

def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


if __name__ == "__main__":
    # path_to_database = '/data/neil/DS_10283_3336/'  # if run on GPU
    # path_to_features = './LA/Features/'  # if run on GPU
    # path_to_protocol = '/home/hashim/PhD/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/'

    path_to_features = os.path.join(config.feat_dir, config.laundering_type, config.laundering_param, 'lfcc_features_airasvspoof')
    protocol_filenames = config.protocol_filenames
    path_to_protocol = os.path.join(config.db_folder, 'protocols')
    feat_len = 750
    padding = 'repeat'
    training_set = ASVspoofLaundered(path_to_features, path_to_protocol, protocol_filenames, 'train',
                                'LFCC', genuine_only = False, feat_len=feat_len, padding=padding)
    
    print(len(training_set))

    print(training_set.protocol_df.head)

    print(training_set[0])


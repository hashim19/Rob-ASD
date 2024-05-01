import numpy as np
import torch
import os

from gmm import GaussianMixture
from Asvspoof_dataset import open_pkl, PKL_dataset
from torch.utils.data import DataLoader


def train_gmm(data_label, features, train_keys, train_folders, audio_ext, dict_file, ncomp, feat_dir='features', init_only=False):

    print("Training GMM for {} data".format(data_label))

    # define gmm filename to save in
    gmm_save_file = '_'.join((dict_file, data_label, '.pkl'))

    # path to the dataset
    path_to_dataset = os.path.join(feat_dir, features + '_features', 'train', data_label)

    ############# load data using dataloader ################

    # train_data = PKL_dataset(path_to_dataset, data_label)

    # print(len(train_data))

    # train_dataloader = DataLoader(train_data, batch_size=len(train_data))

    # cqcc_features = next(iter(train_dataloader))

    # print("CQCC Feature shape {}".format(cqcc_features.size()))


    #############  load all data from pickle files into numpy array ##############
    # load pickle files
    total_files = os.listdir(path_to_dataset)

    # pickle all data into one file
    all_data_pickle_file = path_to_dataset + '.pkl'

    if not os.path.exists(all_data_pickle_file):

        for i, pkl_file in enumerate(total_files):

            pkl_data = open_pkl(os.path.join(path_to_dataset, pkl_file))

            print(i)

            if i == 0: 
                train_data = pkl_data
            else:
                train_data = np.vstack([train_data, pkl_data])

            # print(train_data.shape)
        
        with open(all_data_pickle_file, 'wb') as f:
            pickle.dump(train_data, f)

    else:

        train_data = open_pkl(all_data_pickle_file)

    train_data = torch.from_numpy(train_data)

    print(train_data.size())

    # for i, train_data in enumerate(train_dataloader):

    #     print(i)
    #     print(train_data.size())

    #     train_data = torch.flatten(train_data, start_dim=1)

    # print(train_data.size())

    gmm_model = GaussianMixture(ncomp, train_data.shape[1])

    gmm_model.cuda()
    
    history = gmm_model.fit(train_data)

    with open(gmm_save_file, 'wb') as m:
            pickle.dump(gmm_model, m)


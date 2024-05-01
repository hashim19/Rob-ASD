from torch.utils.data import Dataset, TensorDataset
import os
import pickle
import numpy as np
import torch

from torch.utils.data import DataLoader
# from lightkit.data import DataLoader


def open_pkl(pkl_path):

    with open(pkl_path, "rb") as f:
          d = pickle.load(f)
    return d.astype('float32')


def gmm_custom_collate(batch):

    c = torch.cat(batch, dim=0)

    return c


class PKL_dataset(Dataset):

    def __init__(self, dataset_pth, data_label):
        self.data_dir = dataset_pth
        self.files_ls = os.listdir(dataset_pth)
        self.len = len(self.files_ls)
        self.label = data_label

    def __len__(self):
        return self.len

    def transform(self, data):

        if data.shape[0] < 1000:

            return torch.from_numpy(np.pad(data, [(0, 1000 - data.shape[0]), (0,0)], 'mean'))

        else:

            return torch.from_numpy(data[:1000])

        return torch.from_numpy(data)

        # np.flatten(data)

    def __getitem__(self, idx):

        file_path = os.path.join(self.data_dir, self.files_ls[idx])
        
        pkl_data = open_pkl(file_path)
        transformed_pkl_data = self.transform(pkl_data)
        
        return transformed_pkl_data


if __name__ == "__main__":

    path_to_dataset = '/home/hashim/PhD/Audio_Spoof_Detection/Baseline-CQCC-GMM/python/cqcc_features/bonafide'

    cqcc_data = PKL_dataset(path_to_dataset, 'bonafide')

    pkl_dataloader = DataLoader(cqcc_data, batch_size=2, collate_fn=gmm_custom_collate)

    cqcc_features = next(iter(pkl_dataloader))

    print("CQCC Feature shape {}".format(cqcc_features.size()))

    # M = 20
    # D = 12
    # N = 30
    # a = torch.rand((N,D))
    # b = torch.rand((N,D))

    # c = gmm_custom_collate([a, b])

    # print(c.size())
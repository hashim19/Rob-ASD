import os
import pickle
import numpy as np
import pickle_blosc
import lzma
import mgzip
import gzip
import h5py

import shutil

feat_dir = '/data/Features/'

feat_out_dir = 'npy'

laundering_type = 'Noise_Addition'
# laundering_params = ['AsvSpoofData_2019_babble_0_0_5', 'AsvSpoofData_2019_babble_10_10_5', 'AsvSpoofData_2019_babble_20_20_5', 'AsvSpoofData_2019_cafe_0_0_5', 'AsvSpoofData_2019_cafe_10_10_5',
#                         'AsvSpoofData_2019_cafe_20_20_5', 'AsvSpoofData_2019_street_0_0_5', 'AsvSpoofData_2019_street_10_10_5', 'AsvSpoofData_2019_street_20_20_5',
#                         'AsvSpoofData_2019_volvo_0_0_5', 'AsvSpoofData_2019_volvo_10_10_5', '']

laundering_dir = os.path.join(feat_dir, laundering_type)
laundering_params = os.listdir(laundering_dir)
# laundering_params = ['babble_0']

# feat_types = ['cqcc_features', 'lfcc_features', 'lfcc_features_airasvspoof']
feat_types = ['lfcc_features_airasvspoof']


for lp in laundering_params:

    for ft in feat_types:

        out_dir = os.path.join(laundering_dir, lp)
        # out_dir = feat_dir_ft

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_fullfile_h5 = os.path.join(out_dir, ft + '.h5')

        # reading from hdf5 file
        # with h5py.File(out_fullfile_h5, 'r') as h5f:
            
        #     for name, dataset in h5f.items():
        #         print(name, dataset.shape, dataset.dtype)

                # name_ls = name.split('_')
                # new_filename = '_'.join((name_ls[0], name_ls[1], name_ls[2], name_ls[3], name_ls[4]))
                # print(new_filename)
                # h5f.move(name, )

        if not os.path.exists(out_fullfile_h5):

            feat_dir_ft = os.path.join(laundering_dir, lp, ft, 'eval')

            # list all files in the directory
            feat_files = os.listdir(feat_dir_ft)

            with h5py.File(out_fullfile_h5, 'w') as h5f:

                for i, feat_pkl_file in enumerate(feat_files):

                    feat_pkl_fullfile = os.path.join(feat_dir_ft, feat_pkl_file)

                    try:
                        with open(feat_pkl_fullfile, 'rb') as pf:
                            feat_data = pickle.load(pf)
                    
                    except:
                        print("Not able to open pickle file using normal pickle open method")

                        feat_data = pickle_blosc.unpickle(feat_pkl_fullfile)
                        

                    print("Saving {} features for file {} at iteration {} in hdf5 format".format(ft, feat_pkl_file, i))

                    # out_fullfile = os.path.join(out_dir, feat_pkl_file)
                    # pickle_blosc.pickle(feat_data, out_fullfile)

                    # print(feat_data)

                    dset = h5f.create_dataset(feat_pkl_file.split('.')[0], data=feat_data, dtype=np.float32)

        else:

            print("{} dataset already created".format(out_fullfile_h5))

        ######### delete feature directory ########

        # read from hdf5 file before deleting
        # with h5py.File(out_fullfile_h5, 'r') as h5f:
            
        #     # print(list(h5f.keys()))
        #     num_keys = len(list(h5f.keys()))
        
        #     dlt_dir = os.path.join(out_dir, ft)
        #     if num_keys in [71235, 71236, 71237, 71238, 71239]:
        #         try:
        #             print("Deleting the {} directory".format(dlt_dir))
        #             shutil.rmtree(dlt_dir)
        #         except OSError as e:
        #             print("Error: %s - %s." % (e.filename, e.strerror))

        #     else:
        #         print("Not Deleting {} directory since number of files in dataset is {} which is not equal to 71237".format(dlt_dir, num_keys))

                # if i == 0:

                #     with h5py.File(out_fullfile_h5, 'w') as f:

                #         dset = f.create_dataset(feat_pkl_file.split('.')[0], data=feat_data, dtype=np.float32)

                # else:
                    
                #     with h5py.File(out_fullfile_h5, 'a') as f:

                #         dset = f.create_dataset(feat_pkl_file.split('.')[0], data=feat_data, dtype=np.float32)

            # out_fullfile_npy = os.path.join(out_dir, feat_pkl_file.split('.')[0] + '.npy')

            # f = gzip.GzipFile(out_fullfile_npy + '.gz', "w")

            # np.save(file=f, arr=feat_data)

            # f.close()

            # np.save(out_fullfile_npy, feat_data)



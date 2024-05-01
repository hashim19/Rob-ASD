import os
import pickle
import pickle_blosc
import lzma
import mgzip

feat_dir = '/data/Features/'

feat_out_dir = 'pickle_blosc'

laundering_type = 'ASVspoof2019_LA_eval'
# laundering_params = ['AsvSpoofData_2019_babble_0_0_5', 'AsvSpoofData_2019_babble_10_10_5', 'AsvSpoofData_2019_babble_20_20_5', 'AsvSpoofData_2019_cafe_0_0_5', 'AsvSpoofData_2019_cafe_10_10_5',
#                         'AsvSpoofData_2019_cafe_20_20_5', 'AsvSpoofData_2019_street_0_0_5', 'AsvSpoofData_2019_street_10_10_5', 'AsvSpoofData_2019_street_20_20_5',
#                         'AsvSpoofData_2019_volvo_0_0_5', 'AsvSpoofData_2019_volvo_10_10_5', '']

laundering_dir = os.path.join(feat_dir, laundering_type)
laundering_params = os.listdir(laundering_dir)
# laundering_params = ['AsvSpoofData_2019_babble_0_0_5']

feat_types = ['cqcc_features', 'lfcc_features', 'lfcc_features_airasvspoof']


for lp in laundering_params:

    for ft in feat_types:

        feat_dir_ft = os.path.join(laundering_dir, lp, ft, 'eval')

        # out_dir = os.path.join(laundering_dir, lp, feat_out_dir)
        out_dir = feat_dir_ft

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # list all files in the directory
        feat_files = os.listdir(feat_dir_ft)

        for i, feat_pkl_file in enumerate(feat_files):

            feat_pkl_fullfile = os.path.join(feat_dir_ft, feat_pkl_file)

            try:
                with open(feat_pkl_fullfile, 'rb') as f:
                    feat_data = pickle.load(f)
            
            except:
                continue


            print("Compressing {} features for file {} at iteration {}".format(ft, feat_pkl_file, i))

            out_fullfile = os.path.join(out_dir, feat_pkl_file)
            pickle_blosc.pickle(feat_data, out_fullfile)
            
            # if feat_out_dir == 'mgzip':
            #     out_fullfile_lzma = os.path.join(out_dir, feat_pkl_file.split('.')[0] + '.mgzip')
                
            #     with mgzip.open(out_fullfile_lzma, "wb") as f:
            #         pickle.dump(feat_data, f)

            # out_fullfile_lzma = os.path.join(out_dir, feat_pkl_file.split('.')[0] + '.xz')
            # print(out_fullfile_lzma)
            








# from sklearn.mixture import GaussianMixture
from pycave.bayes import GaussianMixture
from pycave import set_logging_level
from random import sample
import pandas
import pickle
import math
import numpy as np
import os
import logging
import time
from Asvspoof_dataset import PKL_dataset, open_pkl, gmm_custom_collate
from extract_features import extract_features

set_logging_level(logging.INFO)


def train_gmm(data_label, features, train_keys, train_folders, audio_ext, dict_file, ncomp, feat_dir='features', init_only=False):

    print("Training GMM for {} data".format(data_label))

    # define directory to save the gmm model
    gmm_save_dir = '_'.join((dict_file, data_label))

    # path to the dataset
    path_to_dataset = os.path.join(feat_dir, features + '_features', 'train', data_label)

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

    
    # train the Gmm model on the data
    gmm = GaussianMixture(num_components=ncomp,
                            covariance_type='diag',
                            batch_size=1024,
                            covariance_regularization=0.1,
                            # init_strategy='kmeans++',
                            trainer_params=dict(accelerator='gpu', devices=[2], max_epochs=100))

    history = gmm.fit(train_data)

    # save the trained model
    gmm.save(gmm_save_dir)

    return gmm

# scoring function
def scoring(scores_file, dict_file, features, eval_ndx, eval_folder, audio_ext, features_cached=True, flag_debug=False):
    logging.info('Scoring eval data')

    gmm_bona = GaussianMixture().load(dict_file['bona'])
    gmm_spoof = GaussianMixture().load(dict_file['spoof'])

    # with open(dict_file, "rb") as tf:
    #     gmm_dict = pickle.load(tf)
    #     gmm_bona.set_params(gmm_dict['bona'])
    #     gmm_spoof.set_params(gmm_dict['spoof'])

    # gmm_bona.load(dict_file['bona'])
    # gmm_spoof.load(dict_file['spoof'])

    # gmm_bona.load_attributes(dict_file['bona'])

    print(gmm_bona.get_params())
    print(gmm_spoof.get_params())

    print(gmm_bona.converged_)
    print(gmm_bona.num_iter_)
    print(gmm_bona.nll_)

    print(gmm_spoof.converged_)
    print(gmm_spoof.num_iter_)
    print(gmm_spoof.nll_)

    pd = pandas.read_csv(eval_ndx, sep=' ', header=None)

    print(pd)

    if flag_debug:
        pd = pd[:1000]

    files = pd[1].values
    scr = np.zeros_like(files, dtype=np.log(1).dtype)

    total_time = 0
    for i, file in enumerate(files):
        loop_start_time = time.time()

        if (i+1) % 1000 == 0:
            logging.info("\t...%d/%d..." % (i+1, len(files)))

        # try:
        Tx = extract_features(eval_folder + file + audio_ext, features=features, feat_root='./features_out', data_type='eval', cached=features_cached)
        print(i)
        
        extraction_checkpoint = time.time()
        print("feature extraction time {}".format(extraction_checkpoint-loop_start_time))

        Tx = Tx.astype('float32')

        print(Tx.shape)

        # Tx_tensor = torch.from_numpy(Tx)

        # print(Tx_tensor.size())
        
        bona_score = gmm_bona.score(Tx)
        spoof_score = gmm_spoof.score(Tx)

        print(bona_score)
        print(spoof_score)

        scr[i] = bona_score - spoof_score

        scoring_checkpoint = time.time()

        print("scoring time {}".format(scoring_checkpoint - extraction_checkpoint))

        loop_time = scoring_checkpoint - loop_start_time
        print("total time {}".format(loop_time))

        total_time += loop_time

        if i == 100:
            print("avg loop time {}".format(total_time/100))

        # except Exception as e:
        #     logging.warning(e)
        #     scr[i] = log(1)

    pd_out = pandas.DataFrame({'files': files, 'scores': scr})
    pd_out.to_csv(scores_file, sep=' ', header=False, index=False)

    logging.info('\t... scoring completed.\n')
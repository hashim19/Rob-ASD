from sklearn.mixture import GaussianMixture
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
from scipy.special import logsumexp


def train_gmm(data_label, features, train_keys, train_folders, audio_ext, dict_file, ncomp, feat_dir='features', init_only=False):
    logging.info('Start GMM training.')

    gmm_save_dir = os.path.join(dict_file, data_label)

    if not os.path.exists(gmm_save_dir):
        os.makedirs(gmm_save_dir)

    partial_gmm_dict_file = os.path.join(gmm_save_dir, '_'.join(('init', 'partial.pkl')))

    if os.path.exists(partial_gmm_dict_file):
        gmm = GaussianMixture(covariance_type='diag')
        with open(partial_gmm_dict_file, "rb") as tf:
            gmm._set_parameters(pickle.load(tf))
    else:
        data = list()
        for k, train_key in enumerate(train_keys):
            pd = pandas.read_csv(train_key, sep=' ', header=None)
            files = pd[pd[4] == data_label][1]
            # files_subset = sample(list(files), 1000)  # random init with 1000 files
            files_subset = (files.reset_index()[1]).loc[list(range(0, len(files), 10))]  # only every 10th file init
            for file in files_subset:
                # Tx = extract_features(train_folders[k] + file + audio_ext, features=features, cached=True)
                Tx = extract_features(train_folders[k] + file + audio_ext, features=features, data_label=data_label,
                        data_type='train', feat_root=feat_dir, cached=True)
                
                data.append(Tx)

        X = np.vstack(data)
        print(X.shape)

        gmm = GaussianMixture(n_components=ncomp,
                              random_state=None,
                              covariance_type='diag',
                              max_iter=100,
                              verbose=2,
                              verbose_interval=1).fit(X)

        logging.info('GMM init done - llh: %.5f' % gmm.lower_bound_)

        with open(partial_gmm_dict_file, "wb") as f:
            pickle.dump(gmm._get_parameters(), f)

    if init_only:
        return gmm

    # EM training
    prev_lower_bound = -np.infty
    for i in range(1000):

        partial_gmm_dict_file = os.path.join(gmm_save_dir, '_'.join((str(i), 'partial.pkl')))
        # partial_gmm_dict_file = '_'.join((dict_file, data_label, str(i), 'partial.pkl'))

        if os.path.exists(partial_gmm_dict_file):

            print("gmm is already trained for the {} epoch".format(i))

            with open(partial_gmm_dict_file, "rb") as tf:
                gmm._set_parameters(pickle.load(tf))
                continue

        nk_acc = np.zeros_like(gmm.weights_)
        mu_acc = np.zeros_like(gmm.means_)
        sigma_acc = np.zeros_like(gmm.covariances_)
        log_prob_norm_acc = 0
        n_samples = 0
        for k, train_key in enumerate(train_keys):
            pd = pandas.read_csv(train_key, sep=' ', header=None)
            files = pd[pd[4] == data_label][1]

            for file in files.values:
                Tx = extract_features(train_folders[k] + file + audio_ext, features=features, data_label=data_label,
                        data_type='train', feat_root=feat_dir, cached=True)
                n_samples += Tx.shape[0]

                # e step
                weighted_log_prob = gmm._estimate_weighted_log_prob(Tx)
                log_prob_norm = logsumexp(weighted_log_prob, axis=1)
                with np.errstate(under='ignore'):
                    # ignore underflow
                    log_resp = weighted_log_prob - log_prob_norm[:, None]
                log_prob_norm_acc += log_prob_norm.sum()

                # m step preparation
                resp = np.exp(log_resp)
                # print(resp.shape)
                # print(Tx.shape)
                nk_acc += resp.sum(axis=0) + 10 * np.finfo(np.log(1).dtype).eps
                mu_acc += resp.T @ Tx
                sigma_acc += resp.T @ (Tx ** 2)

        # m step
        gmm.means_ = mu_acc / nk_acc[:, None]
        gmm.covariances_ = sigma_acc / nk_acc[:, None] - gmm.means_ ** 2 + gmm.reg_covar
        gmm.weights_ = nk_acc / n_samples
        gmm.weights_ /= gmm.weights_.sum()
        if (gmm.covariances_ <= 0.0).any():
            raise ValueError("ill-defined empirical covariance")
        gmm.precisions_cholesky_ = 1. / np.sqrt(gmm.covariances_)

        with open(partial_gmm_dict_file, "wb") as f:
            pickle.dump(gmm._get_parameters(), f)

        # infos
        lower_bound = log_prob_norm_acc / n_samples
        change = lower_bound - prev_lower_bound
        # logging.info("  Iteration %d\t llh %.5f\t ll change %.5f" % (i, lower_bound, change))
        print("  Iteration {} llh {} ll change {}".format(i, lower_bound, change))

        prev_lower_bound = lower_bound

        if abs(change) < gmm.tol:
            logging.info('  Coverged; too small change')
            gmm.converged_ = True
            break
    
    gmm_dict_final_file = os.path.join(gmm_save_dir, 'gmm_final.pkl')

    with open(gmm_dict_final_file, "wb") as f:
            pickle.dump(gmm._get_parameters(), f)

    return gmm


def scoring(scores_file, dict_file, features, eval_ndx, eval_folder, audio_ext, feat_dir='features', features_cached=True, flag_debug=False):
    logging.info('Scoring eval data')

    gmm_bona = GaussianMixture(covariance_type='diag')
    gmm_spoof = GaussianMixture(covariance_type='diag')

    bona_path = dict_file['bona']
    spoof_path = dict_file['spoof']

    with open(bona_path, "rb") as b:
        gmm_dict = pickle.load(b)
        gmm_bona._set_parameters(gmm_dict)
    
    with open(spoof_path, "rb") as s:
        gmm_dict = pickle.load(s)
        gmm_spoof._set_parameters(gmm_dict)

    pd = pandas.read_csv(eval_ndx, sep=' ', header=None)
    # pd = pandas.read_csv(eval_ndx, sep=',', header=None)
    # pd = pd.drop(0)
    # if flag_debug:
    #     pd = pd[:1000]

    files = pd[1].values
    print(files)
    print(len(files))

    scr = np.zeros_like(files, dtype=np.log(1).dtype)

    total_time = 0
    for i, file in enumerate(files):

        # if file == 'LA_E_8070617_RT_0_6' or file == 'LA_E_2800292_RT_0_9' or file == 'LA_E_1573988_RT_0_9':
        #     continue

        loop_start_time = time.time()

        if (i+1) % 1000 == 0:
            logging.info("\t...%d/%d..." % (i+1, len(files)))

        # try:
        Tx = extract_features(eval_folder + file + audio_ext, features=features, feat_root=feat_dir, data_type='eval', cached=features_cached)
        print(i)
        
        extraction_checkpoint = time.time()
        print("feature extraction time {}".format(extraction_checkpoint-loop_start_time))

        Tx = Tx.astype('float32')

        print(Tx.shape)

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

    pd_out = pandas.DataFrame({'files': files, 'scores': scr})
    pd_out.to_csv(scores_file, sep=' ', header=False, index=False)

    logging.info('\t... scoring completed.\n')


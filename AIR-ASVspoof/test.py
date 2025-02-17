import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import ASVspoof2019, InTheWild, ASVspoofLaundered, ASVspoof5Laundered
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np

import umap
import umap.plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

import sys
sys.path.append("../")
import config as config

def test_model(feat_model_path, loss_model_path, part, add_loss, device, data_dir, protocol_file_path, feat_path, d_name):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    # if "checkpoint" in dirname(feat_model_path):
    #     dir_path = dirname(dirname(feat_model_path))
    # else:
    #     dir_path = dirname(feat_model_path)

    score_dir = config.score_dir

    model = torch.load(feat_model_path, map_location="cuda")
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None
    # test_set = ASVspoof2019("LA", "./LA/Features/",
    #                         "/home/hashim/PhD/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/", part,
    #                         "LFCC", feat_len=750, padding="repeat")

    if config.db_type == 'in_the_wild':
        test_set = InTheWild(feat_path, protocol_file_path, part, "LFCC", feat_len=750, padding="repeat")
    
    elif config.db_type == 'asvspoof_eval':
        test_set = ASVspoof2019("LA", feat_path, protocol_file_path, part, "LFCC", feat_len=750, padding="repeat")
    
    elif config.db_type == 'asvspoof_eval_laundered':
        test_set =  ASVspoofLaundered(feat_path, protocol_file_path, config.protocol_filenames, part, "LFCC", feat_len=750, padding="repeat", feature_format=config.feature_format)

    elif config.db_type == 'asvspoof5_eval_laundered':
        test_set =  ASVspoof5Laundered(feat_path, protocol_file_path, config.protocol_filenames, part, "LFCC", feat_len=750, padding="repeat", feature_format=config.feature_format)

    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    model.eval()

    if config.db_type == 'asvspoof5_eval_laundered':
        with open(os.path.join(score_dir, d_name + '_checkpoint_cm_score.txt'), 'w') as cm_score_file:
            for i, (lfcc, audio_fn) in enumerate(tqdm(testDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(device)

                feats, lfcc_outputs = model(lfcc)

                score = F.softmax(lfcc_outputs)[:, 0]

                # if add_loss == "ocsoftmax":
                #     ang_isoloss, score = loss_model(feats)
                # elif add_loss == "amsoftmax":
                #     outputs, moutputs = loss_model(feats, labels)
                #     score = F.softmax(outputs, dim=1)[:, 0]

                for j in range(len(audio_fn)):

                    cm_score_file.write('%s %s\n' % (audio_fn[j], score[j].item()))
                    
    else:
        feat_ls = []
        labels_ls = []
        with open(os.path.join(score_dir, d_name + '_checkpoint_cm_score.txt'), 'w') as cm_score_file:
            for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(testDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(device)
                tags = tags.to(device)
                labels = labels.to(device)

                feats, lfcc_outputs = model(lfcc)

                score = F.softmax(lfcc_outputs)[:, 0]

                if add_loss == "ocsoftmax":
                    ang_isoloss, score = loss_model(feats, labels)
                elif add_loss == "amsoftmax":
                    outputs, moutputs = loss_model(feats, labels)
                    score = F.softmax(outputs, dim=1)[:, 0]

                # print(type(feats))
                # print(feats.shape)
                feat_ls.append(feats.detach().cpu().numpy())
                # print(len(feat_ls))

                labels_ls.extend(labels.cpu())

                # for j in range(labels.size(0)):

                #     labels_ls.append("spoof" if labels[j].data.cpu().numpy() else "bonafide")

                # for j in range(labels.size(0)):
                #     # cm_score_file.write(
                #     #     '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                #     #                           "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                #     #                           score[j].item()))

                #     cm_score_file.write(
                #         '%s %s %s\n' % (audio_fn[j], "spoof" if labels[j].data.cpu().numpy() else "bonafide", score[j].item()))
        
        # feat_array = np.concatenate(feat_ls)
        feat_array = np.vstack(feat_ls)
        print(feat_array.shape)
        labels = np.array(labels_ls)
        print(labels)
        print(labels.shape)
        # unique_labels = np.unique(labels_ls)
        # print(unique_labels)

        ################# Dimension reduction ###############

        figures_dir = './figures'

        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # UMAP
        # reducer = umap.UMAP(random_state=42, n_neighbors=50)

        # embedding_mapper = reducer.fit(feat_array)
        # # print(embedding.shape)

        # # plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=1, labels=labels)
        # # plt.title('UMAP projection of the OCSoftmax Embeddings', fontsize=24)

        # umap.plot.points(embedding_mapper, labels=labels)

        # figure_name = 'umap_ocsoftmax.png'

        # TSNE
        perplexity=100
        early_exagg = 50
        palette = ['#800F2F', '#023E8A']
        markers = ['o', 'o']

        tsne = TSNE(n_components=2, verbose=0, random_state=123, perplexity=perplexity, early_exaggeration=early_exagg, n_iter=5000, n_jobs=8)
        z = tsne.fit_transform(feat_array)

        df = pd.DataFrame()
        df["y"] = labels
        df["$X_1$"] = z[:,0]
        df["$X_2$"] = z[:,1]

        sns.scatterplot(x='$X_1$', y='$X_2$', hue=df.y.tolist(), style=df.y.tolist(), s=10, markers=markers, 
                        palette=palette, data=df).set(title = "TSNE projection of the OCSoftmax Embeddings") 

        figure_name = 'tsne_ocsoftmax.png'

        fig_sav_path = os.path.join(figures_dir, figure_name)
        plt.savefig(fig_sav_path, bbox_inches='tight')
    
    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, d_name + '_checkpoint_cm_score.txt'),
    #                                         "/home/hashim/PhD/Data/AsvSpoofData_2019/train/")

    # eer_cm = compute_eer_and_tdcf(os.path.join(score_dir, d_name + '_checkpoint_cm_score.txt'),
    #                                         "/home/hashim/PhD/Data/AsvSpoofData_2019/train/")
    # return eer_cm

def test(model_dir, add_loss, device, data_dir='data', protocol_path='', feat_dir='', data_name = ''):
    model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")

    test_model(model_path, loss_model_path, "eval", add_loss, device, data_dir, protocol_path, feat_dir, data_name)

def test_individual_attacks(cm_score_file):
    asv_score_file = os.path.join('/data/neil/DS_10283_3336',
                                  'LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(7,20):
        # Extract target, nontarget, and spoof scores from the ASV scores
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_sources == 'A%02d' % attack_idx]

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

        if eer_cm < other_eer_cm:
            # Compute t-DCF
            tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,
                                                        True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]

        else:
            tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == 'bonafide'],
                                                        other_cm_scores[cm_sources == 'A%02d' % attack_idx],
                                                        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
        eer_cm_lst.append(min(eer_cm, other_eer_cm))
        min_tDCF_lst.append(min_tDCF)

    return eer_cm_lst, min_tDCF_lst


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models1028/ocsoftmax")
    # parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
    #                     choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    # parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    # parser.add_argument("--f", type=str, )
    # args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "./models1028/ocsoftmax"
    loss = "ocsoftmax"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_folder = config.db_folder  # put your database root path here
    db_type = config.db_type

    laundering_type = config.laundering_type
    laundering_param = config.laundering_param
    protocol_pth = config.protocol_filenames[0]

    data_names = config.data_names

    if db_type == 'in_the_wild':
        eval_folder = os.path.join(db_folder, 'release_in_the_wild')
        eval_ndx = os.path.join(db_folder, 'protocols', protocol_pth)
        
    elif db_type == 'asvspoof_eval_laundered':
        eval_folder = os.path.join(db_folder, 'flac')
        eval_ndx = os.path.join(db_folder, 'protocols', protocol_pth.split('.')[0] + '_' 'tmp.txt')

    elif db_type == 'asvspoof_eval':
        eval_folder = os.path.join(db_folder, 'flac')

        eval_ndx = os.path.join(db_folder, 'protocols', protocol_pth)

    elif db_type == 'asvspoof5_eval_laundered':
        eval_folder = os.path.join(db_folder, data_names[0], 'flac')

        eval_ndx = os.path.join(db_folder, 'protocols', protocol_pth)

    Feat_dir = os.path.join(config.feat_dir, laundering_type, laundering_param, 'lfcc_features_airasvspoof')

    test(model_dir, loss, device, data_dir=eval_folder, protocol_path=eval_ndx, feat_dir=Feat_dir, data_name = 'OCSoftmax_'+ laundering_type + '_' + laundering_param)

    # eer_cm_lst, min_tDCF_lst = test_individual_attacks(os.path.join(args.model_dir, 'checkpoint_cm_score.txt'))
    # print(eer_cm_lst)
    # print(min_tDCF_lst)

    # cm_score_dir_path = "./models1028/ocsoftmax"
    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(cm_score_dir_path, 'checkpoint_cm_score.txt'), "/home/hashim/PhD/Data/AsvSpoofData_2019/train/")

    # remove the tmp protocol file
    # print("removing the temporary protocol file!")
    # os.remove(eval_ndx)



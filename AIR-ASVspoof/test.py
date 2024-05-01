import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np

def test_model(feat_model_path, loss_model_path, part, add_loss, device, data_dir, protocol_file_path, feat_path, d_name):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    model = torch.load(feat_model_path, map_location="cuda")
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None
    # test_set = ASVspoof2019("LA", "./LA/Features/",
    #                         "/home/hashim/PhD/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/", part,
    #                         "LFCC", feat_len=750, padding="repeat")

    test_set = ASVspoof2019("LA", feat_path, protocol_file_path, part, "LFCC", feat_len=750, padding="repeat")

    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    model.eval()

    with open(os.path.join(dir_path, d_name + '_checkpoint_cm_score.txt'), 'w') as cm_score_file:
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

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s A%02d %s %s\n' % (audio_fn[j], tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item()))

    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, d_name + '_checkpoint_cm_score.txt'),
    #                                         "/home/hashim/PhD/Data/AsvSpoofData_2019/train/")

    eer_cm = compute_eer_and_tdcf(os.path.join(dir_path, d_name + '_checkpoint_cm_score.txt'),
                                            "/home/hashim/PhD/Data/AsvSpoofData_2019/train/")
    return eer_cm

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

    model_dir = "./models/ocsoftmax"
    loss = "ocsoftmax"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_folder = '/data/Data/'  # put your database root path here

    # laundering_type = 'Noise_Addition/'
    # laundering = 'AsvSpoofData_2019_WN_20_20_5'
    # protocol_pth = 'white_20_20_5_protocol.txt'

    # laundering_type = 'Reverberation/'
    # laundering = 'AsvSpoofData_2019_RT_0_9'
    # protocol_pth = 'Protocol_ASV_RT_0_9.txt'

    # laundering_type = 'Recompression/'
    # laundering = 'recompression_320k'
    # protocol_pth = 'recompression_protocol_320k.txt'

    # laundering_type = 'Resampling/'
    # laundering = 'resample_44100'
    # protocol_pth = 'resample_44100.txt'

    laundering_type = 'Filtering/'
    laundering = 'low_pass_filt_7000'
    protocol_pth = 'low_pass_filt_7000_protocol.txt'
    
    eval_folder = db_folder + laundering_type + laundering + '/'
    # eval_folder = db_folder + laundering_type + laundering + '/' + 'flac/'
    eval_ndx = db_folder + 'AsvSpoofData_2019_protocols/' + protocol_pth
    # eval_ndx = db_folder + laundering_type + 'ASVspoof2019_LA_cm_protocols/' + protocol_pth
    feat_dir = os.path.join('/data/Features/', laundering_type, laundering, 'lfcc_features_airasvspoof')
    # feat_dir = os.path.join('/data/Features/', laundering, 'lfcc_features_new')

    test(model_dir, loss, device, data_dir=eval_folder, protocol_path=eval_ndx, feat_dir=feat_dir, data_name = laundering)

    # eer_cm_lst, min_tDCF_lst = test_individual_attacks(os.path.join(args.model_dir, 'checkpoint_cm_score.txt'))
    # print(eer_cm_lst)
    # print(min_tDCF_lst)

    # cm_score_dir_path = "./models1028/ocsoftmax"
    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(cm_score_dir_path, 'checkpoint_cm_score.txt'), "/home/hashim/PhD/Data/AsvSpoofData_2019/train/")


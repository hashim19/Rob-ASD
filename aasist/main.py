"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import pandas as pd
import sys
sys.path.append("../")
import config as config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list, genSpoof_list_wild)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        args_config = json.loads(f_json.read())
    model_config = args_config["model_config"]
    optim_config = args_config["optim_config"]
    optim_config["epochs"] = args_config["num_epochs"]
    track = args_config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in args_config:
        args_config["eval_all_best"] = "True"
    if "freq_aug" not in args_config:
        args_config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, args_config)

    # define database related paths
    db_folder = config.db_folder  # put your database root path here
    db_type = config.db_type
    data_names = config.data_names

    orig_database_path = Path(args_config["orig_data_path"])

    laundering_type = config.laundering_type
    laundering_param = config.laundering_param
    protocol_filenames = config.protocol_filenames
    audio_ext = config.audio_ext
    data_types = config.data_types
    data_labels = config.data_labels

    output_dir = Path(args.output_dir)
    score_dir = Path(config.score_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)

    eval_pf_ls = []
    train_pf_ls = []
    dev_pf_ls = []

    pathToDatabase_eval = []
    pathToDatabase_train = []
    pathToDatabase_dev = []

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

                eval_pf_ls.append(evalProtocolFile)

            elif db_type == 'asvspoof_eval_laundered':
                pathToDatabase = os.path.join(db_folder, 'flac')

                evalprotcol = pd.read_csv(evalProtocolFile, sep=' ', names=["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY", "Laundering_Type", "Laundering_Param"])
                
                # create a temporary protocol file, this file will be used by test.py
                evalprotcol_tmp = evalprotcol.loc[evalprotcol['Laundering_Param'] == laundering_param]
                evalprotcol_tmp = evalprotcol_tmp[["Speaker_Id", "AUDIO_FILE_NAME", "SYSTEM_ID", "KEY"]]
                evalprotcol_tmp.insert(loc=3, column="Not_Used_for_LA", value='-')
                evalprotcol_tmp.to_csv(os.path.join(db_folder, 'protocols', protocol_filename.split('.')[0] + '_' 'tmp.txt'), header=False, index=False, sep=" ")

                evalProtocolFile_tmp = os.path.join(db_folder, 'protocols', evalProtocolFile.split('.')[0] + '_' 'tmp.txt')

                eval_pf_ls.append(evalProtocolFile_tmp)

            pathToDatabase_eval.append(pathToDatabase)

        elif data_type == 'train' or data_type == 'dev':

            pathToDatabase = os.path.join(db_folder, data_name, 'flac')

            protocol_file_path = os.path.join(db_folder, 'protocols', protocol_filename)

            if data_type == 'train':
                train_pf_ls.append(protocol_file_path)
                pathToDatabase_train.append(pathToDatabase)

            elif data_type == 'dev':
                dev_pf_ls.append(protocol_file_path)
                pathToDatabase_dev.append(pathToDatabase)

    # dev_trial_path = (database_path /
    #                   "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
    #                       track, prefix_2019))
    # eval_trial_path = (
    #     database_path /
    #     "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
    #         track, prefix_2019))

    eval_out = os.path.join(score_dir, 'RawGAT_' + laundering_type + '_' + laundering_param + '_eval_CM_scores.txt')

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        args_config["num_epochs"], args_config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = score_dir / Path('AASIST_' + laundering_type + '_' + laundering_param + '_' + str(args_config["eval_output"]))
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    # trn_loader, dev_loader, eval_loader = get_loader(
    #     database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:

        eval_loader = get_loader(pathToDatabase_eval, eval_pf_ls, args.seed, args_config=args_config, config=config, data_type='eval')

        model.load_state_dict(
            torch.load(args_config["model_path"], map_location=device))
        print("Model loaded : {}".format(args_config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_pf_ls, config)
        # calculate_tDCF_EER(cm_scores_file=eval_score_path,
        #                    asv_score_file=orig_database_path /
        #                    config["asv_score_path"],
        #                    output_file=model_tag / "t-DCF_EER.txt")
        # print("DONE.")
        # eval_eer, eval_tdcf = calculate_tDCF_EER(
        #     cm_scores_file=eval_score_path,
        #     asv_score_file=orig_database_path / config["asv_score_path"],
        #     output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    
    # get train and dev data loader
    trn_loader = get_loader(pathToDatabase_train, train_pf_ls, args.seed, args_config=args_config, config=config, data_type='train')
    dev_loader = get_loader(pathToDatabase_dev, dev_pf_ls, args.seed, args_config=args_config, config=config, data_type='dev')

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(args_config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, args_config)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_pf_ls, config)
        # dev_eer, dev_tdcf = calculate_tDCF_EER(
        #     cm_scores_file=metric_path/"dev_score.txt",
        #     asv_score_file=orig_database_path/args_config["asv_score_path"],
        #     output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
        #     printout=False)
        
        dev_eer = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=orig_database_path/args_config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        
        # print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
        #     running_loss, dev_eer, dev_tdcf))
        
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(
            running_loss, dev_eer))
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        # writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        # best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
    #         if str_to_bool(args_config["eval_all_best"]):
    #             produce_evaluation_file(eval_loader, model, device,
    #                                     eval_score_path, eval_pf_ls, config)
    #             # eval_eer, eval_tdcf = calculate_tDCF_EER(
    #             #     cm_scores_file=eval_score_path,
    #             #     asv_score_file=orig_database_path / args_config["asv_score_path"],
    #             #     output_file=metric_path /
    #             #     "t-DCF_EER_{:03d}epo.txt".format(epoch))
                
    #             eval_eer = calculate_tDCF_EER(
    #                 cm_scores_file=eval_score_path,
    #                 asv_score_file=orig_database_path / args_config["asv_score_path"],
    #                 output_file=metric_path /
    #                 "t-DCF_EER_{:03d}epo.txt".format(epoch))

    #             log_text = "epoch{:03d}, ".format(epoch)
    #             if eval_eer < best_eval_eer:
    #                 log_text += "best eer, {:.4f}%".format(eval_eer)
    #                 best_eval_eer = eval_eer
    #                 torch.save(model.state_dict(),
    #                            model_save_path / "best.pth")
                    
    #             # if eval_tdcf < best_eval_tdcf:
    #             #     log_text += "best tdcf, {:.4f}".format(eval_tdcf)
    #             #     best_eval_tdcf = eval_tdcf
    #             #     torch.save(model.state_dict(),
    #             #                model_save_path / "best.pth")
    #             if len(log_text) > 0:
    #                 print(log_text)
    #                 f_log.write(log_text + "\n")

    #         print("Saving epoch {} for swa".format(epoch))
    #         optimizer_swa.update_swa()
    #         n_swa_update += 1
    #     writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
    #     # writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    # print("Start final evaluation")
    # epoch += 1
    # if n_swa_update > 0:
    #     optimizer_swa.swap_swa_sgd()
    #     optimizer_swa.bn_update(trn_loader, model, device=device)
    # produce_evaluation_file(eval_loader, model, device, eval_score_path,
    #                         eval_pf_ls, config)
    # eval_eer = calculate_tDCF_EER(cm_scores_file=eval_score_path,
    #                                          asv_score_file=orig_database_path /
    #                                          args_config["asv_score_path"],
    #                                          output_file=model_tag / "t-DCF_EER.txt")
    # f_log = open(model_tag / "metric_log.txt", "a")
    # f_log.write("=" * 5 + "\n")
    # f_log.write("EER: {:.3f}".format(eval_eer))
    # f_log.close()

    # torch.save(model.state_dict(),
    #            model_save_path / "swa.pth")

    # if eval_eer <= best_eval_eer:
    #     best_eval_eer = eval_eer
    # if eval_tdcf <= best_eval_tdcf:
    #     best_eval_tdcf = eval_tdcf
    #     torch.save(model.state_dict(),
    #                model_save_path / "best.pth")
    # print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
    #     best_eval_eer, best_eval_tdcf))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


# def get_loader(
#         database_path: str,
#         seed: int,
#         config: dict) -> List[torch.utils.data.DataLoader]:
#     """Make PyTorch DataLoaders for train / developement / evaluation"""
#     track = config["track"]
#     prefix_2019 = "ASVspoof2019.{}".format(track)

#     trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
#     dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
#     eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

#     trn_list_path = (database_path /
#                      "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
#                          track, prefix_2019))
#     dev_trial_path = (database_path /
#                       "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
#                           track, prefix_2019))
#     eval_trial_path = (
#         database_path /
#         "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
#             track, prefix_2019))

#     d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
#                                             is_train=True,
#                                             is_eval=False)
#     print("no. training files:", len(file_train))

#     train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
#                                            labels=d_label_trn,
#                                            base_dir=trn_database_path)
#     gen = torch.Generator()
#     gen.manual_seed(seed)
#     trn_loader = DataLoader(train_set,
#                             batch_size=config["batch_size"],
#                             shuffle=True,
#                             drop_last=True,
#                             pin_memory=True,
#                             worker_init_fn=seed_worker,
#                             generator=gen)

#     _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
#                                 is_train=False,
#                                 is_eval=False)
#     print("no. validation files:", len(file_dev))

#     dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
#                                             base_dir=dev_database_path)
#     dev_loader = DataLoader(dev_set,
#                             batch_size=config["batch_size"],
#                             shuffle=False,
#                             drop_last=False,
#                             pin_memory=True)

#     file_eval = genSpoof_list(dir_meta=eval_trial_path,
#                               is_train=False,
#                               is_eval=True)
#     eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
#                                              base_dir=eval_database_path)
#     eval_loader = DataLoader(eval_set,
#                              batch_size=config["batch_size"],
#                              shuffle=False,
#                              drop_last=False,
#                              pin_memory=True)

#     return trn_loader, dev_loader, eval_loader


# new dataloader
def get_loader(
        database_path: str,
        protocol_path: str,
        seed: int,
        args_config: dict,
        config,
        data_type: str) -> List[torch.utils.data.DataLoader]:

    
    if data_type == 'eval':

        if config.db_type == 'in_the_wild':
            file_eval = genSpoof_list_wild(dir_meta=protocol_path)
        
        elif config.db_type == 'asvspoof_eval_laundered':
            file_eval = genSpoof_list(dir_meta=protocol_path,
                                    is_train=False,
                                    is_eval=True)

        eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                                base_dir=database_path,
                                                audio_ext=config.audio_ext)
        
        data_loader = DataLoader(eval_set,
                                 batch_size=args_config["batch_size"],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)
        
    elif data_type == 'dev':

        _, file_dev = genSpoof_list(dir_meta=protocol_path,
                                    is_train=False,
                                    is_eval=False)
        
        print("no. validation files:", len(file_dev))

        dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                                base_dir=database_path,
                                                audio_ext=config.audio_ext)
        data_loader = DataLoader(dev_set,
                                batch_size=args_config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        
    elif data_type == 'train':

        d_label_trn, file_train = genSpoof_list(dir_meta=protocol_path,
                                                is_train=True,
                                                is_eval=False)

        print("no. training files:", len(file_train))

        train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                            labels=d_label_trn,
                                            base_dir=database_path,
                                            audio_ext=config.audio_ext)
        gen = torch.Generator()
        gen.manual_seed(seed)
        data_loader = DataLoader(train_set,
                                batch_size=args_config["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                generator=gen)


    

    return data_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    config) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()

    trial_lines = []

    for tp in trial_path:
        with open(tp, "r") as f_trl:
            trial_lines.extend(f_trl.readlines())

    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):

            if config.db_type == 'in_the_wild':
                utt_id, _, key = trl.strip().split(',')

                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(utt_id, key, sco))

            elif config.db_type == 'asvspoof_eval_laundered':
                _, utt_id, _, src, key = trl.strip().split(' ')

                assert fn == utt_id
                fh.write("{} {} {}\n".format(utt_id, key, sco))

            elif config.db_type == 'asvspoof_train_laundered':
                _, utt_id, _, src, key = trl.strip().split(' ')[:5]

                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))

            
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())

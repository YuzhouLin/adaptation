import argparse
import os
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt

import time
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from Models.rawConvNet import Model
from Models.model_training import pre_train_model, trian_TL, train_ADANN, train_FT
import utils
from load_prepared_dataset_in_dataloaders import load_dataloaders, load_dataloaders_adaptation
from train_networks import run_training
import optuna

parser = argparse.ArgumentParser()
method = parser.add_mutually_exclusive_group()
parser.add_argument('-filter', '--filtered', help='if data needs to be filtered?', action='store_true')
method.add_argument('-normalisation', '--normalised', help='if our normalisation is applied?', action='store_true')
method.add_argument('-ADANN', '--ADANN_used', help='if ADANN is applied?', action='store_true')

args = parser.parse_args()


CLASS_N = 7
BLOCK_N = 3
FILTER_SIZE = (1, 16)
EPOCHS = 500
DEVICE = utils.get_device()


def pretrain(params, trainloaders):
    since = time.time()    
    model_name = f"{params['weight_path']}{params['ex_n']}_pretrained.pt"
    #  Load the model
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=.35, filter_size=FILTER_SIZE)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.5, 0.999))

    criterion = nn.CrossEntropyLoss(reduction='mean')
    best_loss = np.inf

    if params['ADANN_used']:
        weight_domain_loss = 1e-1
        precision = 1e-8
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15, verbose=True, eps=precision)
        patience = 30
        patience_increase = 30

        list_dictionaries_BN_weights = []
        for index_BN_weights in range(len(dataloaders_validation)):
            state_dict = model.state_dict()
            batch_norm_dict = {}
            for key in state_dict:
                if "batchNorm" in key:
                    batch_norm_dict.update({key: state_dict[key]})
            list_dictionaries_BN_weights.append(copy.deepcopy(batch_norm_dict))

        best_weights = copy.deepcopy(model.state_dict())
        best_bn_statistics = copy.deepcopy(list_dictionaries_BN_weights)

        for epoch in range(1, EPOCHS+1):
            val_loss = train_ADANN(model, trainloaders, list_dictionaries_BN_weights, criterion, optimizer, weight_domain_loss, DEVICE)
            scheduler.step(val_loss)
            if val_loss + precision < best_loss:
                print("New best validation loss:", val_loss)
                best_loss = val_loss
                patience = patience_increase + epoch
                # save the pretrained model
                torch.save(model.state_dict(), model_name)
            if epoch > patience:
                break

    else:
        early_stopping_iter = 10
        early_stopping_counter = 0

        for epoch in range(1,EPOCHS+1):
            train_losses = trian_TL(model, trainloaders, criterion, optimizer, DEVICE)

            train_loss = train_losses['train']
            valid_loss = train_losses['val']
            print(f"Epoch:{epoch}, train_loss:{train_loss}, valid_loss:{valid_loss}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                # save the pretrained model
                torch.save(model.state_dict(), model_name)
            else:
                early_stopping_counter += 1
            if early_stopping_counter > early_stopping_iter:
                break
    time_taken = time.time()-since
    time_taken = "{:.2f}".format(time_taken)
    temp_file = f"{params['weight_path']}/pre_train_time.csv"
    df_new = pd.DataFrame({'Experiment':[params['ex_n']], 'Pre_train_time':[time_taken], 'Best_val_loss': [best_loss]})
    if not os.path.exists(temp_file):
        df = df_new
    else:
        df = pd.read_csv(temp_file, index_col=False)
        df = df.append(df_new, ignore_index=True)
    
    df.to_csv(temp_file, index=False)
    print(df)
    return #best_loss

def adaptation(params, data, epoch_n = 1):
    since = time.time()
    pre_trained_model = f"{params['weight_path']}{params['ex_n']}_pretrained.pt"
    #  Load the pre_trained_model
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=.35, filter_size=FILTER_SIZE)
    model.load_state_dict(torch.load(pre_trained_model, map_location=DEVICE))
    '''
    for name, param in model.named_parameters():
        if "batchNorm" not in name:
        #if "output" not in name:
            param.requires_grad = False
    '''
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(1, epoch_n+1):
        train_loss = train_FT(model, data, criterion, optimizer, DEVICE)
        torch.save(model.state_dict(), f"{params['weight_path']}{params['ex_n']}_adapted.pt")
    time_taken = time.time()-since
    time_taken = "{:.2f}".format(time_taken)
    temp_file = f"{params['weight_path']}/adap_time.csv"
    df_new = pd.DataFrame({'Experiment':[params['ex_n']], 'Adap_time':[time_taken], 'Adap_loss': [train_loss]})
    if not os.path.exists(temp_file):
        df = df_new
    else:
        df = pd.read_csv(temp_file, index_col=False)
        df = df.append(df_new, ignore_index=True)

    df.to_csv(temp_file, index=False)
    print(df)

    return

def testing(params, data, ft=0):
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=.35, filter_size=FILTER_SIZE)
    if ft==0:
        loaded_model = f"{params['weight_path']}{params['ex_n']}_pretrained.pt"
    else:
        loaded_model = f"{params['weight_path']}{params['ex_n']}_adapted.pt"
    model.load_state_dict(torch.load(loaded_model, map_location=DEVICE))

    predictions = []
    ground_truths = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in data:
            output, _ = model(inputs)
            _, predicted = torch.max(output.data, 1)
            #print(predicted)
            #print(labels)
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.numpy())
        print(" Accuracy: ", np.mean(np.array(predictions) == np.array(ground_truths)))
    return


if __name__ == "__main__":

    data_prepare_required = False  # Set it as True for the first time use

    #  Data preparation parameters
    prepare_params = {}
    prepare_params['output_dir'] = "Dataset/Intact_dataset/NinaProDB5"
    prepare_params['input_dir'] = '/cluster/home/cug/yl339/CNN_rawdata_TF_GPU/NinaPro/DB5'
    prepare_params['sb_list'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 subjects
    prepare_params['cycle_list'] = [1, 2, 3, 4, 5, 6]  # 6 cycles
    prepare_params['class_list'] = [0, 5, 6, 9, 10, 13, 14] # 7 gestures including rest
    # i.e., "Rest", "Open Hand", "Power Grip", "Wrist Flexion", "Wrist Extension", "Wrist Pronation", "Wrist Supination"
    prepare_params['fs'] = 200  # sampling frequency: (Hz)

    #  Data process parameters
    process_params = {}
    process_params['input_dir'] = prepare_params['output_dir']
    process_params['sb_list'] = prepare_params['sb_list']
    process_params['cycle_list'] = prepare_params['cycle_list']
    process_params['gesture_n'] = len(prepare_params['class_list'])
    process_params['fs'] = prepare_params['fs']
    process_params['window_size'] = 50
    process_params['size_non_overlap'] = 15


    if data_prepare_required:
        utils.data_prepare(**prepare_params)
        utils.data_process(**process_params)

    params = {
        'batch_size': 128,
        'lr': 1e-3,
        'filtered_required': args.filtered,
        'normalised_required': args.normalised,
        'ADANN_used': args.ADANN_used
    }

    for target_sb in process_params['sb_list']:
        data_path = process_params['input_dir']+'/Participant%d/EMG/Processed/' % target_sb

        results_path = f'Results/Participant{target_sb}'

        if params['filtered_required']:
            weight_path = results_path + '/weights/filtered/'
            study_path = results_path + '/studys/filtered/'
            test_path = results_path + '/tests/filtered/'
        else:
            weight_path = results_path + '/weights/raw/'
            study_path = results_path + '/studys/raw/'
            test_path = results_path + '/tests/raw/'

        params['data_path'] = data_path
        params['weight_path'] = weight_path
        params['study_path'] = study_path
        params['test_path'] = test_path

        if params['ADANN_used']:
            params['ex_n'] = 'C'
        else:
            params['ex_n'] = 'B' if params['normalised_required'] else 'A'
        '''
        # Load the determined hyperparameters
        loaded_study = optuna.load_study(
            study_name="STUDY", storage= f"sqlite:///{params['study_path']}/{params['ex_n']}.db")
        temp_best_trial = loaded_study.best_trial
        # Update for the optimal hyperparameters
        for key, value in temp_best_trial.params.items():
            params[key] = value
        '''
        # Get the dataset ready
        dataloaders_train, \
        dataloaders_validation, \
        dataloaders_adaptation, \
        dataloaders_test = \
            utils.load_dataloaders_DB5(path=params['data_path'], batch_size=params['batch_size'], filtered=params['filtered_required'], normalised=params['normalised_required'], aggre_required=not params['ADANN_used'])

        print(dataloaders_adaptation)
        print(len(dataloaders_adaptation.dataset))
        '''
        #  Get trainloaders ready
        trainloaders = {
            "train": dataloaders_train,
            "val": dataloaders_validation,
        }
        pretrain(params, trainloaders)
        '''
        adaptation(params, dataloaders_test, epoch_n = 10)
        testing(params, dataloaders_adaptation, ft=1)
        testing(params, dataloaders_test, ft=1)

        exit()


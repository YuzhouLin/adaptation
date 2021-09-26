import argparse
import os
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Models.rawConvNet import Model
from Models.model_training import pre_train_model, trian_TL, train_ADANN
import utils
from load_prepared_dataset_in_dataloaders import load_dataloaders, load_dataloaders_adaptation
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

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


def pretrain_raw_convNet(train_data, valid_data, filter_size=(1, 11)):
    dataloaders_train = train_data
    dataloaders_validation = valid_data
    # Define Model
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=0.35, filter_size=filter_size).cuda()

    # Define Loss functions
    cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()
    cross_entropy_loss_domains = nn.CrossEntropyLoss(reduction='mean').cuda()
    
    # Define Optimizer
    learning_rate = 0.0404709
    print(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Define Scheduler
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15, verbose=True, eps=precision)

    best_weights, _ = pre_train_model(model=model, cross_entropy_loss_for_class=cross_entropy_loss_classes, cross_entropy_loss_for_domain=cross_entropy_loss_domains, optimizer_class=optimizer, scheduler=scheduler, dataloaders={"train": dataloaders_train, "val": dataloaders_validation}, precision=precision)


    weight_path = 'weights/ADANN'
    if not(os.path.exists(weight_path)):
        os.makedirs(weight_path)
    torch.save(best_weights, f=weight_path+"/TL_best_weights.pt")


def test_network_raw_convNet(adaptation_data, test_data, 
                            path_weights = 'weights/ADANN/TL_best_weights.pt',
                            filter_size=(1,26)):
    
    result_path = 'results/ADANN'
    if not(os.path.exists(result_path)):
        os.makedirs(result_path)

    with open(result_path+"/test_accuracy_filter_size_"+str(filter_size[1])+".txt", "a") as myfile:
        myfile.write("Test")

    dataloaders_test = test_data
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=.35, filter_size=filter_size).cuda()
    best_weights = torch.load(path_weights)
    model.load_state_dict(copy.deepcopy(best_weights))

    # Frozen parameters except BatchNorm ones
    for name, param in model.named_parameters():
        if 'batchNorm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Define Loss functions
    cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

    # Define Optimizer
    learning_rate = 0.0404709
    print(model.parameters())
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    # Define Scheduler
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15, verbose=True, eps=precision)

    # Train only one epoch with adapatation data
    for _,data in enumerate(adaptation_data):
        # get the source inputs
        inputs_source, labels_source = data
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        # pass the source input through the model
        output_class_source, output_domain_source, _ = model(inputs_source)
        loss_classification_source = cross_entropy_loss_classes(output_class_source, labels_source)
        optimizer.zero_grad()
        loss_classification_source.backward(retain_graph=True)
        optimizer.step()
        
    # Save the weights for this dataset

    best_weights = copy.deepcopy(model.state_dict())

    weight_path = 'weights/ADANN'
    if not(os.path.exists(weight_path)):
        os.makedirs(weight_path)
    torch.save(best_weights, f=weight_path+"/TL_best_weights1.pt")


    predictions = []
    ground_truth = []
    accuracies = []
    
    participant_index = 1
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloaders_test:
            inputs = inputs.cuda()
            output, _ = model(inputs)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.numpy())
    accuraccy = np.mean(np.array(predictions) == np.array(ground_truth))
    print("Participant: ", participant_index, " Accuracy: ", accuraccy)
    
    with open("results/ours/test_accuracy_filter_size_" + str(filter_size[1]) + ".txt", "a") as myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truth) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies)))

    return predictions, ground_truth


def run_training(params, save_required=False):
    #  Load dataset
    dataloaders_train, \
    dataloaders_validation, \
    dataloaders_adaptation, \
    dataloaders_test = \
        utils.load_dataloaders_DB5(path=params['data_path'], batch_size=params['batch_size'], filtered=params['filtered_required'], normalised=params['normalised_required'], aggre_required=not params['ADANN_used'])

    #  Get trainloaders ready
    trainloaders = {
        "train": dataloaders_train,
        "val": dataloaders_validation,
    }

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

        patience = 30
        patience_increase = 30
        for epoch in range(1, EPOCHS+1):
            val_loss = train_ADANN(model, trainloaders, list_dictionaries_BN_weights, criterion, optimizer, weight_domain_loss, DEVICE)
            scheduler.step(val_loss)
            if val_loss + precision < best_loss:
                print("New best validation loss:", val_loss)
                best_loss = val_loss
                patience = patience_increase + epoch
                # add codes here to save model if necessary
                if save_required:
                     torch.save(model.state_dict(), params['weight_path']+params['ex_n']+'.bin')
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
                # add codes here to save model if necessary
                if save_required:
                    torch.save(model.state_dict(), params['weight_path']+params['ex_n']+'.bin')
            else:
                early_stopping_counter += 1
            if early_stopping_counter > early_stopping_iter:
                break
    return best_loss, dataloaders_adaptation, dataloaders_test

def objective(trial, params):
    params['lr'] = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    params['batch_size'] =  trial.suggest_int("batch_size", 256, 512, step = 256)

    temp_loss,_,_ = run_training(params)

    return temp_loss


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

        if not os.path.exists(study_path):
            os.makedirs(study_path)

        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        if params['ADANN_used']:
            params['ex_n'] = 'C'
        else:
            params['ex_n'] = 'B' if params['normalised_required'] else 'A'







        #  Study
        sampler = TPESampler()
        study = optuna.create_study(
            direction="minimize",  # maximize or minimize our objective
            sampler=sampler,  # parametrs sampling strategy
            pruner=MedianPruner(
                n_startup_trials=15,
                n_warmup_steps=5,  # let's say num epochs
                interval_steps=2,
            ),
            study_name='STUDY',
            storage="sqlite:///"+study_path+f"/{params['ex_n']}.db",  # storing study results, other storages are available too, see documentation.
            load_if_exists=True
        )

        study.optimize(lambda trial: objective(trial, params), n_trials=25)



        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        exit()
    

    '''
    for _, data in enumerate(dataloaders_test, 0):
        # get the inputs
        inputs, labels = data
        print(inputs.size())
        print(labels.size())
        exit()
    
    pretrain_raw_convNet(dataloaders_train, dataloaders_validation, filter_size=FILTER_SIZE)
    predictions, ground_truth = test_network_raw_convNet(dataloaders_adaptation, dataloaders_test, filter_size=FILTER_SIZE)

    '''





    participants_dataloaders_train, \
    participants_dataloaders_validation, \
    participants_dataloaders_test = \
        load_dataloaders(path="Dataset/processed_dataset", number_of_cycle=3,batch_size=256)
    participants_dataloaders_adaptation =  load_dataloaders_adaptation(path="Dataset/processed_dataset", batch_size=256)
    target_dataloaders_adaptation = participants_dataloaders_adaptation[0]
    target_dataloaders_test = participants_dataloaders_test[0]
    #pretrain_raw_convNet(participants_dataloaders_train[1:], participants_dataloaders_validation[1:], filter_size=FILTER_SIZE)
    predictions, ground_truth = test_network_raw_convNet(target_dataloaders_adaptation, target_dataloaders_test, filter_size=FILTER_SIZE)

    '''
    classes = ["Open Hand", "Power Grip", "Wrist Flexion", "Wrist Extension", "Wrist Pronation", "Wrist Supination"]
    font_size = 10
    sns.set(style='dark')

    fig, axs = print_confusion_matrix(ground_truth=ground_truth, predictions=predictions,
                                      class_names=classes, title="ConvNet using our normalisation method", fontsize=font_size)

    #fig.suptitle("ConvNet using AdaDANN training", fontsize=28)
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')  # works fine on Windows!
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.gcf().subplots_adjust(top=0.90)
    plt.savefig('temp_ours_result.png')
    '''




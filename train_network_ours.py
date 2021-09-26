import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Models.rawConvNet import Model
from Models.model_training import train_model_no_TL
from utils import print_confusion_matrix
from load_prepared_dataset_in_dataloaders import load_dataloaders_ours

CLASS_N = 6
BLOCK_N = 3
FILTER_SIZE = (1, 16)


def pretrain_raw_convNet(train_data, valid_data, filter_size=(1, 11)):
    dataloaders_train = train_data
    dataloaders_validation = valid_data
    # Define Model
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=0.35, filter_size=filter_size).cuda()

    # Define Loss functions
    cross_entropy_loss_classes = nn.CrossEntropyLoss(reduction='mean').cuda()

    # Define Optimizer
    learning_rate = 0.01#0.0404709
    print(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Define Scheduler
    precision = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15,
                                                     verbose=True, eps=precision)

    best_weights = train_model_no_TL(model=model, criterion=cross_entropy_loss_classes, optimizer=optimizer, scheduler=scheduler, dataloaders={"train": dataloaders_train, "val": dataloaders_validation}, precision=precision)

    weight_path = 'weights/ours'
    if not(os.path.exists(weight_path)):
        os.makedirs(weight_path)
    torch.save(best_weights, f=weight_path+"/TL_best_weights.pt")


def test_network_raw_convNet(test_data, 
                            path_weights = 'weights//ours/TL_best_weights.pt',
                            filter_size=(1,26)):
    with open("results/ours/test_accuracy_filter_size_" + str(filter_size[1]) + ".txt", "a") as myfile:
        myfile.write("Test")

    dataloaders_test = test_data
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=.35, filter_size=filter_size).cuda()
    best_weights = torch.load(path_weights)
    model.load_state_dict(best_weights)

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

if __name__ == "__main__":
    '''
    participants_dataloaders_train, \
    participants_dataloaders_validation, \
    participants_dataloaders_test = \
        load_dataloaders(path="Dataset/our_processed_dataset", number_of_cycle=3,batch_size=256)
    pretrain_raw_convNet(participants_dataloaders_train[:-1], participants_dataloaders_validation[:-1], filter_size=FILTER_SIZE)
    predictions, ground_truth = test_network_raw_convNet(participants_dataloaders_test, filter_size=FILTER_SIZE)
    '''

    train_dataset, validation_dataset, test_dataset = load_dataloaders_ours("Dataset/ours/processed_dataset/Participant1", number_of_cycle=3, batch_size=256)
    print(len(train_dataset.dataset))

    pretrain_raw_convNet(train_dataset, validation_dataset, filter_size=FILTER_SIZE)
    predictions, ground_truth = test_network_raw_convNet(test_dataset, filter_size=FILTER_SIZE)

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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from Models.rawConvNet import Model
from Models.model_training import pre_train_model
from utils import print_confusion_matrix
from load_prepared_dataset_in_dataloaders import load_dataloaders

CLASS_N = 6
BLOCK_N = 3
FILTER_SIZE = (1, 16)

def pretrain_raw_convNet(train_data, valid_data, filter_size=(1, 11)):
    participants_dataloaders_train = train_data
    participants_dataloaders_validation = valid_data
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=15,
                                                     verbose=True, eps=precision)

    best_weights, bn_statistics = pre_train_model(model=model, cross_entropy_loss_for_class=cross_entropy_loss_classes,
                                                  cross_entropy_loss_for_domain=cross_entropy_loss_domains, optimizer_class=optimizer,
                                                  scheduler=scheduler, dataloaders=
                                                  {"train": participants_dataloaders_train,
                                                   "val": participants_dataloaders_validation}, precision=precision)

    torch.save(best_weights, f="weights/TL_best_weights.pt")
    torch.save(bn_statistics, f="weights/bn_statistics.pt")

def test_network_raw_convNet(test_data, 
                            path_weights = 'weights/TL_best_weights.pt',
                            path_bn_statistics = "weights/bn_statistics.pt",
                            filter_size=(1,26)):
    with open("results/test_accuracy_filter_size_" + str(filter_size[1]) + ".txt", "a") as myfile:
        myfile.write("Test")

    participants_dataloaders_test = test_data
    model = Model(number_of_class=CLASS_N, number_of_blocks=BLOCK_N, number_of_channels=16, dropout_rate=.35, filter_size=filter_size).cuda()
    best_weights = torch.load(path_weights)
    model.load_state_dict(best_weights)
    list_dictionaries_bn_weights = torch.load(path_bn_statistics)

    predictions = []
    ground_truth = []
    accuracies = []
    for participant_index, bn_weights in enumerate(list_dictionaries_bn_weights):
        predictions_participant = []
        ground_truth_participant = []
        model.load_state_dict(bn_weights, strict=False)
        with torch.no_grad():
            model.eval()
            for inputs, labels in participants_dataloaders_test[participant_index]:
                inputs = inputs.cuda()
                output, _ = model(inputs)
                _, predicted = torch.max(output.data, 1)
                predictions_participant.extend(predicted.cpu().numpy())
                ground_truth_participant.extend(labels.numpy())
        print("Participant: ", participant_index, " Accuracy: ",
              np.mean(np.array(predictions_participant) == np.array(ground_truth_participant)))
        predictions.append(predictions_participant)
        ground_truth.append(ground_truth_participant)
        accuracies.append(np.mean(np.array(predictions_participant) == np.array(ground_truth_participant)))
    print("OVERALL ACCURACY: " + str(np.mean(accuracies)))

    with open("results/test_accuracy_filter_size_" + str(filter_size[1]) + ".txt", "a") as myfile:
        myfile.write("Predictions: \n")
        myfile.write(str(predictions) + '\n')
        myfile.write("Ground Truth: \n")
        myfile.write(str(ground_truth) + '\n')
        myfile.write("ACCURACIES: \n")
        myfile.write(str(accuracies) + '\n')
        myfile.write("OVERALL ACCURACY: " + str(np.mean(accuracies)))

    return predictions, ground_truth

if __name__ == "__main__":
    participants_dataloaders_train, \
    participants_dataloaders_validation, \
    participants_dataloaders_test = \
        load_dataloaders(path="Dataset/processed_dataset", number_of_cycle=3,batch_size=256)
    pretrain_raw_convNet(participants_dataloaders_train, participants_dataloaders_validation, filter_size=FILTER_SIZE)
    predictions, ground_truth = test_network_raw_convNet(participants_dataloaders_test, filter_size=FILTER_SIZE)

    classes = ["Open Hand", "Power Grip", "Wrist Flexion", "Wrist Extension", "Wrist Pronation", "Wrist Supination"]
    font_size = 10
    sns.set(style='dark')

    fig, axs = print_confusion_matrix(ground_truth=ground_truth, predictions=predictions,
                                      class_names=classes, title="ConvNet using ADANN training", fontsize=font_size)

    #fig.suptitle("ConvNet using AdaDANN training", fontsize=28)
    mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')  # works fine on Windows!
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.gcf().subplots_adjust(top=0.90)
    plt.savefig('temp_result.png')

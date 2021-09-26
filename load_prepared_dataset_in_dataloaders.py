import numpy as np
import torch
from torch.utils.data import TensorDataset


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []

    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])
    return new_examples, new_labels


def get_dataloader(examples_datasets, labels_datasets, cycle_used, batch_size=128, drop_last=True, shuffle=True, aggre=False):
    #  aggre = true: aggregating the data of all participants, used for no_TL
    #  aggre = False: used for ADANN
    dataloaders = []
    if aggre:
        X, Y = [], []
        for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
            for cycle in cycle_used:
                X.extend(participant_examples[cycle])
                Y.extend(participant_labels[cycle])
        X = np.expand_dims(X, axis=1)
        data = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)),
            torch.from_numpy(np.array(Y, dtype=np.int64)))
        dataloaders = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    else:
        for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
            X, Y = [], []
            for cycle in cycle_used:
                X.extend(participant_examples[cycle])
                Y.extend(participant_labels[cycle])
            X = np.expand_dims(X, axis=1)  # Main difference
            data = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)), torch.from_numpy(np.array(Y, dtype=np.int64)))
            examplesloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
            dataloaders.append(examplesloader)
    return dataloaders


def load_dataloaders(path, number_of_cycle=16, batch_size=128, valid_cycle_num=1, get_test_set=True, aggre_required=False):
    #  original function 'load_dataloader_for_training_without_TL' == get_test_set = False and aggre_required = True 
    participants_dataloaders_test = []
    if get_test_set:
        'Get testing dataset'
        datasets_test = np.load(path + "/RAW_INT_test.npy",allow_pickle=True)
        #  if across population
        #  datasets_test = np.load(path + "/RAW_AMP_test.npy",allow_pickle=True)
        examples_datasets_test, labels_datasets_test = datasets_test
        test_cycles = list(range(number_of_cycle))
        participants_dataloaders_test = get_dataloader(examples_datasets_test, labels_datasets_test, test_cycles, batch_size=batch_size, drop_last=True, shuffle=False, aggre=aggre_required)
    
    'Get training dataset'
    datasets_train = np.load(path + "/RAW_INT_train.npy",allow_pickle=True)
    examples_datasets_train, labels_datasets_train = datasets_train
    train_cycles = list(range(number_of_cycle))
    valid_cycles = []
    for _ in range(valid_cycle_num): # get validation cycles from training cycles in a reverse order
        valid_cycles.append(train_cycles.pop())
    participants_dataloaders_train = get_dataloader(examples_datasets_train,
                                                    labels_datasets_train, train_cycles,
                                                    batch_size=batch_size, drop_last=True,
                                                    shuffle=True, aggre=aggre_required)
    
    participants_dataloaders_valid = []
    if valid_cycle_num != 0:
        'Get validation dataset'  
        participants_dataloaders_valid = get_dataloader(examples_datasets_train,
                                                        labels_datasets_train, valid_cycles,
                                                        batch_size=batch_size, drop_last=True,
                                                        shuffle=True, aggre=aggre_required)
    return participants_dataloaders_train, participants_dataloaders_valid, participants_dataloaders_test


def load_dataloaders_adaptation(path, batch_size=128):
    #  original function 'load_dataloader_for_training_without_TL' == get_test_set = False and aggre_required = True
    participants_dataloaders_adaptation = []

    'Get adaptation dataset'
    datasets_train = np.load(path + "/RAW_INT_train.npy",allow_pickle=True)
    examples_datasets_train, labels_datasets_train = datasets_train
    ada_cycle = [0] # Take the first cycle as the adaptation data
    participants_dataloaders_adaptation = get_dataloader(examples_datasets_train,
                                                    labels_datasets_train, ada_cycle,
                                                    batch_size=batch_size, drop_last=True,
                                                    shuffle=True, aggre=False)
    return participants_dataloaders_adaptation

def load_dataloaders_ours(path, number_of_cycle=16, batch_size=128, get_test_set=True, aggre_required=True):
    #  original function 'load_dataloader_for_training_without_TL' == get_test_set = False and aggre_required = True 
    participants_dataloaders_test = []
    if get_test_set:
        'Get testing dataset'
        datasets_test = np.load(path + "/RAW_INT_test.npy",allow_pickle=True)
        #  if across population
        #  datasets_test = np.load(path + "/RAW_AMP_test.npy",allow_pickle=True)
        examples_datasets_test, labels_datasets_test = datasets_test
        test_cycles = list(range(number_of_cycle))
        participants_dataloaders_test = get_dataloader(examples_datasets_test, labels_datasets_test, test_cycles, batch_size=batch_size, drop_last=True, shuffle=False, aggre=aggre_required)
    'Get training dataset'
    datasets_train = np.load(path + "/RAW_INT_train.npy",allow_pickle=True)
    examples_datasets_train, labels_datasets_train = datasets_train
    train_cycles = list(range(number_of_cycle))
    participants_dataloaders_train = get_dataloader(examples_datasets_train,
                                                    labels_datasets_train, train_cycles,
                                                    batch_size=batch_size, drop_last=True,
                                                    shuffle=True, aggre=aggre_required)

    datasets_valid = np.load(path + "/RAW_INT_valid.npy",allow_pickle=True)
    examples_datasets_valid, labels_datasets_valid = datasets_valid
    valid_cycles = list(range(number_of_cycle))
    participants_dataloaders_valid = get_dataloader(examples_datasets_valid,
                                                    labels_datasets_valid, valid_cycles,
                                                    batch_size=batch_size, drop_last=True,
                                                    shuffle=True, aggre=aggre_required)
    return participants_dataloaders_train, participants_dataloaders_valid, participants_dataloaders_test



if __name__ == "__main__":

    train_dataset, validation_dataset, test_dataset = load_dataloaders_ours("Dataset/ours/processed_dataset/Participant1", number_of_cycle=3)
    #  Eech dataset: [participants * torch.utils.data.dataloader.DataLoader]
    print(len(train_dataset.dataset))
    print(len(validation_dataset.dataset))
    print(len(test_dataset.dataset))
    exit()
    train_dataset, validation_dataset, test_dataset = load_dataloaders("Dataset/processed_dataset", number_of_cycle=3, valid_cycle_num=1)
    #  Eech dataset: [participants * torch.utils.data.dataloader.DataLoader]
    print(len(train_dataset[0].dataset))
    print(len(validation_dataset[0].dataset))
    print(len(test_dataset[0].dataset))


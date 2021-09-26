import numpy as np
import torch
from torch.utils.data import TensorDataset

test_file = "FILTERED_test.npy"


data_path = "Dataset/Intact_dataset/NinaProDB5/Participant1/EMG/Processed/"

datasets_test = np.load(data_path+test_file, allow_pickle=True)

examples_datasets_test, labels_datasets_test = datasets_test

print(len(labels_datasets_test[0]))

cycle_used = [0]
#cycle_used = list(range(1,6))
dataloaders = []
X, Y = [], []
for participant_examples, participant_labels in zip(examples_datasets_test, labels_datasets_test):
    for cycle in cycle_used:
        X.extend(participant_examples[cycle])
        Y.extend(participant_labels[cycle])
    X = np.expand_dims(X, axis=1)
data = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)), torch.from_numpy(np.array(Y, dtype=np.int64)))
dataloaders = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False, drop_last=False)

for inputs, labels in dataloaders:
    print(labels)

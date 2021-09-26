import os
import numpy as np
import scipy.io as io
from scipy import signal
import torch
from torch.utils.data import TensorDataset
# from scipy.stats import wilcoxon
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import seaborn as sns
# import pandas as pd
# from sklearn.metrics import confusion_matrix as confusion_matrix_function

##  Functions used for data loading
#  Read from .mat file and save to multiple gesture_a_b.txt files where a is cycle index and b is gesture index
def data_prepare(input_dir, output_dir, sb_list, cycle_list, class_list, fs):
    for s_id in sb_list:
        datapath = input_dir +'/s'+str(s_id)+'/S'+str(s_id)+'_E2_A1.mat'
        emg = io.loadmat(datapath)['emg']
        label = io.loadmat(datapath)['restimulus']
        cycle = io.loadmat(datapath)['rerepetition']

        for rep_n in cycle_list:  # rep_n: repetition number
            for g in class_list:  # g: gesture index
                class_index = class_list.index(g)
                emg_data = emg[np.nonzero(np.logical_and(cycle==rep_n, label==g))[0]][:,0:16]
                if g == 0:
                    emg_data = emg_data[:fs*5]  #  take first 5s for the resting
                data_dir_sb = output_dir+"/Participant"+str(s_id)+'/EMG'
                if not os.path.exists(data_dir_sb):
                    os.makedirs(data_dir_sb)
                np.savetxt(data_dir_sb+'/gesture_%d_%d.txt' % (rep_n, class_index), emg_data, fmt='%+.2f', delimiter=',')
    # eg, Dataset/Intact_dataset/NinaProDB5/Participant1/EMG/gesture_1_1.txt
    return

#  Read emg from the saved txt file
def read_emg_from_txt(path_emg, gesture_index):
    # path_emg example:  ../gesture_1_
    examples_to_format = []
    for line in open(path_emg + '%d.txt' % gesture_index):
        emg_signal = np.float32(line.strip().split(","))
        examples_to_format.append(emg_signal)
    #  type(examples_to_format): list
    #  np.shape(examples_to_format):  [samples, channels]
    return examples_to_format

#  segment the emg data, 
def segment_emg(path_emg, number_of_gestures, window_size, size_non_overlap, ref_max=None, ref_min=None, fs=None):
    examples = []
    labels = []
    for gesture_index in range(number_of_gestures):
        examples_to_format = read_emg_from_txt(path_emg, gesture_index)
        if fs is not None: # For convinience, filter the whole dataset and do the segmentation
            examples_to_format = np.transpose(filter_data(np.transpose(examples_to_format), fs=fs))
        # print(np.shape(examples_to_format)) # (604, 16), list
        if ref_max is not None:
            examples_to_format = np.float32((examples_to_format - np.min(examples_to_format,axis=0)) / (np.max(examples_to_format, axis=0)- np.min(examples_to_format, axis=0)) * (ref_max[gesture_index] - ref_min[gesture_index]) + ref_min[gesture_index])

        examples_formatted = format_examples(examples_to_format, window_size=window_size, size_non_overlap=size_non_overlap)
        examples.extend(examples_formatted)
        labels.extend(np.ones(len(examples_formatted)) * (gesture_index))
    return examples, labels

#  filter the emg using butter bandpass filter
def filter_data(emg_sample, fs):
    #  emg.shape():  (channel, samples)
    filtered_emg_sample = []
    sos = signal.butter(4, 20, 'hp', fs=fs, output='sos')
    for channel_emg in emg_sample:
        # channel_filtered = butter_bandpass_filter(channel_emg, lowcut=20, highcut=495, fs=1000, order=4)
        channel_filtered = signal.sosfilt(sos, channel_emg)
        filtered_emg_sample.append(channel_filtered)
    return filtered_emg_sample

'''
#  butter bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    lowcut_normalized = lowcut / nyq
    highcut_normalized = highcut / nyq
    b, a = signal.butter(N=order, Wn=[lowcut_normalized, highcut_normalized], btype='band', output="ba")
    y = signal.lfilter(b, a, data)
    return y
'''

def format_examples(emg_examples, window_size=150, size_non_overlap=50):
    #  emg_examples shape: [samples * channels]
    examples_formatted = []
    example = []
    for emg_vector in emg_examples:
        if len(example) == 0:
            example = emg_vector
        else:
            example = np.row_stack((example, emg_vector))
        if len(example) >= window_size:
            # The example is of the shape TIME x CHANNEL. Make it of the shape CHANNEL x TIME
            example = example.transpose()
            examples_formatted.append(example)
            # back to TIME x CHANNEL
            example = example.transpose()
            # Remove part of the data of the example according to the size_non_overlap variable
            example = example[size_non_overlap:]
    return examples_formatted


def processing_data(input_dir, target_sb, sb_list, cycle_list, gesture_n, window_size, size_non_overlap, fs=None):
    target_path = input_dir + '/Participant' + str(target_sb) + '/EMG'

    examples_train, labels_train = [], []
    examples_train_nor, labels_train_nor = [], []
    examples_test, labels_test = [], []

    target_adapt_emg = target_path + '/gesture_1_'  # 1 means the first cycle
    #  Get the new max and new min from the first cycle of the target subject
    ref_max, ref_min = [], []
    for i in range(gesture_n):
        examples_to_format = read_emg_from_txt(target_adapt_emg, i)
        # i.e., (samples, channels)
        if fs is not None:
            examples_to_format = filter_data(np.transpose(examples_to_format),fs)
            examples_to_format = np.transpose(examples_to_format)
            #  back to (samples, channels)
        ref_max.append(np.max(examples_to_format,axis=0))
        ref_min.append(np.min(examples_to_format,axis=0))
    # the shape of ref_max and ref_min:  [gesture_n, channel_n]

    #  Get the training(including validation at this stage), and testing ready
    for sb in sb_list:
        if sb == target_sb:  # get the testing data
            examples_per_test, labels_per_test = [], []  # per subject
            #  Prepare for the testing data
            for cycle in cycle_list:
                target_emg = target_path + '/gesture_%d_' % (cycle)
                examples_t, labels_t = segment_emg(target_emg, gesture_n, window_size, size_non_overlap, ref_max=None, ref_min=None, fs=fs)  # t: target domain
                examples_per_test.append(examples_t)
                labels_per_test.append(labels_t)
            examples_test.append(examples_per_test)
            labels_test.append(labels_per_test)
        else:
        #  Prepare for the training data
            examples_per_train, labels_per_train = [], []
            examples_per_train_nor, labels_per_train_nor = [], []
            for cycle in cycle_list:
                path_emg_train = input_dir + '/Participant%d/EMG/gesture_%d_' %(sb, cycle)
                #  segmentation before without normalisation
                examples_s, labels_s = segment_emg(path_emg_train, gesture_n, window_size, size_non_overlap, ref_max=None, ref_min=None, fs=fs)  # s: source domain
                examples_per_train.append(examples_s)
                labels_per_train.append(labels_s)
                #  segmentation before normalisation
                examples_s_nor, labels_s_nor = segment_emg(path_emg_train, gesture_n, window_size, size_non_overlap,ref_max=ref_max,ref_min=ref_min, fs=fs)
                examples_per_train_nor.append(examples_s_nor)
                labels_per_train_nor.append(labels_s_nor)
            examples_train.append(examples_per_train)
            labels_train.append(labels_per_train)
            examples_train_nor.append(examples_per_train_nor)
            labels_train_nor.append(labels_per_train_nor)
    #  examples_train: (sb * cycle * seg * channel * samples)
    raw_train = (examples_train, labels_train)
    raw_nor_train = (examples_train_nor, labels_train_nor)
    raw_test = (examples_test, labels_test)
    return raw_train, raw_nor_train, raw_test


def data_process(input_dir, sb_list,cycle_list, gesture_n, window_size, size_non_overlap, fs):
    #  Process the data by each subject
    #  Two Processings: without and with data filtering
    #  The processed data will be saved into npy files
    for s_id in sb_list:
        output_dir = input_dir + '/Participant%d/EMG/Processed/' % s_id
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #  e.g., Dataset/Intact_dataset/NinaProDB5/Participant1/EMG/Processed/
        raw_train, raw_nor_train, raw_test = processing_data(input_dir, s_id, sb_list, cycle_list, gesture_n, window_size, size_non_overlap, fs=None)
        np.save(output_dir+'RAW_train.npy', raw_train)
        np.save(output_dir+'RAW_nor_train.npy', raw_nor_train)
        np.save(output_dir+'RAW_test.npy', raw_test)

        # save RAW_test.npy, RAW_train.npy and RAW_nor_train.npy
        filtered_train, filtered_nor_train, filtered_test = processing_data(input_dir, s_id, sb_list, cycle_list, gesture_n, window_size, size_non_overlap, fs=fs)

        np.save(output_dir+'FILTERED_train.npy', filtered_train)
        np.save(output_dir+'FILTERED_nor_train.npy', filtered_nor_train)
        np.save(output_dir+'FILTERED_test.npy', filtered_test)
    return

def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])
    return new_examples, new_labels

def get_dataloader(examples_datasets, labels_datasets, cycle_used, batch_size=128, shuffle=True, aggre=False):
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
        dataloaders = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    else:
        for participant_examples, participant_labels in zip(examples_datasets, labels_datasets):
            X, Y = [], []
            for cycle in cycle_used:
                X.extend(participant_examples[cycle])
                Y.extend(participant_labels[cycle])
            X = np.expand_dims(X, axis=1)  # Main difference
            data = TensorDataset(torch.from_numpy(np.array(X, dtype=np.float32)), torch.from_numpy(np.array(Y, dtype=np.int64)))
            examplesloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=False)
            dataloaders.append(examplesloader)
    return dataloaders

def load_dataloaders_DB5(path, batch_size=128, filtered=True, normalised=False, aggre_required=True):
    #  The default setting for NinaProDB5
    #  adaptation data: cycle 1 from the target sb
    ada_cycles = [0]
    #  testing data: cycle 2-6 from the target sb
    test_cycles = list(range(1,6))
    #  training data: cycles 1, 3, 4, and 6 from the source sbs
    train_cycles = [0, 2, 3, 5]
    #  validation data: cycles 2 and 5 from the source sbs
    valid_cycles = [1, 4]

    if filtered:
        testfile = 'FILTERED_test.npy'
        trainfile = 'FILTERED_nor_train.npy' if normalised else 'FILTERED_train.npy'
    else:
        testfile = 'RAW_test.npy'
        trainfile = 'RAW_nor_train.npy' if normalised else 'RAW_train.npy'

    participants_dataloaders_test = []
    'Get testing dataset'
    datasets_test = np.load(path+testfile, allow_pickle=True)
    examples_datasets_test, labels_datasets_test = datasets_test
    #participants_dataloaders_test = get_dataloader(examples_datasets_test, labels_datasets_test, test_cycles, batch_size=batch_size, shuffle=False, aggre=aggre_required)
    participants_dataloaders_test = get_dataloader(examples_datasets_test, labels_datasets_test, test_cycles, batch_size=batch_size, shuffle=False, aggre=True)
    'Get adaptation dataset'
    #participants_dataloaders_ada = get_dataloader(examples_datasets_test, labels_datasets_test, ada_cycles, batch_size=batch_size, shuffle=False, aggre=aggre_required)
    participants_dataloaders_ada = get_dataloader(examples_datasets_test, labels_datasets_test, ada_cycles, batch_size=batch_size, shuffle=False, aggre=True)

    'Get training dataset'
    datasets_train = np.load(path+trainfile, allow_pickle=True)
    examples_datasets_train, labels_datasets_train = datasets_train
    participants_dataloaders_train = get_dataloader(examples_datasets_train,
                                                    labels_datasets_train, train_cycles,
                                                    batch_size=batch_size, 
                                                    shuffle=True, aggre=aggre_required)
    'Get validation dataset'
    participants_dataloaders_valid = get_dataloader(examples_datasets_train,
                                                    labels_datasets_train, valid_cycles,
                                                    batch_size=batch_size, 
                                                    shuffle=True, aggre=aggre_required)
    return participants_dataloaders_train, participants_dataloaders_valid, participants_dataloaders_ada, participants_dataloaders_test


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


'''
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def print_confusion_matrix(ground_truth, predictions, class_names, fontsize=24,
                           normalize=True, fig=None, axs=None, title=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """

    print(np.shape(predictions[0]))
    # Calculate the confusion matrix across all participants
    predictions = [x for y in predictions for x in y]
    ground_truth = [x for y in ground_truth for x in y]
    #predictions = [x for y in predictions for x in y]
    #ground_truth = [x for y in ground_truth for x in y]
    print(np.shape(ground_truth))
    confusion_matrix_calculated = confusion_matrix_function(np.ravel(np.array(ground_truth)),
                                                            np.ravel(np.array(predictions)))

    if normalize:
        confusion_matrix_calculated = confusion_matrix_calculated.astype('float') /\
                                      confusion_matrix_calculated.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        print("Normalized confusion matrix")
    else:
        fmt = 'd'
        print('Confusion matrix, without normalization')
    df_cm = pd.DataFrame(
        confusion_matrix_calculated, index=class_names, columns=class_names,
    )

    print(confusion_matrix_calculated)
    max_confusion_matrix = np.max(confusion_matrix_calculated)
    min_confusion_matrix = np.min(confusion_matrix_calculated)
    cmap = plt.get_cmap("magma")
    new_cmap = truncate_colormap(cmap, min_confusion_matrix, max_confusion_matrix)

    print(max_confusion_matrix, "  ", min_confusion_matrix)

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cbar=False, annot_kws={"size": fontsize}, cmap=new_cmap)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.set(
        # ... and label them with the respective list entries
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right', fontsize=fontsize)
    heatmap.xaxis.label.set_size(fontsize + 4)
    heatmap.yaxis.label.set_size(fontsize + 4)
    heatmap.title.set_size(fontsize + 6)
    return fig, axs
'''


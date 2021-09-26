import os
import numpy as np

#from PrepareAndLoadData.prepare_dataset_utils import butter_bandpass_filter
from prepare_dataset_utils import butter_bandpass_filter





def get_data_and_process_it_from_file(get_train_data, path, number_of_gestures=7, number_of_cycles=16, window_size=151,
                                      size_non_overlap=50):
    examples_datasets, labels_datasets = [], []
    train_or_test_str = "train" if get_train_data else "test"

    participants_directories = os.listdir(path)
    for participant_directory in participants_directories:
        print("Preparing data of: " + participant_directory)

        examples_participants, labels_participant = [], []
        for cycle in range(number_of_cycles):
            path_emg = path + "/" + participant_directory + "/" + "%s/EMG/gesture_%d_" % (train_or_test_str, cycle)
            examples, labels = [], []
            for gesture_index in range(1,number_of_gestures+1):  # modify if including resting
                examples_to_format = read_emg_from_txt(path_emg, gesture_index)
                # print(np.shape(examples_to_format)) # (604, 16)
                examples_formatted = format_examples(examples_to_format, window_size=window_size, size_non_overlap=size_non_overlap)
                examples.extend(examples_formatted)
                labels.extend(np.ones(len(examples_formatted)) * (gesture_index-1)) #  remove '-1' when including resting 
            examples_participants.append(examples)
            labels_participant.append(labels)

        examples_datasets.append(examples_participants)
        labels_datasets.append(labels_participant)

    #print(np.shape(examples_datasets)) # sb * cycle * seg * channel * samples
    #print(np.shape(labels_datasets))
    return examples_datasets, labels_datasets





def get_data_and_process_it_from_file_ours(target_sb, path, number_of_gestures=7, number_of_cycles=16, window_size=151,
                                      size_non_overlap=50):
    examples_train, labels_train = [], []
    examples_valid, labels_valid = [], []
    examples_test, labels_test = [], []
    #  Get the new max and new min from the first cycle of the target subject
    path_emg = path + "/Participant" + str(target_sb) + "/train/EMG/gesture_0_"
    ref_max, ref_min = [], []
    for gesture_index in range(1,number_of_gestures+1):  # modify if including resting
        examples_to_format = read_emg_from_txt(path_emg, gesture_index)
        # (604, 16)
        ref_max.append(np.max(examples_to_format,axis=0))
        ref_min.append(np.min(examples_to_format,axis=0))
    # the shape of ref_max and ref_min:  [gesture_n, channel_n]

    #  Get the training, validation, and testing ready
    #  For the target subject, no normalisation is required
    #  For other subjects, half of the total cycles are used for training and validation set. Normalisation is required.
    participants_directories = os.listdir(path)
    for participant_directory in participants_directories:
        if participant_directory == 'Participant'+str(target_sb):
            examples_participant, labels_participant = [], []
            #  Prepare for the testing data
            for cycle in range(number_of_cycles):
                path_emg = path + "/Participant" + str(target_sb) + "/test/EMG/gesture_%d_" % (cycle)
                examples, labels = segment_emg_gestures(path_emg, number_of_gestures, window_size, size_non_overlap)
                examples_participant.append(examples)
                labels_participant.append(labels)
            examples_test.append(examples_participant)
            labels_test.append(labels_participant)
        else:
        #  Prepare for the training and validation data
            examples_participant_train, labels_participant_train = [], []
            examples_participant_valid, labels_participant_valid = [], []
            for cycle in range(number_of_cycles):
                path_emg_train = path + "/" + participant_directory + "/train/EMG/gesture_%d_" % (cycle)
                examples, labels = segment_emg_gestures(path_emg_train, number_of_gestures, window_size, size_non_overlap,ref_max=ref_max,ref_min=ref_min) #  segmentation before normalisation

                examples_participant_train.append(examples)
                labels_participant_train.append(labels)

                path_emg_valid = path + "/" + participant_directory + "/test/EMG/gesture_%d_" % (cycle)
                examples, labels = segment_emg_gestures(path_emg_valid, number_of_gestures, window_size, size_non_overlap,ref_max=ref_max,ref_min=ref_min) #  segmentation before normalisation
                examples_participant_valid.append(examples)
                labels_participant_valid.append(labels)
            examples_train.append(examples_participant_train)
            labels_train.append(labels_participant_train)
            examples_valid.append(examples_participant_train)
            labels_valid.append(labels_participant_train)
    #print(np.shape(examples_datasets)) # sb * cycle * seg * channel * samples
    #print(np.shape(labels_datasets))
    return examples_train, labels_train, examples_valid, labels_valid, examples_test, labels_test


def read_data(path, number_of_gestures=7, number_of_cycles=16, window_size=200, size_non_overlap=50):
    print("Loading and preparing datasets...")
    'Get and process the train data'
    print("Taking care of the training data...")
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_it_from_file(get_train_data=True, path=path, number_of_gestures=number_of_gestures, number_of_cycles=number_of_cycles, window_size=window_size, size_non_overlap=size_non_overlap)
    np.save("Dataset/ours/processed_dataset/RAW_INT_train.npy", (list_dataset_train_emg, list_labels_train_emg))
    print("Finished with the training data...")
    'Get and process the test data'
    print("Starting with the test data...")
    list_dataset_train_emg, list_labels_train_emg = get_data_and_process_it_from_file(get_train_data=False, path=path, number_of_gestures=number_of_gestures, number_of_cycles=number_of_cycles, window_size=window_size, size_non_overlap=size_non_overlap)
    np.save("Dataset/processed_dataset/RAW_INT_test.npy", (list_dataset_train_emg, list_labels_train_emg))
    print("Finished with the test data")


def read_data_ours(path, number_of_gestures=7, number_of_cycles=16, window_size=200, size_non_overlap=50):
    target_sb=1
    print("Target subject: ", target_sb)
    examples_train, labels_train, examples_valid, labels_valid, examples_test, labels_test = get_data_and_process_it_from_file_ours(target_sb=target_sb, path=path, number_of_gestures=number_of_gestures, number_of_cycles=number_of_cycles, window_size=window_size, size_non_overlap=size_non_overlap)

    datapath = "Dataset/ours/processed_dataset/Participant"+str(target_sb)
    if not(os.path.exists(datapath)):
        os.makedirs(datapath)
    np.save(datapath+"/RAW_INT_train.npy", (examples_train, labels_train))
    np.save(datapath+"/RAW_INT_valid.npy", (examples_valid, labels_valid))
    np.save(datapath+"/RAW_INT_test.npy", (examples_test, labels_test))


if __name__ == '__main__':
    #read_data(path="Dataset/Intact_dataset", number_of_gestures=6,number_of_cycles=3,window_size=50,size_non_overlap=15)
    read_data_ours(path="Dataset/Intact_dataset", number_of_gestures=6,number_of_cycles=3,window_size=50,size_non_overlap=15)



import os
import numpy as np
import utils





if __name__ == "__main__":

    data_prepare_required = True  # Set it as True for the first time use

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
    process_params['gestures_n'] = len(prepare_params['class_list'])
    process_params['fs'] = prepare_params['fs']
    process_params['window_size'] = 50
    process_params['size_non_overlap'] = 15


    if data_prepare_required:
        utils.data_prepare(**prepare_params)
        utils.data_process(**process_params)


    filtered = True
    normalised = True
    ADANN_used = True

    target_sb = 1
    path = 'Dataset/Intact_dataset/NinaProDB5/Participant%d/EMG/Processed/' % target_sb
    participants_dataloaders_train, participants_dataloaders_valid, participants_dataloaders_ada, participants_dataloaders_test = utils.load_dataloaders_DB5(path, batch_size=128, filtered=filtered, normalised=normalised, aggre_required=not ADANN_used)


    # Read data 
    
    # target sb
    # train_cycle
    # valid_cycle
    # test_cycle
    # adaptation_cycle = []
    # window_size = 50
    # size_non_overlap = 15



    #  Output:
    #  -Dataset/Intact_dataset/NinaProDB5/Participant1/EMG/Processed/
    #  FILTERED_test.npy
    #  RAW_test.npy
    #  FILTERED_train.npy
    #  RAW_train.npy
    #  FILTERED_nor_train.npy
    #  RAW_nor_train.npy
    #  get train and validation data from _train.npy
    #  get adaptation and testing data from _test.npy







    # At this stage, the only difference is normalisation or not 
    # and filter or not


    #  Dataloading can be divided into
    #  A: aggre without normalisation
    #  B: aggre with normalisation
    #  C: ADANN


    #  Baseline
    #  Ours
    #  ADANN without adaptation
    #  ADANN with adaptation
import numpy as np

emg_path = 'Dataset\\Intact_dataset\\NinaProDB5\\Participant1\\EMG\\Processed\\'

raw_test_X, raw_test_Y = np.load(emg_path+'RAW_test.npy',allow_pickle=True)

print(np.shape(raw_test_X))

g = 2
X_test = np.array(raw_test_X[0, 0])
Y_test = np.array(raw_test_Y[0, 0])
X_test_g0 = X_test[Y_test==g]
X_test_max_g0 = np.max(X_test_g0, axis=(0,2))

#print(np.amax(raw_test[0,0],axis=1))

raw_nor_train_X, raw_nor_train_Y = np.load(emg_path+'RAW_nor_train.npy',allow_pickle=True)

print(np.shape(raw_nor_train_X))
s = 4
c = 4

X_raw_nor_train = np.array(raw_nor_train_X[s, c])
Y_raw_nor_train = np.array(raw_nor_train_Y[s, c])
X_nor_train_g0 = X_raw_nor_train[Y_raw_nor_train==g]
X_nor_train_max_g0 = np.max(X_nor_train_g0, axis=(0,2))

raw_train_X, raw_train_Y = np.load(emg_path+'RAW_train.npy',allow_pickle=True)
X_raw_train = np.array(raw_train_X[s, c])
Y_raw_train = np.array(raw_train_Y[s, c])
X_train_g0 = X_raw_train[Y_raw_train==g]
X_train_max_g0 = np.max(X_train_g0, axis=(0,2))

print(X_test_max_g0)
print(X_nor_train_max_g0)
print(X_train_max_g0)

print(X_raw_nor_train[0,0])
print(X_raw_train[0,0])

filtered_test, filtered_test_Y = np.load(emg_path+'FILTERED_test.npy',allow_pickle=True)
X_filtered_test = np.array(filtered_test[0, 0])
Y_filtered_test = np.array(filtered_test_Y[0, 0])
X_filtered_test_g0 = X_filtered_test[Y_filtered_test==g]
X_filtered_test_max_g0 = np.max(X_filtered_test_g0, axis=(0,2))

filtered_nor_train, filtered_nor_train_Y = np.load(emg_path+'FILTERED_nor_train.npy',allow_pickle=True)
X_filtered_nor_train = np.array(filtered_nor_train[s, c])
Y_filtered_nor_train = np.array(filtered_nor_train_Y[s, c])
X_filtered_nor_train_g0 = X_filtered_nor_train[Y_filtered_nor_train==g]
X_filtered_nor_train_max_g0 = np.max(X_filtered_nor_train_g0, axis=(0,2))

filtered_train, filtered_train_Y = np.load(emg_path+'FILTERED_train.npy',allow_pickle=True)

X_filtered_train = np.array(filtered_train[s, c])
Y_filtered_train = np.array(filtered_train_Y[s, c])
X_filtered_train_g0 = X_filtered_train[Y_filtered_train==g]
X_filtered_train_max_g0 = np.max(X_filtered_train_g0, axis=(0,2))


print(X_filtered_test_max_g0)
print(X_filtered_nor_train_max_g0)
print(X_filtered_train_max_g0)

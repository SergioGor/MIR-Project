import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class musDB_class_dataset(Dataset):

    def __init__(self, root_dir, tcontext, split, mode, augment):
        self.root_dir = root_dir
        self.tcontext = tcontext
        self.split = split
        self.mode = mode
        self.augment = augment
        with h5py.File(self.root_dir, 'r') as db:
            # At which frequency index we split into train and val set:
            self.train_end_f_ind = db['f_indexes'][+
            np.where(db['f_indexes'][...] >= int(db['f_indexes'][-1]) * self.split)[0][0]]
            # Number of STFT bins for the validation set
            self.val_stftbins = db['f_indexes'][-1] - self.train_end_f_ind
            self.f_indexes = db['f_indexes'][...]

    def __len__(self):
        # Returns the length of the dataset
        if self.mode == 'train':
            lens = int((self.train_end_f_ind / self.tcontext))
        else:
            lens = int(self.val_stftbins / self.tcontext)
        return lens

    def __getitem__(self, idx):
        # get the slice using the index idx
        with h5py.File(self.root_dir, 'r') as db:
            if self.mode == 'train':
                reader_head = idx
            elif self.mode == 'val':
                reader_head = idx + int(self.train_end_f_ind / self.tcontext)
            track_mag = db['track_mag'][
                        int(reader_head * self.tcontext):int(reader_head * self.tcontext + self.tcontext)]
            track_mag = np.expand_dims(track_mag, 0)
            if (((np.random.rand() >=0.5) and self.augment) and self.mode=='train'):
                track_mag = np.fliplr(track_mag)
            label = db['label'][reader_head]
            std=np.tile(db['std'][...][0, :], (1, self.tcontext, 1))
            track_mag=(track_mag)/std
            track_mag = torch.from_numpy(track_mag)

            sample = {'input': track_mag, 'label': label}

        return sample
import numpy as np
import config
from utils import *
from processing import ideal_ratio_mask, apply_mask

class DatasetGenerator(object):
    def __init__(self, mode, type, dataset=None, magphase=False,
                 bin_start=0, bin_stop=256, timeframes_in=10, timeframes_out=1, batch_size=256, mask='IRM', mask_method='naive', train_val_split=0.85, shuffle_validation=False, persa=False, print_fn=True):
        assert type in ['train', 'test']
        assert dataset in ['TSP', 'CSTR', 'TIMIT', 'LibriSpeech']
        assert mask in ['IRM', None]
        assert bin_start >= 0
        self.timeframes_in = timeframes_in
        self.timeframes_out = timeframes_out
        self.mode = mode
        if self.mode is 'STFT': assert bin_stop <= fft_bins
        self.batch_size = batch_size
        self._type_ = type
        self._dataset_ = dataset
        self._bin_start = bin_start
        self._bin_stop = bin_stop
        self._magphase = magphase
        self._mask = mask
        self.mask_method = mask_method
        self._train_val_split = train_val_split
        self._shuffle_val = shuffle_validation
        self._persa = persa
        dir_clean = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\clean_' + self._type_ + 'set' + FOLDER_EXT
        dir_noisy = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\noisy_' + self._type_ + 'set' + FOLDER_EXT
        list_speech = [k for k in get_list_of_files(dir_clean) if '.npy' in k]
        list_noisy = [k for k in get_list_of_files(dir_noisy) if '.npy' in k]
        list_speech.sort()  # Sort the files
        list_noisy.sort()  #

        if self._shuffle_val:
            ind =  np.random.randint(len(list_speech)-1)
            print('Shuffle to:', ind)
            indices = np.roll(np.arange(len(list_speech)),ind)
            list_speech = np.array(list_speech)[indices]
            list_noisy = np.array(list_noisy)[indices]

        if self._type_ is 'train':
            self.list_speech = np.array(list_speech[:int(len(list_speech) * self._train_val_split)])
            self.list_noisy = np.array(list_noisy[:int(len(list_noisy) * self._train_val_split)])
            self.list_speech_val = np.array(list_speech[int(len(list_speech) * self._train_val_split):])
            self.list_noisy_val = np.array(list_noisy[int(len(list_noisy) * self._train_val_split):])
            if print_fn:
                print('Loading:', self.mode)
                print(self._dataset_ + ' ' + self._type_ + ':\t', len(list_speech), 'Speech files\t-\t validation: %d Speech files' % len(self.list_speech_val))
                print('\t\t\t\t', len(list_noisy), 'Noise files\t-\t\t\t\t %d Noise files' % len(self.list_noisy_val))
                print \
                    ('__________________________________________________________________________________________________')
        else:
            self.list_speech = np.array(list_speech)
            self.list_noisy = np.array(list_noisy)
            self.list_speech_val = self.list_noisy_val = []
            if print_fn:
                print('Loading:')
                print(self._dataset_ + ' ' + self._type_ + ':\t', len(list_speech) ,'Speech files')
                print('\t\t\t', len(list_noisy), 'Noise files')
                print \
                    ('__________________________________________________________________________________________________')

    def get_folder_path(self, label='clean', filetype='wav'):
        assert filetype in ['wav', 'sp']
        assert label in ['clean','noisy']
        return os.path.join(FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_, label + '_' + self._type_ + 'set_' + filetype + '\\')

    def get_total_samples(self, isValidation=False):
        if self._type_ == 'test':
            assert isValidation is False
            return len(self.list_speech)
        return len(self.list_speech_val) if isValidation else len(self.list_speech)

    def load_dataset(self, isValidation=False):
        if self.mode is 'STFT':
            return self.load_dataset_stft(isValidation)
        elif self.mode is 'ENV':
            return self.load_dataset_env(isValidation)
        elif self.mode is 'ENVTFS':
            return self.load_dataset_envtfs(isValidation)
        else:
            return

    def generate_test_sample(self, random_order=True):
        if self.mode is 'STFT':
            return self.generate_test_sample_stft(random_order)
        elif self.mode is 'ENV':
            return self.generate_test_sample_env(random_order)
        elif self.mode is 'ENVTFS':
            return self.generate_test_sample_envtfs(random_order)
        else:
            return

    def generate_stft(self, isValidation=False):
        if self._type_ == 'test':
            assert isValidation is False

        list_speech = self.list_speech_val if isValidation else self.list_speech
        list_noisy = self.list_noisy_val if isValidation else self.list_noisy

        x_train_batch = []
        y_train_batch = []
        sp_speech = np.array([])
        sp_noisy = np.array([])
        iterations = 0
        index = 0
        while True:
            # shape (BATCH_SIZE, bin_stop-bin_start, TIME_FRAMES, mag/phase)
            for _ in range(64):
                speech_sample = np.load(list_speech[index])
                noise_sample = np.load(list_noisy[index])
                if self._persa:
                    norm_factor = 1 / np.maximum(np.amax(speech_sample[..., 0]), np.amax(noise_sample[..., 0]))
                    speech_sample[..., 0], noise_sample[..., 0] = speech_sample[..., 0] * norm_factor, noise_sample[..., 0] * norm_factor
                sp_speech = np.concatenate((sp_speech, speech_sample), axis=1) if sp_speech.size else speech_sample
                sp_noisy = np.concatenate((sp_noisy, noise_sample), axis=1) if sp_noisy.size else noise_sample
                index += 1
                if index == len(list_speech):
                    index = 0
                    indices = np.random.permutation(len(list_speech))
                    list_speech = list_speech[indices]
                    list_noisy = list_noisy[indices]
            sp_noisy = np.pad(sp_noisy, ((0, 0), (self.time_frames_in - self.timeframes_out, 0), (0, 0)))
            sp_speech = np.pad(sp_speech, ((0, 0), (self.time_frames_in - self.timeframes_out, 0), (0, 0)))
            for k, i in enumerate(range(0, sp_speech.shape[1] - self.timeframes_in, 1)): # step is 1 for ensembling
                x_train_batch.append(sp_noisy[BIN_START:BIN_STOP, i: i + self.timeframes_in, :])
                y_train_batch.append(sp_speech[BIN_START:BIN_STOP, i + self.timeframes_in - self.timeframes_out: i + self.timeframes_in, :])

            # permute array
            indices = np.random.permutation(len(x_train_batch))
            x_train_batch = np.array(x_train_batch)[indices].tolist()
            y_train_batch = np.array(y_train_batch)[indices].tolist()

            while len(x_train_batch) > self.batch_size:
                ind = random.sample(range(0,len(x_train_batch)),self.batch_size)
                x, y = x_train_batch[:self.batch_size], y_train_batch[:self.batch_size]
                if self._mask is 'IRM':
                    y[:, :, :, 0] = ideal_ratio_mask(x[:, :, self.timeframes_in - self.timeframes_out:, 0], y[:, :, :, 0], self.mask_method)
                if not self._magphase:
                    # drop the phase and prepare shape for the model
                    x = np.expand_dims(np.array(x)[..., 0], axis=-1)
                    y = np.expand_dims(np.array(y)[..., 0], axis=-1)
                yield np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
                x_train_batch, y_train_batch = x_train_batch[self.batch_size:], y_train_batch[self.batch_size:]
                iterations += 1

    def load_dataset_stft(self, isValidation=False):
        if self._type_ == 'test':
            assert isValidation is False

        list_speech = self.list_speech_val if isValidation else self.list_speech
        list_noisy = self.list_noisy_val if isValidation else self.list_noisy

        x_train = []
        y_train = []
        # shape (BATCH_SIZE, bin_stop-bin_start, TIME_FRAMES, mag/phase)
        for index in range(len(list_speech)):
            # (FFT_BINS, time, mag/phase)
            sp_speech = np.load(list_speech[index])
            sp_noisy = np.load(list_noisy[index])

            if self._persa:
                norm_factor = 1 / np.maximum(np.amax(sp_speech[...,0]), np.amax(sp_noisy[...,0]))
                sp_speech[..., 0], sp_noisy[..., 0] = sp_speech[..., 0] * norm_factor, sp_noisy[..., 0] * norm_factor

            if not self._magphase:
                # drop the phase and prepare shape for the model
                sp_noisy = sp_noisy[:, :, 0][:, :, None]
                sp_speech = sp_speech[:, :, 0][:, :, None]

            if self._mask is 'IRM':
                sp_speech[:, :, 0] = ideal_ratio_mask(sp_noisy[:, :, 0], sp_speech[:, :, 0], self.mask_method)

            sp_noisy = np.pad(sp_noisy, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            sp_speech = np.pad(sp_speech, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            for k, i in enumerate(range(0, sp_speech.shape[1] - self.timeframes_in, 1)):
                x_train.append(sp_noisy[BIN_START:BIN_STOP, i: i + self.timeframes_in, :])
                y_train.append(sp_speech[BIN_START:BIN_STOP, i + self.timeframes_in - self.timeframes_out: i + self.timeframes_in, :])

        assert not np.isnan(np.sum(x_train))
        assert not np.isnan(np.sum(y_train))
        return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def load_dataset_env(self, isValidation=False):
        if self._type_ == 'test':
            assert isValidation is False

        list_speech = self.list_speech_val if isValidation else self.list_speech
        list_noisy = self.list_noisy_val if isValidation else self.list_noisy

        x_train = []
        y_train = []
        # shape (BATCH_SIZE, freq, time)
        for index in range(len(list_speech)):
            # (freqs, time)
            sp_speech = np.array(np.load(list_speech[index]), dtype=np.float32)
            sp_noisy = np.array(np.load(list_noisy[index]), dtype=np.float32)
            if self._persa:
                norm_factor = 1 / np.maximum(np.amax(sp_speech), np.amax(sp_noisy))
                sp_speech, sp_noisy = sp_speech * norm_factor, sp_noisy * norm_factor
            sp_noisy = np.pad(sp_noisy, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            sp_speech = np.pad(sp_speech, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            if self._mask is 'IRM':
                sp_speech = ideal_ratio_mask(sp_noisy, sp_speech, self.mask_method)
            for k, i in enumerate(range(0, sp_speech.shape[1] - self.timeframes_in, 1)):
                x_train.append(sp_noisy[:, i: i + self.timeframes_in])
                y_train.append(sp_speech[:, i + self.timeframes_in - self.timeframes_out: i + self.timeframes_in])
        assert not np.isnan(np.sum(x_train))
        assert not np.isnan(np.sum(y_train))
        return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def load_dataset_envtfs(self, isValidation=False):
        if self._type_ == 'test':
            assert isValidation is False

        list_speech = self.list_speech_val if isValidation else self.list_speech
        list_noisy = self.list_noisy_val if isValidation else self.list_noisy

        x_train = []
        y_train = []
        # shape (BATCH_SIZE, freq, time)
        for index in range(len(list_speech)):
            # (freqs, time)
            sp_speech = np.load(list_speech[index])
            sp_noisy = np.load(list_noisy[index])
            if self._persa:
                norm_factor = 1 / np.maximum(np.amax(sp_speech), np.amax(sp_noisy))
                sp_speech, sp_noisy = sp_speech * norm_factor, sp_noisy * norm_factor

            sp_noisy = np.pad(sp_noisy, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            sp_speech = np.pad(sp_speech, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            if self._mask is 'IRM':
                sp_speech = ideal_ratio_mask(sp_noisy[:env_config['nbands']], sp_speech, self.mask_method)
            for k, i in enumerate(range(0, sp_speech.shape[1] - self.timeframes_in-1, 1)):
                x_train.append(sp_noisy[:, i: i + self.timeframes_in])
                y_train.append(sp_speech[:, i + self.timeframes_in - self.timeframes_out: i + self.timeframes_in])
            assert not np.isnan(np.sum(sp_noisy))
            assert not np.isnan(np.sum(sp_speech))
        return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

  
    def generate_test_sample_stft(self, isValidation=False, random_order=True):
        if self._type_ == 'test':
            assert isValidation is False

        # list_speech = self.list_speech_val if isValidation else self.list_speech
        # list_noisy = self.list_noisy_val if isValidation else self.list_noisy
        dir_clean = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\clean_' + self._type_ + 'set' + FOLDER_EXT
        dir_noisy = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\noisy_' + self._type_ + 'set' + FOLDER_EXT
        list_speech = [k for k in get_list_of_files(dir_clean) if '.npy' in k]
        list_noisy = [k for k in get_list_of_files(dir_noisy) if '.npy' in k]
        list_speech.sort()  # Sort the files
        list_noisy.sort()  #
        if random_order:
            indices = np.random.permutation(len(list_speech))
            list_speech = np.array(list_speech)[indices]
            list_noisy = np.array(list_noisy)[indices]
        index = 0
        while True:
            sp_speech = np.array(np.load(list_speech[index]), dtype=np.float32)
            sp_noisy = np.array(np.load(list_noisy[index]), dtype=np.float32)

            if self._persa:
                norm_factor = 1 / np.maximum(np.amax(sp_speech[..., 0]), np.amax(sp_noisy[..., 0]))
                sp_speech[..., 0], sp_noisy[..., 0] = sp_speech[..., 0] * norm_factor, sp_noisy[..., 0] * norm_factor

            x_test_sample = []
            y_test_sample = []
            pass_sample = False

            sp_noisy = np.pad(sp_noisy, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            sp_speech = np.pad(sp_speech, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            for k, i in enumerate(range(0, sp_speech.shape[1] - self.timeframes_in, 1)):
                x_test_sample.append(sp_noisy[BIN_START:BIN_STOP, i: i + self.timeframes_in, :])
                y_test_sample.append(sp_speech[BIN_START:BIN_STOP, i + self.timeframes_in - self.timeframes_out: i + self.timeframes_in, :])
                pass_sample = True
            if pass_sample:
                x_test_sample = np.array(x_test_sample, dtype=np.float32)
                y_test_sample = np.array(y_test_sample, dtype=np.float32)
                if self._mask is 'IRM':
                    y_test_sample[..., 0] = ideal_ratio_mask(x_test_sample[:, :, self.timeframes_in - self.timeframes_out:, 0],
                                                             y_test_sample[:, :, :, 0], self.mask_method)
                if not self._magphase:
                    # drop the phase and prepare shape for the model
                    y_test_sample = np.expand_dims(np.array(y_test_sample)[..., 0], axis=-1)
                    y_test_sample = np.expand_dims(np.array(y_test_sample)[..., 0], axis=-1)
                assert not np.isnan(np.sum(x_test_sample))
                assert not np.isnan(np.sum(y_test_sample))
                yield list_speech[index], np.array(x_test_sample, dtype=np.float32), np.array(y_test_sample, dtype=np.float32)
            index += 1

    def generate_test_sample_env(self, isValidation=False, random_order=True):
        if self._type_ == 'test':
            assert isValidation is False

        dir_clean = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\clean_' + self._type_ + 'set' + FOLDER_EXT
        dir_noisy = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\noisy_' + self._type_ + 'set' + FOLDER_EXT
        list_speech = [k for k in get_list_of_files(dir_clean) if '.npy' in k]
        list_noisy = [k for k in get_list_of_files(dir_noisy) if '.npy' in k]
        list_speech.sort()  # Sort the files
        list_noisy.sort()  #
        if random_order:
            indices = np.random.permutation(len(list_speech))
            list_speech = np.array(list_speech)[indices]
            list_noisy = np.array(list_noisy)[indices]
        index = 0
        while True:
            sp_speech = np.array(np.load(list_speech[index]), dtype=np.float32)
            sp_noisy = np.array(np.load(list_noisy[index]), dtype=np.float32)

            if self._persa:
                norm_factor = 1 / np.maximum(np.amax(sp_speech[..., 0]), np.amax(sp_noisy[..., 0]))
                sp_speech[..., 0], sp_noisy[..., 0] = sp_speech[..., 0] * norm_factor, sp_noisy[..., 0] * norm_factor

            x_test_sample = []
            y_test_sample = []
            pass_sample = False

            sp_noisy = np.pad(sp_noisy, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            sp_speech = np.pad(sp_speech, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))

            if self._mask is 'IRM':
                sp_speech = ideal_ratio_mask(sp_noisy, sp_speech, self.mask_method)
            for k, i in enumerate(range(0, sp_speech.shape[1] - self.timeframes_in, 1)):
                x_test_sample.append(sp_noisy[:, i: i + self.timeframes_in])
                y_test_sample.append(sp_speech[:, i + self.timeframes_in - self.timeframes_out: i + self.timeframes_in])
                pass_sample = True
            if pass_sample:
                assert not np.isnan(np.sum(x_test_sample))
                assert not np.isnan(np.sum(y_test_sample))
                yield list_speech[index], np.array(x_test_sample, dtype=np.float32), np.array(y_test_sample, dtype=np.float32)
            index += 1

    def generate_test_sample_envtfs(self, isValidation=False, random_order=True):
        if self._type_ == 'test':
            assert isValidation is False

        dir_clean = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\clean_' + self._type_ + 'set' + FOLDER_EXT
        dir_noisy = FOLDER_DATASETS + 'SE_Datasets\\' + self._dataset_ + '\\noisy_' + self._type_ + 'set' + FOLDER_EXT
        list_speech = [k for k in get_list_of_files(dir_clean) if '.npy' in k]
        list_noisy = [k for k in get_list_of_files(dir_noisy) if '.npy' in k]
        list_speech.sort()  # Sort the files
        list_noisy.sort()  #
        if random_order:
            indices = np.random.permutation(len(list_speech))
            list_speech = np.array(list_speech)[indices]
            list_noisy = np.array(list_noisy)[indices]
        index = 0
        while True:
            sp_speech = np.load(list_speech[index])
            sp_noisy = np.load(list_noisy[index])

            x_test_sample = []
            y_test_sample = []
            pass_sample = False

            sp_noisy = np.pad(sp_noisy, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            sp_speech = np.pad(sp_speech, ((0, 0), (self.timeframes_in - self.timeframes_out, 0), (0, 0)))
            if self._mask is 'IRM':
                sp_speech = ideal_ratio_mask(sp_noisy[:env_config['nbands'],:], sp_speech, self.mask_method)
            for k, i in enumerate(range(0, sp_speech.shape[1] - self.timeframes_in, 1)):
                x_test_sample.append(sp_noisy[ :, i: i + self.timeframes_in])
                y_test_sample.append(sp_speech[ :, i + self.timeframes_in - self.timeframes_out: i + self.timeframes_in])
                pass_sample = True
            if pass_sample:
                assert not np.isnan(np.sum(x_test_sample))
                assert not np.isnan(np.sum(y_test_sample))
                yield list_speech[index], np.array(x_test_sample, dtype=np.float32), np.array(y_test_sample, dtype=np.float32)
            index += 1

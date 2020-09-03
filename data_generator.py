import numpy as np
from utils import *
import matplotlib.pyplot as plt
import random
from analysis import *
import librosa
from processing import *
from config import *
import soundfile as sf
from tqdm import tqdm
from time import sleep

def mix_TIMIT_dataset(iterations=1):
    """
    Summary:  Mix clean and noisy speech audio signals
    :parameter
    threshold: threshold adapted to rms of signal
    silent_part_duration: silent parts of each utterance in the start and the end.

    Save output .wav files in /datasets/ directory
    """
    folder_name = FOLDER_DATASETS + 'SE_Datasets/TIMIT/'
    if not check_folder(folder_name):
        create_folder(folder_name)
    dir_speech_train = FOLDER_RAW_DATASETS + 'TIMIT/TRAIN'
    dir_speech_test = FOLDER_RAW_DATASETS + 'TIMIT/TEST'
    dir_noise_train = FOLDER_DATASETS + 'SE_Datasets/noise_train'
    dir_noise_test = FOLDER_DATASETS + 'SE_Datasets/noise_test'

    # Get the list of all wav files in directory tree at given path
    train_list_speech = [k for k in get_list_of_files(dir_speech_train) if '.wav' in k]
    test_list_speech = [k for k in get_list_of_files(dir_speech_test) if '.wav' in k]
    train_list_noise = [k for k in get_list_of_files(dir_noise_train) if '.wav' in k]
    test_list_noise = [k for k in get_list_of_files(dir_noise_test) if '.wav' in k]

    train_list_speech.sort()  # Sort the files
    test_list_speech.sort()
    train_list_noise.sort()
    test_list_noise.sort()

    print('Loading:', len(train_list_speech) + len(test_list_speech), 'Speech files:',
          len(train_list_speech), 'train,', len(test_list_speech), 'test')
    print('        ', len(train_list_noise), 'Train Noise files')
    print('        ', len(test_list_noise), 'Test Noise files')

    for type, list_speech, list_noise, snrs, iter in [('train', train_list_speech, train_list_noise, train_snr_dB, iterations), ('test', test_list_speech, test_list_noise, test_snr_dB, 1)]:
        if check_folder(folder_name + 'clean_' + type + 'set_wav'):
            delete_folder(folder_name + 'clean_' + type + 'set_wav')
            delete_folder(folder_name + 'noisy_' + type + 'set_wav')
        create_folder(folder_name + 'clean_' + type + 'set_wav')
        create_folder(folder_name + 'noisy_' + type + 'set_wav')
        f_total = len(list_speech)
        for it in range(iter):
            print('---' + type + ' set---')
            for f_index, filename in enumerate(list_speech):  # select a part of the data set
                # TIMIT audio files are originially in NIST format. Here we read the transformed wav files
                (fs, x) = read_audio(filename)  # x: speech signal, fs: sampling rate

                if REMOVE_SILENT:  # normalizes
                    x, _ = remove_silent_frames(x, x, fs)

                # load a random noise and slice it to the speech length
                noise_filepath = random.choice(list_noise)
                _, noise_filename = os.path.split(noise_filepath)
                fs_n, x_n = read_audio(noise_filepath)
                assert fs_n == fs == target_fs
                index_noise = random.randint(0, len(x_n) - 1 - len(x))
                noise_slice = x_n[index_noise:index_noise + len(x)]
                del x_n

                new_snr = random.choice(snrs)
                x, mix = mix_speech_and_noise(x, noise_slice, snr=new_snr)
                # new_snr = 10 * np.log10(power(x) / power(mix - x))

                # get file id
                speaker_path, pure_name = os.path.split(filename)
                speaker_id, context_id = os.path.split(speaker_path)
                _, speaker_id = os.path.split(speaker_id)
                new_filename = speaker_id + '-' + context_id + '-' + pure_name[:-5] + '_' + noise_filename[:-4] + '_%ddB' %round(new_snr)
                print('%d/%d -' % (f_index+1, f_total), filename, '\t noise: ', noise_filename[:8],
                      '\tstart: %.1fs\tsnr:' % (index_noise / fs), round(new_snr, 1), 'dB')
                # save audio files
                write_audio(folder_name + '/clean_' + type + 'set_wav/' + new_filename + '.wav', fs, x)
                write_audio(folder_name + '/noisy_' + type + 'set_wav/' + new_filename + '.wav', fs, mix)

def mix_speech_and_noise(x, x_n, snr):
    # old_power = 10*np.log10(np.max(x**2))
    x = x - x.mean()
    x_n = x_n - x_n.mean()

    if SPEECH_INITIAL_NORM:
        # Normalization of audio files to -10 to -13 dBFS
        new_rms = 10 ** (-25 / 20.0)
        scalar = new_rms / rms(x)
        x = x * scalar

    # print('Peak magnitude old: %.2f dB - new: %.2f dB'%(old_power, 10*np.log10(np.max(x**2))))
    # compute SNR between noise and speech and
    # compute the mean power of speech active regions
    power_x = librosa.feature.rms(x, frame_length=256, hop_length=128)[0] ** 2
    power_x = np.mean(power_x[power_x > np.amax(power_x) * 10 ** (-40/10)])
    noise_scalar = np.sqrt(power_x / power(x_n)) * (10**(-snr/20))
    noise_scaled = x_n * noise_scalar
    mix = np.sum([x, noise_scaled], axis=0)
    if np.amax(np.abs(mix)) >= 10**(-3/20):  # at -3 dB
        scale_factor = random.uniform(0.316, 0.5) / np.amax(np.abs(mix))
        x = x * scale_factor
        mix = mix * scale_factor
        print('normalizing...')
    return x, mix

def extract_features(dataset='TIMIT'):
    print('Extract Features', FOLDER_EXT)
    folder_name = FOLDER_DATASETS + 'SE_Datasets\\' + dataset + '\\'
    for type in ['train', 'test']:
        if check_folder(folder_name + 'clean_'+type+'set' + FOLDER_EXT):
            delete_folder(folder_name + 'clean_' + type + 'set' + FOLDER_EXT)
            delete_folder(folder_name + 'noisy_' + type + 'set' + FOLDER_EXT)
        folder_clean = create_folder(folder_name + 'clean_'+type+'set' + FOLDER_EXT)
        folder_noisy = create_folder(folder_name + 'noisy_'+type+'set' + FOLDER_EXT)

        dir_clean = os.path.join(FOLDER_DATASETS + 'SE_Datasets\\' + dataset,
                                    'clean_' + type + 'set_wav\\')
        dir_noisy = os.path.join(FOLDER_DATASETS + 'SE_Datasets\\' + dataset,
                                    'noisy_' + type + 'set_wav\\')
        filenames = [os.path.split(k)[-1] for k in get_list_of_files(dir_clean) if '.wav' in k]
        total_time_frames = 0
        ratios = []

        for i, filename in enumerate(tqdm(filenames,desc=type)):
            fs, x = read_audio(dir_clean + filename)
            fs_n, mix = read_audio(dir_noisy + filename)
            assert fs == fs_n
            assert len(x) == len(mix)

            if MODE is 'STFT':
                if PRE_EMPHASIS:
                    x = preemphasis(x)
                    mix = preemphasis(mix)
                sp_speech = specgram(x, **fft_config)
                sp_noisy = specgram(mix, **fft_config)

            elif MODE is 'ENV':
                sp_speech = envelope_spectrogram(x, fs, **env_config)[:, :, np.newaxis]
                sp_noisy = envelope_spectrogram(mix, fs, **env_config)[:, :, np.newaxis]

            elif MODE is 'ENVTFS':
                sp_speech = envelope_spectrogram(x, fs, **env_config)[:, :, np.newaxis]
                sp_noisy = envelope_spectrogram(mix, fs, **env_config)
                tfs_noisy = tfs_spectrogram(mix, fs, **env_config) # (frequency, time)
                sp_noisy = np.concatenate([sp_noisy, tfs_noisy], axis=0)[:, :, np.newaxis]

            assert not np.isnan(np.sum(sp_speech))
            assert not np.isnan(np.sum(sp_noisy))
            np.save(file=folder_clean + '/' + filename[:-4], arr=np.array(sp_speech, dtype=np.float32), allow_pickle=False)
            np.save(file=folder_noisy + '/' + filename[:-4], arr=np.array(sp_noisy, dtype=np.float32), allow_pickle=False)
            total_time_frames += sp_speech.shape[1]
        print(type, 'time frames:', total_time_frames)
    print('TIMIT extracted:', FOLDER_EXT)

if __name__ == '__main__':
    # sleep(10)
    mix_TIMIT_dataset(iterations=2)
    extract_features('TIMIT')

import os, itertools, types, random, subprocess
import utils
from helpers import mean_std
from sklearn.model_selection import ParameterGrid
from config import *
from HandleInput import DatasetGenerator
import numpy as np
from utils import read_audio, write_audio, rms
from processing import specgram, ispecgram, deemphasis, ienvelope_spectrogram, apply_mask
from time import sleep
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import matplotlib.cm as cm
from pystoi.stoi import stoi
from pypesq import pypesq
import csv
import shutil
from tqdm import tqdm
from oct2py import octave
from torchTools import *
plt.style.use('seaborn-deep')

def evaluate(folder_audio):
    results_file = os.path.join(FOLDER, 'results.csv')
    if os.path.exists(results_file):
        results_file = os.path.join(FOLDER, 'results'+os.path.split(folder_audio)[1]+'.csv')
    with open(results_file, mode='a', newline='') as csv_file:
        PR_STOIS = []
        OR_STOIS = []
        fieldnames = ['Sample', 'Speech','Noise','SNR','STOI orig.', 'STOI pred.','eSTOI orig.', 'eSTOI pred.', 'PESQ orig.', 'PESQ pred.']
        class excel_semicolon(csv.excel):
            delimiter = ';'
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect=excel_semicolon, extrasaction='ignore')
        writer.writeheader()
        sleep(0.1) # for tqdm
        pred_stois, orig_stois = [], []
        pred_estois, orig_estois = [], []
        pred_pesqs, orig_pesqs= [], []
        speech_names, noise_names = [], []
        snrs = []
        index = 0
        n = get_count_of_audiofiles(folder_audio) // 3
        for i in tqdm(range(n), total=n, desc='Calculating STOI & PESQ'):
            list_audio = [k for k in get_list_of_files(folder_audio) if '.wav' in k]
            list_audio.sort()
            assert len(list_audio) % 3 == 0
            filename = list_audio[index][:-9]
            fsx, x = read_audio(filename +'noisy.wav')
            fsy, y = read_audio(filename +'clean.wav')
            fsyh, y_hat = read_audio(filename +'predi.wav')
            x, y = x[:len(y_hat)], y[:len(y_hat)]
            assert fsx == fsy == fsyh == target_fs
            assert len(x) == len(y) == len(y_hat)

            index += 3
            # filenames
            _, f = os.path.split(filename)
            speech_noise_name = f[:-5] if f[-4] is '-' else f[:-4]
            sn = speech_noise_name.split('_')
            sn = [x.strip() for x in sn if x.strip()]
            speech_name = sn[0]
            noise_name = sn[1]
            speech_names.append(speech_name)
            noise_names.append(noise_name)
            # snr
            snr_string = f[-5:-3]
            snr = int(snr_string[1]) if snr_string[0] is '_' else int(snr_string)
            snrs.append(snr)
            # STOI
            pred_stoi = np.round(stoi(y, y_hat, target_fs), 3)
            orig_stoi = np.round(stoi(y, x, target_fs), 3)
            # eSTOI
            pred_estoi = np.round(stoi(y, y_hat, target_fs,extended=True), 3)
            orig_estoi = np.round(stoi(y, x, target_fs,extended=True), 3)
            # PESQ
            pred_pesq = np.round(pypesq(fs=target_fs, ref=y, deg=y_hat, mode='wb'), 3)
            orig_pesq = np.round(pypesq(fs=target_fs, ref=y, deg=x, mode='wb'), 3)
            # Results
            pred_stois.append(pred_stoi)
            pred_estois.append(pred_estoi)
            pred_pesqs.append(pred_pesq)
            orig_stois.append(orig_stoi)
            orig_estois.append(orig_estoi)
            orig_pesqs.append(orig_pesq)
            writer.writerow({'Sample': i,
                             'Speech': speech_name,
                             'Noise': noise_name,
                             'SNR': snr,
                             'STOI orig.': orig_stoi,
                             'STOI pred.': pred_stoi,
                             'eSTOI orig.': orig_estoi,
                             'eSTOI pred.': pred_estoi,
                             'PESQ orig.': orig_pesq,
                             'PESQ pred.': pred_pesq})
        sleep(0.15) # for tqdm

        # Results analysis with pandas
        csv_file.close()
        total_metrics = 'Orig. STOI: %s - eSTOI: %s - PESQ: %s \nPred. STOI: %s - eSTOI: %s - PESQ: %s' % \
                        (mean_std(np.array(orig_stois)), mean_std(np.array(orig_estois)), mean_std(np.array(orig_pesqs)),
                         mean_std(np.array(pred_stois)), mean_std(np.array(pred_estois)), mean_std(np.array(pred_pesqs)))
        with open(os.path.join(FOLDER, 'results_total.txt'), 'a') as file:
            file.write(total_metrics)
            file.close()
        df = pd.read_csv(results_file, sep=';')
        fig, ax = plt.subplots()
        df.groupby('Noise').mean()['STOI orig.'].plot(kind='bar', ax=ax, position=1, width=0.3, color='C0')
        df.groupby('Noise').mean()['STOI pred.'].plot(kind='bar', ax=ax, position=0, width=0.3, color='C1')
        plt.legend()
        plt.savefig(FOLDER + '/metrics_1stoi.png', dpi=600)  # , bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

        fig, ax = plt.subplots()
        df.groupby('Noise').mean()['eSTOI orig.'].plot(kind='bar', ax=ax, position=1, width=0.3, color='C0')
        df.groupby('Noise').mean()['eSTOI pred.'].plot(kind='bar', ax=ax, position=0, width=0.3, color='C1')
        plt.legend()
        plt.savefig(FOLDER + '/metrics_2estoi.png', dpi=600)  # , bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        fig, ax = plt.subplots()
        df.groupby('Noise').mean()['PESQ orig.'].plot(kind='bar', ax=ax, position=1, width=0.3, color='C0')
        df.groupby('Noise').mean()['PESQ pred.'].plot(kind='bar', ax=ax, position=0, width=0.3, color='C1')
        plt.legend()
        plt.savefig(FOLDER + '/metrics_3pesq.png', dpi=600)  # , bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        fig, ax = plt.subplots()
        df.groupby('SNR').mean()['STOI orig.'].plot(kind='bar', ax=ax, position=1, width=0.3, color='C0')
        df.groupby('SNR').mean()['STOI pred.'].plot(kind='bar', ax=ax, position=0, width=0.3, color='C1')
        plt.legend()
        plt.savefig(FOLDER + '/metrics_snr_1stoi.png', dpi=600)  # , bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        fig, ax = plt.subplots()
        df.groupby('SNR').mean()['eSTOI orig.'].plot(kind='bar', ax=ax, position=1, width=0.3, color='C0')
        df.groupby('SNR').mean()['eSTOI pred.'].plot(kind='bar', ax=ax, position=0, width=0.3, color='C1')
        plt.legend()
        plt.savefig(FOLDER + '/metrics_snr_2estoi.png', dpi=600)  # , bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        fig, ax = plt.subplots()
        df.groupby('SNR').mean()['PESQ orig.'].plot(kind='bar', ax=ax, position=1, width=0.3, color='C0')
        df.groupby('SNR').mean()['PESQ pred.'].plot(kind='bar', ax=ax, position=0, width=0.3, color='C1')
        plt.legend()
        plt.savefig(FOLDER + '/metrics_snr_3pesq.png', dpi=600)  # , bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        PR_STOIS.extend(pred_stois)
        OR_STOIS.extend(orig_stois)

        print('__________________________________________________________________________________________________')
        print('Evaluation Results: (%d files)\n' % (n))
        print(total_metrics)
        print('__________________________________________________________________________________________________')
        
    return total_metrics

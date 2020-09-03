import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import test_snr_dB
import pandas as pd
from scipy.stats import ttest_1samp


def plot_paper_results(folder_envtfs, folder_stft):
    sns.set(style="whitegrid")
    df_env = pd.read_csv('models\\' + folder_envtfs + '\\results.csv', sep=';')
    df_stft = pd.read_csv('models\\' + folder_stft + '\\results.csv', sep=';')

    df_orig = df_env.copy()
    df_orig = df_orig.drop(['eSTOI pred.'],axis=1)
    df_orig = df_orig.drop(['PESQ pred.'],axis=1)
    df_orig = df_orig.rename(columns={'eSTOI orig.':'eSTOI pred.'})
    df_orig = df_orig.rename(columns={'PESQ orig.':'PESQ pred.'})
    df_orig[' '] = 'Original'
    df_env[' '] = 'ENV-TFS'
    df_stft[' '] = 'STFT'
    df = pd.concat([df_orig, df_stft, df_env])

    sns.set(style="ticks",font='STIXGeneral')

    fig = plt.figure(figsize=(11, 4.5))
    size=16
    plt.subplot(121)
    ax = sns.boxplot(x='SNR', y='eSTOI pred.', hue=' ', data=df, fliersize=1)
    plt.xlabel('SNR (dB)', {'size': size})
    plt.ylabel('eSTOI', {'size': size})
    ax.legend_.remove()
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.8)
    ax.tick_params(labelsize=size)
    lines, labels = ax.get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper center')
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.53, 0.10), shadow = False, ncol = 3, prop={'size': size-3})
    plt.tight_layout()
    # plt.savefig('fig4.1_estoi_total.pdf',dpi=2000)
    # plt.show()

    # plt.figure(figsize=(11, 4.5))
    plt.subplot(122)
    ax = sns.boxplot(x='SNR', y='PESQ pred.', hue=' ', data=df, fliersize=1)
    ax.legend_.remove()
    ax.tick_params(labelsize=size)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol = 3)
    plt.xlabel('SNR (dB)',{'size': size})
    plt.ylabel('PESQ', {'size': size})
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.8)
    plt.tight_layout()
    plt.savefig('fig4_estoi_pesq_total.pdf',dpi=2000)
    plt.show()

    # multi plot
    sns.set(style="ticks",font='STIXGeneral',font_scale=1.3)
    g = sns.relplot(x="SNR", y="eSTOI pred.", hue = " ", col = "Noise", data = df, kind = "line",
                    col_wrap=5, height=2.5, aspect=0.8, legend='full')
    # plt.tight_layout()
    g.fig.subplots_adjust(wspace=0.10)
    g.set_ylabels('eSTOI')
    g.set_xlabels('SNR (dB)')
    g.set(xticks=[-6, 0, 6])
    g.set(xlim=(min(test_snr_dB), max(test_snr_dB)))
    g.set(ylim=(0, 1))
    g.set_titles("{col_name}",)
    # for a in g.axes:
    #     a.axhline(a.get_yticks()[1], alpha=0.5, color='grey')
    leg = g._legend
    leg.set_bbox_to_anchor([0.84, 0.86])  # coordinates of lower left of bounding box
    leg._loc = 1
    plt.savefig('fig5_estoi_per_noise.pdf',bbox_inches='tight',dpi=2000)
    plt.show()

    # eSTOI increase histogram
    plt.figure()
    ax = sns.distplot(df_env['eSTOI pred.'] - df_env['eSTOI orig.'], kde_kws={"shade": True}, norm_hist=True, label='ENV-TFS')
    sns.distplot(df_stft['eSTOI pred.'] - df_stft['eSTOI orig.'], kde_kws={"shade": True}, norm_hist=True, label='STFT')
    plt.legend()
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xlabel('eSTOI increase')
    plt.ylabel('density')
    plt.tight_layout()
    plt.show()

    # PESQ increase per snr histogram
    # ax = sns.kdeplot(df_env['SNR'], df_env['PESQ pred.'] - df_env['PESQ orig.'],  cmap="Reds", shade=True,shade_lowest=False, label='ENV')
    # sns.kdeplot(df_stft['SNR'], df_stft['PESQ pred.'] - df_stft['PESQ orig.'], cmap="Blues", shade=True,shade_lowest=False, label='STFT')

    ax = sns.distplot(df_env['PESQ pred.'] - df_env['PESQ orig.'], kde_kws={"shade": True}, norm_hist=True,
                      label='ENV-TFS')
    sns.distplot(df_stft['PESQ pred.'] - df_stft['PESQ orig.'], kde_kws={"shade": True}, norm_hist=True, label='STFT')
    plt.legend()
    vals = ax.get_xticks()
    plt.xlabel('PESQ increase')
    plt.ylabel('density')
    plt.tight_layout()
    plt.show()

    return

def plot_matlab_results(folder_envtfs, folder_stft):
    df_env1 = pd.read_excel('models\\' + folder_envtfs + '\\HA_1.xls')
    df_env2 = pd.read_excel('models\\' + folder_envtfs + '\\HA_2.xls')
    df_env3 = pd.read_excel('models\\' + folder_envtfs + '\\HA_3.xls')
    df_env4 = pd.read_excel('models\\' + folder_envtfs + '\\HA_4.xls')
    df_env5 = pd.read_excel('models\\' + folder_envtfs + '\\HA_5.xls')
    df_env6 = pd.read_excel('models\\' + folder_envtfs + '\\HA_6.xls')

    df_stft1 = pd.read_excel('models\\' + folder_stft + '\\HA_1.xls')
    df_stft2 = pd.read_excel('models\\' + folder_stft + '\\HA_2.xls')
    df_stft3 = pd.read_excel('models\\' + folder_stft + '\\HA_3.xls')
    df_stft4 = pd.read_excel('models\\' + folder_stft + '\\HA_4.xls')
    df_stft5 = pd.read_excel('models\\' + folder_stft + '\\HA_5.xls')
    df_stft6 = pd.read_excel('models\\' + folder_stft + '\\HA_6.xls')

    df_env1['Profile'] = 'HL1'
    df_env2['Profile'] = 'HL2'
    df_env3['Profile'] = 'HL3'
    df_env4['Profile'] = 'HL4'
    df_env5['Profile'] = 'HL5'
    df_env6['Profile'] = 'HL6'

    df_stft1['Profile'] = 'HL1'
    df_stft2['Profile'] = 'HL2'
    df_stft3['Profile'] = 'HL3'
    df_stft4['Profile'] = 'HL4'
    df_stft5['Profile'] = 'HL5'
    df_stft6['Profile'] = 'HL6'

    df_env = pd.concat([df_env1, df_env2, df_env3, df_env4, df_env5, df_env6])
    df_stft = pd.concat([df_stft1, df_stft2, df_stft3, df_stft4, df_stft5, df_stft6])

    df_envtemp = [df_env1, df_env2, df_env3, df_env4, df_env5, df_env6]
    df_stftemp = [df_stft1, df_stft2, df_stft3, df_stft4, df_stft5, df_stft6]

    for i in range(6):
        df = df_envtemp[i]
        dfstft = df_stftemp[i]
        print('HASPI', i+1)
        print("Origin: %.1f ± %.1f" % (100* df.mean()['HASPI_orig'], 100*df.std()['HASPI_orig']))
        print("STFT:   %.1f ± %.1f" %(100* dfstft.mean()['HASPI_predi'], 100*dfstft.std()['HASPI_predi']))
        print("ENVTFS: %.1f ± %.1f" %(100* df.mean()['HASPI_predi'], 100*df.std()['HASPI_predi']))

    for i in range(6):
        df = df_envtemp[i]
        dfstft = df_stftemp[i]
        print('HASQI', i + 1)
        print("Origin: %.1f ± %.1f" % (100 * df.mean()['HASQI_orig'], 100 * df.std()['HASQI_orig']))
        print("STFT:   %.1f ± %.1f" %(100* dfstft.mean()['HASqI_predi'], 100*dfstft.std()['HASqI_predi']))
        print("ENVTFS: %.1f ± %.1f" % (100 * df.mean()['HASqI_predi'], 100 * df.std()['HASqI_predi']))

    df_orig = df_env.copy()
    df_orig = df_orig.drop(['HASPI_predi'], axis=1)
    df_orig = df_orig.rename(columns={'HASPI_orig': 'HASPI_predi'})
    df_orig[' '] = 'Original'
    df_env[' '] = 'ENV-TFS'
    df_stft[' '] = 'STFT'
    df = pd.concat([df_orig, df_stft, df_env])

    sns.set(style="ticks", font='STIXGeneral', font_scale=1.3)
    g = sns.relplot(x="snrs", y="HASPI_predi", hue=' ', col="Profile", data=df, kind="line",
                    col_wrap=3, height=2.5, aspect=0.8, legend='full')
    # plt.tight_layout()
    g.fig.subplots_adjust(wspace=0.10)
    g.set_ylabels('HASPI')
    g.set_xlabels('SNR (dB)')
    g.set(xticks=[-6, 0, 6])
    g.set(xlim=(min(test_snr_dB), max(test_snr_dB)))
    g.set(ylim=(0, 1))
    g.set_titles("{col_name}", )
    # for a in g.axes:
    #     a.axhline(a.get_yticks()[1], alpha=0.5, color='grey')
    leg = g._legend

    leg.set_bbox_to_anchor([0.89, 0.84])  # coordinates of lower left of bounding box
    leg._loc = 1

    from matplotlib.transforms import Bbox
    plt.savefig('fig6_haspi_per_audiogram.pdf', bbox_inches=Bbox([[0., 0.], [6.8, 5.]]),dpi=2000)
    plt.show()



def print_matlab_results(folder_envtfs, folder_stft):
    df_env1 = pd.read_excel('models\\' + folder_envtfs + '\\HA_1.xls')
    df_env2 = pd.read_excel('models\\' + folder_envtfs + '\\HA_2.xls')
    df_env3 = pd.read_excel('models\\' + folder_envtfs + '\\HA_3.xls')
    df_env4 = pd.read_excel('models\\' + folder_envtfs + '\\HA_4.xls')
    df_env5 = pd.read_excel('models\\' + folder_envtfs + '\\HA_5.xls')
    df_env6 = pd.read_excel('models\\' + folder_envtfs + '\\HA_6.xls')

    df_stft1 = pd.read_excel('models\\' + folder_stft + '\\HA_1.xls')
    df_stft2 = pd.read_excel('models\\' + folder_stft + '\\HA_2.xls')
    df_stft3 = pd.read_excel('models\\' + folder_stft + '\\HA_3.xls')
    df_stft4 = pd.read_excel('models\\' + folder_stft + '\\HA_4.xls')
    df_stft5 = pd.read_excel('models\\' + folder_stft + '\\HA_5.xls')
    df_stft6 = pd.read_excel('models\\' + folder_stft + '\\HA_6.xls')

    df_env1['Profile'] = 'HA1'
    df_env2['Profile'] = 'HA2'
    df_env3['Profile'] = 'HA3'
    df_env4['Profile'] = 'HA4'
    df_env5['Profile'] = 'HA5'
    df_env6['Profile'] = 'HA6'

    df_stft1['Profile'] = 'HA1'
    df_stft2['Profile'] = 'HA2'
    df_stft3['Profile'] = 'HA3'
    df_stft4['Profile'] = 'HA4'
    df_stft5['Profile'] = 'HA5'
    df_stft6['Profile'] = 'HA6'

    df_env = pd.concat([df_env1, df_env2, df_env3, df_env4, df_env5, df_env6])
    df_stft = pd.concat([df_stft1, df_stft2, df_stft3, df_stft4, df_stft5, df_stft6])

    df_envtemp = [df_env1, df_env2, df_env3, df_env4, df_env5, df_env6]
    df_stftemp = [df_stft1, df_stft2, df_stft3, df_stft4, df_stft5, df_stft6]

    print('ENV vs STFT HASPI')
    print(df_env.mean()['HASPI_predi'], df_stft.mean()['HASPI_predi'])
    print('ENV vs STFT HASQI')
    print(df_env.mean()['HASqI_predi'], df_stft.mean()['HASqI_predi'])

    print('ENV-TFS HASPI improvement per SNR')
    print(df_env.groupby('snrs').mean()['HASPI_predi']-df_env.groupby('snrs').mean()['HASPI_orig'])
    print('STFT HASPI improvement per SNR')
    print(df_stft.groupby('snrs').mean()['HASPI_predi'] - df_stft.groupby('snrs').mean()['HASPI_orig'])

    print('ENV vs STFT HASPI differences per SNR')
    print(df_env.groupby('snrs').mean()['HASPI_predi'] - df_stft.groupby('snrs').mean()['HASPI_predi'])

    print('ENV vs STFT HASQI differences per SNR')
    print(df_env.groupby('snrs').mean()['HASqI_predi'] - df_stft.groupby('snrs').mean()['HASqI_predi'])

    print('ENV-TFS increases HASPI')
    hyp = (df_env['HASPI_predi'] - df_env['HASPI_orig'] > 0)
    print(hyp.value_counts() / (len(df_env['HASPI_predi'])))

    print('ENV-TFS increases HASQI')
    hyp = (df_env['HASqI_predi'] - df_env['HASQI_orig'] > 0)
    print(hyp.value_counts() / (len(df_env['HASqI_predi'])))

    print('ENV-TFS method gives better HASPI than STFT')
    hyp=(df_env['HASPI_predi'] - df_stft['HASPI_predi'] > 0)
    print(hyp.value_counts() / (6*len(df_env['HASPI_predi'])))

    print('ENV-TFS method gives better HASQI than STFT')
    hyp =(df_env['HASqI_predi'] - df_stft['HASqI_predi'] > 0)
    print(hyp.value_counts() / (6*len(df_env['HASqI_predi'])))



def print_stat_results(folder_envtfs, folder_stft):

    sns.set(style="whitegrid")

    df_env = pd.read_csv('models\\' + folder_envtfs + '\\results.csv', sep=';')
    df_stft = pd.read_csv('models\\' + folder_stft + '\\results.csv', sep=';')

    df_orig = df_env.copy()
    df_orig = df_orig.drop(['eSTOI pred.'], axis=1)
    df_orig = df_orig.drop(['PESQ pred.'], axis=1)
    df_orig = df_orig.rename(columns={'eSTOI orig.': 'eSTOI pred.'})
    df_orig = df_orig.rename(columns={'PESQ orig.': 'PESQ pred.'})
    df_orig[' '] = 'Original'
    df_env[' '] = 'ENV-TFS'
    df_stft[' '] = 'STFT'
    df = pd.concat([df_orig, df_stft, df_env])

    print('ENV vs STFT eSTOI')
    print(df_env.mean()['eSTOI pred.'], df_stft.mean()['eSTOI pred.'])
    print('ENV vs STFT PESQ')
    print(df_env.mean()['PESQ pred.'], df_stft.mean()['PESQ pred.'])

    print('ENV-TFS eSTOI improvement per SNR')
    print(df_env.groupby('SNR').mean()['eSTOI pred.']-df_env.groupby('SNR').mean()['eSTOI orig.'])
    print('STFT eSTOI improvement per SNR')
    print(df_stft.groupby('SNR').mean()['eSTOI pred.'] - df_stft.groupby('SNR').mean()['eSTOI orig.'])

    print('ENV vs STFT eSTOI differences per SNR')
    print(df_env.groupby('SNR').mean()['eSTOI pred.'] - df_stft.groupby('SNR').mean()['eSTOI pred.'])

    print('ENV vs STFT PESQ differences per SNR')
    print(df_env.groupby('SNR').mean()['PESQ pred.'] - df_stft.groupby('SNR').mean()['PESQ pred.'])

    print('ENV-TFS increases PESQ')
    hyp = (df_env['PESQ pred.'] - df_env['PESQ orig.'] > 0)
    print('p-value:', ttest_1samp(hyp, True)[1])
    print(hyp.value_counts() / len(df_env['eSTOI pred.']))

    print('ENV-TFS increases eSTOI')
    hyp = (df_env['eSTOI pred.'] - df_env['eSTOI orig.'] > 0)
    print('p-value:', ttest_1samp(hyp, True)[1])
    print(hyp.value_counts() / len(df_env['eSTOI pred.']))

    print('ENV-TFS method gives better PESQ than STFT')
    hyp=(df_env['PESQ pred.'] - df_stft['PESQ pred.'] > 0)
    print('p-value:', ttest_1samp(hyp, True)[1])
    print(hyp.value_counts() / len(df_env['eSTOI pred.']))

    print('ENV-TFS method gives better eSTOI than STFT')
    hyp =(df_env['eSTOI pred.'] - df_stft['eSTOI pred.'] > 0)
    print('p-value:', ttest_1samp(hyp, True)[1])
    print(hyp.value_counts() / len(df_env['eSTOI pred.']))


if __name__ == '__main__':
    folder_envtfs = '08-22_ENVTFS_635_full'
    folder_stft =  '08-24_STFT_608_full' #  '08-24_STFT_608_full'
    # print_stat_results(folder_envtfs, folder_stft)
    # plot_paper_results(folder_envtfs, folder_stft)
    plot_matlab_results(folder_envtfs, folder_stft)
    print_matlab_results(folder_envtfs, folder_stft)
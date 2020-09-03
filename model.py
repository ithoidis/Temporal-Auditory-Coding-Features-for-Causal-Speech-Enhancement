import torch
import torch.nn as nn

class Thoi_GRU(nn.Module):
    def __init__(self, mode, input_shape, output_shape, dyn_range, compression,
                 n_units, n_layers, dropout, activation_out):
        super(Thoi_GRU, self).__init__()
        self.mode = mode
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation_out = activation_out
        self.dropout_rate = dropout
        self.time_frames = input_shape[0]
        self.features = output_shape[1]
        self.compress_factor =  nn.Parameter(float(compression) * torch.ones(self.features),
                                            requires_grad=False)
        self.dyn_range = nn.Parameter(float(dyn_range) * torch.ones(self.features), requires_grad=False)

        # layers
        self.gru = nn.GRU(input_shape[-1], self.n_units, self.n_layers, bias=True, dropout=self.dropout_rate, bidirectional=False, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fcout = nn.Linear(self.n_units, self.output_shape[1])

        if self.activation_out is 'sigmoid':
            self.act_out = torch.sigmoid
        elif self.activation_out is 'hard_sigmoid':
            self.act_out = hard_sigmoid
        elif self.activation_out is 'linear':
            self.act_out = torch.nn.Identity()
        else:
            raise ValueError('Error: %s activation out method not supported. Try ' % (self.activation_out))

    def forward(self, input):
        # input shape: batch, seq, dim
        x = input[:,:,:self.output_shape[1]].contiguous() if self.mode is 'ENVTFS' else input
        x = torch.log(x + self.compress_factor.expand_as(x) ** (-self.dyn_range.expand_as(x) / 20.)) / torch.log(self.compress_factor.expand_as(x))
        if self.mode is 'ENVTFS':
            x = torch.cat([x, input[:,:,self.output_shape[1]:]],dim=-1)
    
        x, h_c = self.gru(x) #  (batch, seq, feat * dir) and (layers * dir, batch, feat)
       
        x = x[:, -1, :]
        x = self.dropout1(x)

        x = x.view(-1, self.n_units)
        x = self.fcout(x)
        x = self.act_out(x)

        out = x.view(-1, *self.output_shape)

        return out


train_snr_dB = np.linspace(-6, 2, 16+1)
test_snr_dB = [-8, -6, -4, -2, 0, 2, 4, 6]
REMOVE_SILENT = True
SPEECH_INITIAL_NORM = False      # Normalize whole speech files before any processing stage from -10 to -13 dB
target_fs = 16000                # Hz
N_FFT, HOP_LENGTH = 512, 256
PRE_EMPHASIS = True

MODE = 'STFT'

if MODE is 'STFT':
    BIN_START = 2
    BIN_STOP = 192
    MASK = 'IRM'
    fft_bins = int((N_FFT / 2) + 1)
    fft_config = dict(n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', center=True, dphase=False)
    FOLDER_EXT = '_%s_%d_%d' % (MODE, N_FFT, HOP_LENGTH)

if MODE in ['ENV', 'ENVTFS', 'AUD']:
    env_config = dict(fmin=80, fmax=6000, nbands=128, tau_ms=8, use_hilbert=False)
    MASK = 'IRM'
    fft_bins = env_config['nbands']
    BIN_START, BIN_STOP = 0, env_config['nbands']
    FOLDER_EXT = '_%s_%d_%d_%d_%d' % (MODE, env_config['fmin'], env_config['fmax'], env_config['nbands'], env_config['tau_ms'])


params = dict(get_model='GRU',
                  tframes=(10, 1),
                  dyn_range=60,
                  compression=10,
                  batch_size=2048,
                  lr=1e-4,
                  min_delta=5e-4,
                  patience=10,
                  shuffle_val=False,
                  n_layers=2,
                  n_units=512,
                  dropout=0.4,
                  activation_out='sigmoid',
                  loss_function='mse')
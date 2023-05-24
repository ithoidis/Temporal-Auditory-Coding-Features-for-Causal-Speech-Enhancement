# Temporal Auditory Coding Features for Causal Speech Enhancement
code and detailed results for the paper:

Thoidis, I.; Vrysis, L.; Markou, D.; Papanikolaou, G. Temporal Auditory Coding Features for Causal Speech Enhancement. Electronics 2020, 9, 1698.
https://www.mdpi.com/2079-9292/9/10/1698#cite

## Abstract
Perceptually motivated audio signal processing and feature extraction have played a key role in the determination of high-level semantic processes and the development of emerging systems and applications, such as mobile phone telecommunication and hearing aids. In the era of deep learning, speech enhancement methods based on neural networks have seen great success, mainly operating on the log-power spectra. Although these approaches surpass the need for exhaustive feature extraction and selection, it is still unclear whether they target the important sound characteristics related to speech perception. In this study, we propose a novel set of auditory-motivated features for single-channel speech enhancement by fusing temporal envelope and temporal fine structure information in the context of vocoder-like processing. A causal gated recurrent unit (GRU) neural network is employed to recover the low-frequency amplitude modulations of speech. Experimental results indicate that the exploited system achieves considerable gains for normal-hearing and hearing-impaired listeners, in terms of objective intelligibility and quality metrics. The proposed auditory-motivated feature set achieved better objective intelligibility results compared to the conventional log-magnitude spectrogram features, while mixed results were observed for simulated listeners with hearing loss. Finally, we demonstrate that the proposed analysis/synthesis framework provides satisfactory reconstruction accuracy of speech signals.


### Non-stationary Gabor transform implementation

Make sure that you have installed the [ltfatpy](https://pypi.org/project/ltfatpy/) library and the [NSGT filterbank implementation](https://github.com/nerrull/Adeli-Timbre-Hierarchical-Model). 


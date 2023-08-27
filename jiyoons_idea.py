import librosa
import os
import soundfile
import numpy as np


sr = 22050
n_fft = 1024
hop_length = 256
n_mels = 80

mushroom_kingdom_diffwav = np.load('/zooper2/esther.whang/mario/preprocss/mario2.wav.spec.npy')
mushroom_kingdom_talknet , sr = librosa.load('/zooper2/esther.whang/mario/mario2_TalkNet.wav')
S = librosa.feature.melspectrogram(mushroom_kingdom_talknet, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
print(mushroom_kingdom_diffwav[1,150])
print(S_DB[1,0])